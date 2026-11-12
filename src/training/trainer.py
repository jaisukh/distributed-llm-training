from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_scheduler

from src.config import TrainConfig
from src.training.checkpointing import save_checkpoint
from src.training.distributed import DistInfo, barrier, is_dist_available_and_initialized
from src.utils.logging_utils import log_metrics
from src.utils.profiling import estimate_tflops, gpu_mem_mb, maybe_profile


class CustomQLoRATrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model,
        tokenizer,
        train_loader,
        eval_loader,
        train_sampler,
        dist_info: DistInfo,
        logger,
        wandb_run=None,
        total_params_for_flops: Optional[int] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_sampler = train_sampler
        self.dist = dist_info
        self.logger = logger
        self.wandb_run = wandb_run
        self.total_params_for_flops = total_params_for_flops

        # CPU training: always FP32, no mixed precision
        self.use_bf16 = False
        self.use_fp16 = False
        self.scaler = None  # No GradScaler on CPU

        # Wrap DDP if multi-GPU available (unlikely on CPU, but code path stays)
        if dist_info.world_size > 1 and dist_info.device.type == "cuda":
            model = DDP(
                model,
                device_ids=[dist_info.local_rank],
                output_device=dist_info.local_rank,
                find_unused_parameters=cfg.find_unused_parameters,
            )
        self.model = model

        # Only optimizer over trainable params (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # steps
        updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is not None:
            self.max_steps = cfg.max_train_steps
        else:
            self.max_steps = updates_per_epoch * cfg.num_epochs

        warmup_steps = int(self.max_steps * cfg.warmup_ratio)
        self.scheduler = get_scheduler(
            cfg.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )

        self.global_step = 0  # optimizer steps
        self.raw_n_params = sum(p.numel() for p in trainable_params)

    def _autocast(self):
        """No-op autocast for CPU training."""
        return nullcontext()

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def train(self):
        cfg = self.cfg
        model = self.model
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum = cfg.gradient_accumulation_steps
        running_loss = 0.0
        tokens_seen = 0
        step_t0 = time.time()

        with maybe_profile(cfg.profile, cfg.profile_dir) as prof_cm:
            profiler = prof_cm if cfg.profile else None

            for epoch in range(cfg.num_epochs):
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)

                for micro_step, batch in enumerate(self.train_loader):
                    batch = {k: v.to(self.dist.device, non_blocking=False) for k, v in batch.items()}
                    tokens_seen += int(batch["attention_mask"].sum().item())

                    is_last_micro = ((micro_step + 1) % accum == 0)

                    # Skip gradient sync on non-terminal micro-batches under DDP
                    if (
                        is_dist_available_and_initialized()
                        and hasattr(model, "no_sync")
                        and not is_last_micro
                    ):
                        sync_ctx = model.no_sync()
                    else:
                        sync_ctx = nullcontext()

                    with sync_ctx:
                        with self._autocast():
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                            )
                            loss = outputs.loss
                            # scale for accumulation
                            loss = loss / accum

                        loss.backward()

                    running_loss += loss.detach().float().item()

                    if is_last_micro:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            cfg.max_grad_norm,
                        )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                        # logging
                        if self.global_step % cfg.logging_steps == 0:
                            loss_tensor = torch.tensor(running_loss, device=self.dist.device)
                            if is_dist_available_and_initialized():
                                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                            step_time = time.time() - step_t0
                            mem = gpu_mem_mb(self.dist.device)
                            n_for_flops = self.total_params_for_flops or self.raw_n_params
                            tflops = estimate_tflops(
                                num_params_trainable=n_for_flops,
                                seq_len=batch["input_ids"].size(1),
                                batch_size=batch["input_ids"].size(0) * accum,
                                step_time_s=max(step_time, 1e-6),
                                world_size=self.dist.world_size,
                            )
                            metrics = {
                                "train/loss": loss_tensor.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch + (micro_step + 1) / max(len(self.train_loader), 1),
                                "system/step_time_s": step_time,
                            }
                            # Add VRAM metrics only if CUDA is available
                            if self.dist.device.type == "cuda":
                                metrics.update({
                                    "system/vram_alloc_mb": mem["alloc_mb"],
                                    "system/vram_reserved_mb": mem["reserved_mb"],
                                    "system/vram_util": mem.get("util_frac", 0.0),
                                })
                            log_metrics(metrics, self.global_step, self.wandb_run, self.logger)
                            running_loss = 0.0
                            step_t0 = time.time()

                        if self.global_step % cfg.eval_steps == 0:
                            self.evaluate()
                            model.train()

                        if self.global_step % cfg.save_steps == 0:
                            barrier()
                            save_checkpoint(
                                cfg.output_dir,
                                self.global_step,
                                model,
                                self.optimizer,
                                self.scheduler,
                                self.scaler,
                                meta={"global_step": self.global_step, "epoch": epoch},
                                is_main=self.dist.is_main,
                            )
                            barrier()

                        if profiler is not None:
                            profiler.step()

                        if self.global_step >= self.max_steps:
                            break

                if self.global_step >= self.max_steps:
                    break

        # final save
        barrier()
        save_checkpoint(
            cfg.output_dir,
            self.global_step,
            model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            meta={"global_step": self.global_step, "final": True},
            is_main=self.dist.is_main,
        )
        barrier()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.dist.device)
        total_count = torch.tensor(0.0, device=self.dist.device)

        for batch in self.eval_loader:
            batch = {k: v.to(self.dist.device, non_blocking=False) for k, v in batch.items()}
            with self._autocast():
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            total_loss += out.loss.detach().float()
            total_count += 1.0

        if is_dist_available_and_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

        mean_loss = (total_loss / total_count.clamp_min(1.0)).item()
        ppl = math.exp(min(mean_loss, 20))  # clamp to avoid overflow
        log_metrics(
            {"eval/loss": mean_loss, "eval/perplexity": ppl},
            self.global_step,
            self.wandb_run,
            self.logger,
        )
        return mean_loss