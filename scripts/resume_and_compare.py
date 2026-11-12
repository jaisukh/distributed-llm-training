"""
Resume training from a checkpoint for 20 more steps and compare with the first run.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.training.checkpointing import load_trainer_state
from src.training.distributed import DistInfo
from src.utils.logging_utils import setup_logger


def main():
    # --- Config ---
    cfg = load_config(
        "configs/train_config.yaml",
        overrides={
            "max_train_steps": 40,  # total steps (20 old + 20 new)
            "device": "cpu",
            "use_wandb": False,
            "logging_steps": 1,
        },
    )

    logger = setup_logger("resume", is_main=True)
    dist_info = DistInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), is_main=True)

    ckpt_dir = Path("outputs/cpu-qlora-run1/checkpoint-20")
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint not found at {ckpt_dir}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("RESUME: Loading base model + LoRA adapters from checkpoint-20")
    logger.info("=" * 70)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load base model ---
    logger.info(f"Loading base model: {cfg.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = base_model.config.eos_token_id

    # --- Load LoRA adapters from checkpoint ---
    logger.info(f"Loading LoRA adapters from: {ckpt_dir}")
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=True)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / Total: {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # --- Data ---
    train_loader, eval_loader, train_sampler = build_dataloaders(cfg, tokenizer, dist_info)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Eval: {len(eval_loader.dataset)}")

    # --- Optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    # --- Scheduler ---
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    max_steps = cfg.max_train_steps  # 40 total
    warmup_steps = int(max_steps * cfg.warmup_ratio)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # --- Load trainer state (optimizer, scheduler, step) ---
    resumed_step = load_trainer_state(
        str(ckpt_dir / "trainer_state.pt"),
        optimizer,
        scheduler,
        scaler=None,
    )
    logger.info(f"Resumed from step {resumed_step}")

    # --- Train for 20 more steps (steps 21-40) ---
    logger.info("=" * 70)
    logger.info(f"TRAINING: Steps {resumed_step+1} → {max_steps} (20 more steps)")
    logger.info("=" * 70)

    accum = cfg.gradient_accumulation_steps
    global_step = resumed_step
    running_loss = 0.0
    step_t0 = time.time()

    # Store metrics for comparison
    run1_steps = list(range(1, 21))
    run1_losses = [2.1647, 1.8866, 1.4004, 1.2994, 1.0263, 0.9621, 0.9201, 0.9569, 0.7265, 1.0834,
                   0.7452, 1.1010, 0.9506, 1.1288, 1.3152, 0.9807, 0.8843, 0.9481, 1.0585, 0.9663]

    run2_losses = []
    run2_times = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        for micro_step, batch in enumerate(train_loader):
            batch = {k: v.to(dist_info.device) for k, v in batch.items()}

            is_last_micro = ((micro_step + 1) % accum == 0)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / accum
            loss.backward()

            running_loss += loss.detach().float().item()

            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_time = time.time() - step_t0
                lr = scheduler.get_last_lr()[0]
                avg_loss = running_loss

                run2_losses.append(avg_loss)
                run2_times.append(step_time)

                logger.info(
                    f"step {global_step:>3} | loss: {avg_loss:.4f} | lr: {lr:.6f} | "
                    f"time: {step_time:.1f}s"
                )

                running_loss = 0.0
                step_t0 = time.time()

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    logger.info("=" * 70)
    logger.info("COMPARISON: Run 1 (steps 1-20) vs Run 2 (steps 21-40)")
    logger.info("=" * 70)

    print(f"\n{'Step':>6} | {'Run1 Loss':>10} | {'Run2 Loss':>10} | {'Δ Loss':>10} | {'Run2 Time':>10}")
    print("-" * 62)

    for i in range(20):
        r1 = run1_losses[i]
        r2 = run2_losses[i]
        delta = r2 - r1
        sign = "↑" if delta > 0 else "↓"
        print(f"{i+1:>6} | {r1:>10.4f} | {r2:>10.4f} | {delta:>+9.4f} {sign} | {run2_times[i]:>9.1f}s")

    avg_r1 = sum(run1_losses) / len(run1_losses)
    avg_r2 = sum(run2_losses) / len(run2_losses)
    best_r1 = min(run1_losses)
    best_r2 = min(run2_losses)
    avg_time = sum(run2_times) / len(run2_times)

    print("-" * 62)
    print(f"{'AVG':>6} | {avg_r1:>10.4f} | {avg_r2:>10.4f} | {avg_r2-avg_r1:>+9.4f} |")
    print(f"{'BEST':>6} | {best_r1:>10.4f} | {best_r2:>10.4f} | {best_r2-best_r1:>+9.4f} |")
    print(f"\nRun 2 avg step time: {avg_time:.1f}s  |  Total: {sum(run2_times):.0f}s")

    # --- Save final checkpoint ---
    out_dir = Path("outputs/cpu-qlora-run1/checkpoint-40")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
         "scaler": None, "step": global_step, "meta": {"global_step": global_step}},
        str(out_dir / "trainer_state.pt"),
    )
    logger.info(f"Saved checkpoint-40 to {out_dir}")


if __name__ == "__main__":
    main()