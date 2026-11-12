from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.model.load_qlora import build_qlora_model, build_tokenizer, print_trainable_summary
from src.training.distributed import cleanup_distributed, setup_distributed
from src.training.trainer import CustomQLoRATrainer
from src.utils.logging_utils import init_wandb, setup_logger


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def parse_args():
    p = argparse.ArgumentParser(description="Custom QLoRA fine-tune (CPU-friendly)")
    p.add_argument("--config", type=str, default="configs/train_config.yaml")
    p.add_argument("--model_name_or_path", type=str, default=None, help="Override model name")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_train_steps", type=int, default=None, help="Override max steps")
    p.add_argument("--use_wandb", type=lambda x: str(x).lower() == "true", default=None)
    p.add_argument("--profile", type=lambda x: str(x).lower() == "true", default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "max_train_steps": args.max_train_steps,
        "use_wandb": args.use_wandb,
        "profile": args.profile,
        "device": args.device,
    }
    cfg = load_config(args.config, overrides)

    dist_info = setup_distributed(backend=cfg.backend)
    set_seed(cfg.seed, dist_info.rank)
    logger = setup_logger(is_main=dist_info.is_main)
    if dist_info.is_main:
        logger.info(f"World size={dist_info.world_size} device={dist_info.device}")
        logger.info(f"Config: {cfg.to_dict()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = build_tokenizer(cfg.model_name_or_path)

    # For CPU: no device_map needed (model stays on CPU)
    device_map = None
    if dist_info.device.type == "cuda":
        device_map = {"": dist_info.local_rank}

    logger.info(f"Loading model {cfg.model_name_or_path}...")
    model, trainable, frozen = build_qlora_model(cfg, device_map=device_map)
    print_trainable_summary(model, is_main=dist_info.is_main)

    train_loader, eval_loader, train_sampler = build_dataloaders(cfg, tokenizer, dist_info)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Eval samples: {len(eval_loader.dataset)}")

    wandb_run = init_wandb(cfg, dist_info.is_main)

    # rough full-model param count for FLOP heuristic
    total_params = sum(p.numel() for p in model.parameters())

    trainer = CustomQLoRATrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_sampler=train_sampler,
        dist_info=dist_info,
        logger=logger,
        wandb_run=wandb_run,
        total_params_for_flops=total_params,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()
        logger.info("Done.")


if __name__ == "__main__":
    main()