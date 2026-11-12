from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.config import TrainConfig


def setup_logger(name: str = "train", is_main: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO if is_main else logging.ERROR)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def init_wandb(cfg: TrainConfig, is_main: bool):
    if not cfg.use_wandb or not is_main:
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=cfg.to_dict(),
        dir=cfg.output_dir,
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: int, wandb_run=None, logger: Optional[logging.Logger] = None):
    if logger:
        msg = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        logger.info(f"step {step} | {msg}")
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)