from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    output_dir: str,
    step: int,
    model,
    optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    meta: Dict[str, Any],
    is_main: bool,
) -> None:
    if not is_main:
        return
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP
    raw = model.module if hasattr(model, "module") else model

    # Save only adapters (small) via PEFT
    raw.save_pretrained(ckpt_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "meta": meta,
        },
        ckpt_dir / "trainer_state.pt",
    )
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_trainer_state(path: str, optimizer, scheduler, scaler) -> int:
    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0))