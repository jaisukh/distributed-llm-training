from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TrainConfig:
    # model
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "outputs/run"
    seed: int = 42

    # quant — disabled for CPU training
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_compute_dtype: str = "float32"

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # data
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    max_seq_length: int = 512
    text_field: Optional[str] = None
    eval_ratio: float = 0.02
    num_workers: int = 0  # 0 for CPU to avoid multiprocessing issues

    # train
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    max_train_steps: Optional[int] = None

    # precision — CPU uses FP32
    bf16: bool = False
    fp16: bool = False

    # dist — single-process for CPU
    backend: str = "gloo"
    find_unused_parameters: bool = False

    # tracking
    use_wandb: bool = False
    wandb_project: str = "custom-qlora-cpu"
    wandb_run_name: Optional[str] = None
    profile: bool = False
    profile_steps: int = 20
    profile_dir: str = "profiles"

    # CPU-specific
    device: str = "cpu"  # force cpu mode

    def validate(self) -> None:
        if self.bf16 and self.fp16:
            raise ValueError("Choose either bf16 or fp16, not both.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("bnb_4bit_quant_type must be nf4 or fp4")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})
    cfg = TrainConfig(**raw)
    cfg.validate()
    return cfg