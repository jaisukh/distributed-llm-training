from __future__ import annotations

from typing import Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import TrainConfig


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype string: {name}")


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_qlora_model(cfg: TrainConfig, device_map: str | dict | None = None):
    """
    Load base model and attach LoRA adapters.
    For CPU training, 4-bit quantization is disabled (use standard FP32).
    """
    compute_dtype = _dtype_from_str(cfg.bnb_4bit_compute_dtype)

    bnb_config = None
    if cfg.load_in_4bit and torch.cuda.is_available():
        # Only use 4-bit if CUDA is available
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # CPU: load in FP32, no quantization
    torch_dtype = torch.float32
    if cfg.load_in_4bit and torch.cuda.is_available():
        torch_dtype = None  # bnb handles this

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pad_token_id = model.config.eos_token_id

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Keep LoRA adapters in FP32 (stable on CPU)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Frozen base should not have grads. PEFT already sets this; assert.
    trainable, frozen = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()

    return model, trainable, frozen


def print_trainable_summary(model, is_main: bool = True) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(
            f"Trainable params: {trainable:,} || "
            f"All params: {total:,} || "
            f"Trainable%: {100 * trainable / max(total, 1):.4f}"
        )
    return trainable, total