import os
import tempfile
from pathlib import Path

import yaml

from src.config import TrainConfig, load_config


def test_default_config():
    cfg = TrainConfig()
    assert cfg.lora_r == 8
    assert cfg.load_in_4bit is False  # Disabled for CPU
    assert cfg.device == "cpu"
    assert cfg.gradient_accumulation_steps >= 1


def test_config_validation_rejects_both_precision():
    cfg = TrainConfig(bf16=True, fp16=True)
    try:
        cfg.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not both" in str(e)


def test_config_from_yaml():
    data = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_r": 16,
        "max_seq_length": 256,
        "max_train_steps": 10,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    try:
        cfg = load_config(path)
        assert cfg.model_name_or_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.lora_r == 16
        assert cfg.max_seq_length == 256
        assert cfg.max_train_steps == 10
    finally:
        os.unlink(path)


def test_config_overrides():
    data = {"model_name_or_path": "base-model", "max_train_steps": 100}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    try:
        cfg = load_config(path, overrides={"max_train_steps": 50, "lora_r": 32})
        assert cfg.model_name_or_path == "base-model"
        assert cfg.max_train_steps == 50
        assert cfg.lora_r == 32
    finally:
        os.unlink(path)