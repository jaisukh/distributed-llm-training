import torch

from src.data.dataset import collate_causal_lm


def test_collate_single():
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        }
    ]
    batch = collate_causal_lm(features, pad_token_id=0)
    assert batch["input_ids"].shape == (1, 3)
    assert batch["labels"].shape == (1, 3)
    assert (batch["labels"] == batch["input_ids"]).all()


def test_collate_padding():
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([4, 5]),
        },
    ]
    batch = collate_causal_lm(features, pad_token_id=0)
    assert batch["input_ids"].shape == (2, 3)
    # Second sequence should be padded
    assert batch["input_ids"][1, 2].item() == 0
    # Labels should be -100 on padded positions
    assert batch["labels"][1, 2].item() == -100
    # attention_mask should be 0 on padded positions
    assert batch["attention_mask"][1, 2].item() == 0


def test_collate_empty():
    """Empty batch should be handled gracefully."""
    batch = collate_causal_lm([], pad_token_id=0)
    assert batch["input_ids"].shape == (0, 0)
    assert batch["labels"].shape == (0, 0)
    assert batch["attention_mask"].shape == (0, 0)