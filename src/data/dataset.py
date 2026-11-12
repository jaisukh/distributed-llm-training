from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.config import TrainConfig


ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_alpaca_row(row: Dict[str, Any]) -> str:
    instruction = row.get("instruction", "")
    inp = row.get("input", "") or ""
    output = row.get("output", "") or row.get("response", "")
    if not inp:
        inp = ""
    return ALPACA_PROMPT.format(instruction=instruction, input=inp, output=output)


class CausalLMJsonDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


def collate_causal_lm(features: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding batch collator.
    Labels for pad positions are -100 so CrossEntropy ignores them.
    """
    if not features:
        return {
            "input_ids": torch.zeros(0, 0, dtype=torch.long),
            "attention_mask": torch.zeros(0, 0, dtype=torch.long),
            "labels": torch.zeros(0, 0, dtype=torch.long),
        }

    max_len = max(f["input_ids"].size(0) for f in features)

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - f["input_ids"].size(0)
        input_ids.append(
            torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=pad_token_id)
        )
        attention_mask.append(
            torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0)
        )
        labels.append(
            torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def load_texts(cfg: TrainConfig) -> List[str]:
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    texts: List[str] = []
    for row in ds:
        if cfg.text_field:
            texts.append(row[cfg.text_field])
        else:
            texts.append(format_alpaca_row(row))
    return texts


def build_dataloaders(cfg: TrainConfig, tokenizer, dist_info):
    texts = load_texts(cfg)
    n = len(texts)
    n_eval = max(1, int(n * cfg.eval_ratio))
    # simple deterministic split by seed shuffle
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g).tolist()
    eval_idx = set(perm[:n_eval])
    train_texts = [texts[i] for i in range(n) if i not in eval_idx]
    eval_texts = [texts[i] for i in perm[:n_eval]]

    train_ds = CausalLMJsonDataset(train_texts, tokenizer, cfg.max_seq_length)
    eval_ds = CausalLMJsonDataset(eval_texts, tokenizer, cfg.max_seq_length)

    train_sampler = None
    eval_sampler = None
    if dist_info.world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=False,
            drop_last=False,
        )

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        return collate_causal_lm(batch, pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=False,  # False for CPU
        collate_fn=collate,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.per_device_eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,  # False for CPU
        collate_fn=collate,
        drop_last=False,
    )
    return train_loader, eval_loader, train_sampler