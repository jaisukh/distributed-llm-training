from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(backend: str = "gloo") -> DistInfo:
    """
    Init process group using torchrun env vars.
    Single-process fallback for CPU training.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        # CPU single-process mode
        device = torch.device("cpu")

    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=(rank == 0),
    )


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()