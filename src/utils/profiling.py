from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch


def gpu_mem_mb(device: torch.device) -> dict:
    """Memory stats — returns zeros on CPU."""
    if device.type != "cuda":
        return {
            "alloc_mb": 0.0,
            "reserved_mb": 0.0,
            "max_alloc_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
            "util_frac": 0.0,
        }
    free, total = torch.cuda.mem_get_info(device)
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    return {
        "alloc_mb": alloc / (1024**2),
        "reserved_mb": reserved / (1024**2),
        "max_alloc_mb": max_alloc / (1024**2),
        "free_mb": free / (1024**2),
        "total_mb": total / (1024**2),
        "util_frac": 1.0 - (free / total),
    }


def estimate_tflops(
    num_params_trainable: int,
    seq_len: int,
    batch_size: int,
    step_time_s: float,
    world_size: int = 1,
) -> float:
    """
    Rough transformer training FLOP estimate (not exact hardware TFLOPs).
    """
    if step_time_s <= 0:
        return 0.0
    tokens = batch_size * seq_len * world_size
    flops = 6.0 * num_params_trainable * tokens
    return (flops / step_time_s) / 1e12


@contextmanager
def maybe_profile(enabled: bool, profile_dir: str, wait=1, warmup=1, active=3, repeat=1):
    if not enabled:
        yield None
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        yield prof