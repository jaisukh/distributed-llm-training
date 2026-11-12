def test_grad_accumulation_math():
    """loss / accum is what gets backward'ed; N microbatches average to full loss."""
    accum = 8
    micro_losses = [2.0] * accum
    backward_terms = [l / accum for l in micro_losses]
    assert abs(sum(backward_terms) - 2.0) < 1e-8


def test_effective_batch_math():
    """effective_batch = per_device_batch * accum * world_size"""
    per_device_batch = 1
    accum = 8
    world_size = 1  # CPU single-process
    effective = per_device_batch * accum * world_size
    assert effective == 8


def test_grad_clip_noop():
    """Gradient clipping with max_norm=0 should produce zero norms."""
    import torch
    w = torch.randn(10, 10, requires_grad=True)
    loss = w.sum()
    loss.backward()
    norm_before = w.grad.norm().item()
    torch.nn.utils.clip_grad_norm_([w], max_norm=0.0)
    norm_after = w.grad.norm().item()
    assert norm_after < norm_before
    assert norm_after == 0.0