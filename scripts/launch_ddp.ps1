# DDP launch script (multi-GPU) for Windows
# Only works if CUDA + NCCL stack is available
# Run from the project root directory

$env:CUDA_VISIBLE_DEVICES = "0,1"

torchrun --standalone --nproc_per_node=2 `
    -m src.train `
    --config configs/train_config.yaml `
    --max_train_steps 50