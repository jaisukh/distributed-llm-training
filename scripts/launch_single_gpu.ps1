# CPU training launch script for Windows
# Run from the project root directory

$env:WANDB_MODE = "offline"

python -m src.train `
    --config configs/train_config.yaml `
    --max_train_steps 20 `
    --use_wandb false `
    --device cpu