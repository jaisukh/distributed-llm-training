# Distributed Training

A from-scratch **LoRA fine-tuning stack for causal language models** — built with a custom
PyTorch training loop (no HuggingFace `Trainer`), manual Distributed Data Parallel (DDP)
support, and designed to be **CPU-friendly** so you can fine-tune small models (<1.5B params,
e.g. TinyLlama-1.1B) on a laptop while still learning every internal detail: gradient
accumulation, loss scaling, gradient clipping, distributed sampling, checkpoint/resume.

The same code path scales to **multi-GPU DDP** via `torchrun` — nothing about the loop changes.

---

## Why this project?

Most fine-tuning tutorials hand you `Trainer.train()` and hide everything interesting.
This stack reimplements the pieces deliberately:

- **Custom training loop**: `forward → loss → backward → clip → step → zero_grad`
- **Correct gradient accumulation math**: `loss / N` per micro-batch + `model.no_sync()` on
  non-terminal micro-batches under DDP so gradients are only synced once per optimizer step
- **Manual DDP wiring**: `DistributedSampler`, rank-aware seeding, barrier-synchronized
  checkpointing, cross-rank loss averaging via `all_reduce`
- **CPU as a first-class device**: FP32 everywhere, no quantization/mixed-precision required,
  `num_workers=0`, gloo backend fallback when CUDA is absent

## Features

| Feature | Details |
|---|---|
| Custom training loop | No HF `Trainer`; full visibility into every step |
| LoRA adapters | Via [PEFT](https://github.com/huggingface/peft); adapters kept in FP32 for stability; targets all 7 attention+MLP projections |
| Gradient accumulation | Correct `loss / accum_steps` scaling with DDP `no_sync()` on non-terminal micro-batches |
| Gradient clipping | `clip_grad_norm_` over trainable (LoRA) parameters only |
| Manual DDP support | Single-process CPU or multi-GPU `torchrun` launch — same entry point |
| Checkpointing | LoRA adapters + optimizer/scheduler state + metadata, saved from rank 0 only |
| Resume & compare | Script to resume from a checkpoint, train further steps, and diff losses run-over-run |
| Inference | Interactive / single-prompt generation from any saved adapter checkpoint |
| Logging | Console logging (main rank) + optional Weights & Biases |
| Profiling | Optional `torch.profiler` traces exportable to TensorBoard; VRAM stats on CUDA; rough TFLOP estimate |
| Unit tests | Grad-accumulation math, config validation, collator behavior |

## Architecture

```
configs/train_config.yaml          src/config.py
        │                                │  TrainConfig dataclass (~40 knobs) + YAML load + CLI overrides
        ▼                                ▼
   ┌─────────────────────── src/train.py ─────────────────────────┐
   │  setup_distributed() → set_seed(seed + rank)                 │
   │  build_tokenizer() ──► src/model/load_qlora.py               │
   │      base model (FP32 on CPU / bnb 4-bit on CUDA)            │
   │      + PEFT LoRA wrappers + gradient checkpointing           │
   │  build_dataloaders() ─► src/data/dataset.py                  │
   │      Alpaca prompt formatting → tokenize → dynamic padding   │
   │      (pad tokens on inputs, -100 labels on pads)             │
   │  CustomQLoRATrainer ───► src/training/trainer.py             │
   │      micro-batch fwd/bwd (+loss/accum, no_sync)              │
   │      every accum steps: clip → AdamW step → cosine schedule  │
   │      eval loop → loss + perplexity (all_reduce across ranks) │
   │      checkpoints ─────► src/training/checkpointing.py        │
   └──────────────────────────────────────────────────────────────┘
```

Supporting utilities:

- `src/training/distributed.py` — reads `torchrun` env vars (`WORLD_SIZE`, `RANK`,
  `LOCAL_RANK`), initializes the process group (`gloo` by default), exposes
  `DistInfo`, `barrier()`, graceful single-process fallback
- `src/utils/logging_utils.py` — main-rank console logger + optional W&B
- `src/utils/profiling.py` — memory stats (zeros on CPU), TFLOPs heuristic, profiler context manager

## Project Structure

```
├── configs/
│   └── train_config.yaml         # All hyperparameters (mirrors TrainConfig defaults)
├── scripts/
│   ├── launch_single_gpu.ps1     # Windows CPU smoke-test launcher
│   ├── launch_ddp.ps1            # Windows 2-GPU torchrun launcher
│   ├── launch_ddp.sh             # Linux multi-GPU torchrun launcher (NPROC / CUDA_VISIBLE_DEVICES)
│   ├── inference.py              # Generate from a trained LoRA checkpoint (single prompt or REPL)
│   ├── resume_and_compare.py     # Resume from checkpoint-20 → 40 steps and print a loss comparison table
│   └── print_comparison.py       # Static print of that comparison table
├── src/
│   ├── train.py                  # Entry point
│   ├── config.py                 # Typed config, YAML loading, CLI overrides, validation
│   ├── data/dataset.py           # Alpaca formatting, tokenization, dynamic-padding collator, DistributedSampler
│   ├── model/load_qlora.py       # Base-model loading (+optional bitsandbytes 4-bit) and LoRA attachment
│   ├── training/
│   │   ├── distributed.py        # Process-group setup from torchrun env vars
│   │   ├── trainer.py            # The custom training/eval loop
│   │   └── checkpointing.py      # Adapter + trainer-state save/load
│   └── utils/
│       ├── logging_utils.py      # Console logger + W&B integration
│       └── profiling.py          # Memory/TFLOP helpers, torch.profiler wrapper
├── tests/
│   ├── test_loss_scaling.py      # loss/accum math, effective-batch math, clip no-op check
│   ├── test_config.py            # Defaults + validation rules
│   └── test_dataset_collate.py   # Collator shapes / label masking
├── requirements.txt
├── data/.gitkeep                 # Your local datasets live here (git-ignored)
└── outputs/.gitkeep              # Checkpoints land here (git-ignored)
```

## Requirements

- **Python 3.10–3.11** (tested on 3.11)
- **~6 GB free disk** for the first-time download of TinyLlama-1.1B weights + Alpaca dataset
- **≥8 GB RAM** recommended for FP32 CPU training of a 1.1B model at `max_seq_length=512`
- Internet access on first run (HuggingFace Hub pulls model + dataset automatically)
- Multi-GPU mode additionally needs CUDA GPUs + a matching PyTorch build;
  4-bit QLoRA additionally needs `bitsandbytes` (CUDA-only — not in `requirements.txt`
  since it is disabled on CPU)

## Setup

```bash
# 1) Clone
git clone https://github.com/<your-username>/Distributed-Training.git
cd Distributed-Training

# 2) Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the unit tests to verify the install
python -m pytest tests/ -v
```

Optional extras:

```bash
pip install bitsandbytes      # only needed for 4-bit QLoRA on CUDA
wandb login                   # only if you enable use_wandb: true
```

## Running

### 1. CPU smoke test (20 steps, ~15–30 min)

```bash
python -m src.train --config configs/train_config.yaml --max_train_steps 20 --use_wandb false --device cpu
```

On Windows you can instead use the launcher (also sets `WANDB_MODE=offline`):

```powershell
.\scripts\launch_single_gpu.ps1
```

What happens on first run:
1. Downloads TinyLlama-1.1B-Chat-v1.0 (~2.2 GB in safetensors format, loaded FP32 ≈ 4.4 GB RAM)
   and the Alpaca instruction dataset from the HuggingFace Hub
2. Prints a trainable-parameter summary:

   ```
   Trainable params: ~7M || All params: ~1.1B || Trainable%: ~0.63%
   ```
3. Logs one line per step (with `logging_steps: 1`):

   ```
   INFO - step 10 | train/loss: 1.1030 | train/lr: 0.000250 | system/step_time_s: 52.3
   ```

Loss should drop from ~2.2 toward ~1.0 within the first 20–40 steps on Alpaca.

CLI overrides available without editing YAML:
`--config`, `--model_name_or_path`, `--output_dir`, `--max_train_steps`, `--use_wandb true|false`,
`--profile true|false`, `--device cpu|cuda`.

### 2. Full training run

Edit `configs/train_config.yaml`: set `max_train_steps: null` (or remove the key),
then tune `num_epochs`, `learning_rate`, `gradient_accumulation_steps`. Effective batch size is:

```
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × world_size
```

Run without the `--max_train_steps` override. Evaluation (loss + perplexity) runs every
`eval_steps` updates; checkpoints every `save_steps`.

With W&B tracking enabled (`use_wandb: true`, after `wandb login`) the same metrics stream to
project `custom-qlora-cpu`.

### 3. Multi-GPU DDP (torchrun)

Linux/macOS:

```bash
# Defaults: CUDA_VISIBLE_DEVICES=0,1,2,3 ; override like below
NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_ddp.sh --max_train_steps 100
```

Windows PowerShell (2 GPUs):

```powershell
.\scripts\launch_ddp.ps1
```

How distribution works under the hood:
- Each rank gets a shard of the data via `DistributedSampler(shuffle=True, seed=cfg.seed)`
- Non-terminal micro-batches run inside `model.no_sync()`; gradients all-reduce only on the
  last micro-batch before each optimizer step (once per update, not once per forward)
- Logged loss is averaged across ranks with `dist.all_reduce(ReduceOp.AVG)`
- Checkpoints are written by rank 0, fenced with barriers so other ranks wait
- Per-rank seeds differ (`seed + rank`) for robust dropout/shuffling behavior

> **Tip:** for CUDA multi-GPU, change `backend: "gloo"` → `"nccl"` in
> `configs/train_config.yaml` for much faster collectives. Gloo keeps working (including on
> CPU for process setup/testing), just slower.
>
> **Limitation:** multi-*process* training requires CUDA. With `world_size > 1` on a
> CPU-only box, ranks will train independently *without* gradient synchronization (DDP
> wrapping is CUDA-gated). Use a single CPU process unless you have GPUs.

### 4. Checkpoints, resume, and run comparison

Each checkpoint directory contains the LoRA adapters (`adapter_model.safetensors` +
`adapter_config.json`), the full trainer state (`optimizer`, `scheduler`, scaler, step) in
`trainer_state.pt`, and a human-readable `meta.json`:

```
outputs/cpu-qlora-run1/
└── checkpoint-20/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── trainer_state.pt
    └── meta.json
```

Resume-style demo — reloads adapters (trainable), restores optimizer/scheduler state from
`checkpoint-20`, trains steps 21→40, prints a side-by-side loss/time table vs. run 1, then
saves `checkpoint-40`:

```bash
python scripts/resume_and_compare.py
```

(The round-trip is exercised through `src.training.checkpointing.load_trainer_state`.)

### 5. Inference

Interactive chat against your trained adapters:

```bash
python scripts/inference.py                          # uses outputs/cpu-qlora-run1/checkpoint-40
python scripts/inference.py --checkpoint checkpoint-20
python scripts/inference.py --prompt "Explain gravity in one paragraph"
python scripts/inference.py --temperature 0.5 --top-p 0.9 --max-new-tokens 128
```

Prompts are formatted with the same `### Instruction:` / `### Response:` template used during
training, base model + adapter merged via `PeftModel.from_pretrained`, nucleus sampling on CPU.

## Configuration Reference

Full list lives in `configs/train_config.yaml` / `src/config.py::TrainConfig` (validated on
load — e.g. rejects `bf16=true` together with `fp16=true`). Highlights:

| Group | Key | Default | Notes |
|---|---|---|---|
| Model | `model_name_or_path` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Any HF causal-LM ≤ ~2B works well |
| Model | `lora_r` / `lora_alpha` | 8 / 16 | Lower rank = faster on CPU |
| Model | `lora_target_modules` | q,k,v,o,gate,up,down_proj | Which linear layers get adapters |
| Quant | `load_in_4bit` | `false` | NF4/fp4 via bitsandbytes; requires CUDA, auto-disabled otherwise |
| Data | `dataset_name` | `tatsu-lab/alpaca` | Or a custom text field via `text_field` |
| Data | `max_seq_length` | 512 | Main CPU-speed lever alongside batch size |
| Data | `eval_ratio` | 0.02 | Deterministic seeded train/eval split |
| Data | `num_workers` | 0 | Keep 0 on Windows/CPU |
| Training | `per_device_train_batch_size` | 1 | Bump if RAM allows |
| Training | `gradient_accumulation_steps` | 4 | Loss-scaled correctly; syncs once per step |
| Training | `learning_rate` | 5e-4 | Typical for LoRA (much higher than full FT) |
| Training | `lr_scheduler_type` / `warmup_ratio` | `cosine` / 0.03 | From transformers `get_scheduler` |
| Training | `logging_steps` / `eval_steps` / `save_steps` | 1 / 200 / 200 | Step granularity |
| Dist | `backend` | `gloo` | Switch to `nccl` for CUDA multi-GPU |
| Tracking | `use_wandb` / `profile` | `false` | W&B + TensorBoard trace export |

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 | Small enough to actually finish on CPU, real enough to be interesting |
| Precision | FP32 everywhere (adapters forced FP32) | Universal compatibility incl. CPU; bf16/fp16 config hooks retained for GPU |
| 4-bit quantization | Off by default | Needs CUDA + bitsandbytes; code path kept for QLoRA proper on GPU |
| LoRA rank | 8, all projections | Small adapter footprint (<1% params), stable AdamW memory |
| Gradient accumulation | `loss/N` backward + `no_sync()` + terminal-step clip/optimize | Mathematically equals a full-size batch's average loss (unit-tested) |
| Sequence packing | None — dynamic padding only | Simpler; pad positions masked out with `-100` labels |
| Padding side | Right, pad token = EOS | Causal LM convention; attention mask excludes pads |
| Checkpoints | Adapters + optimizer/scheduler state, rank-0-only save | Adapters stay tiny; resume reproduces mid-run state exactly |
| `max_grad_norm` | 1.0 on trainable params only | Standard LoRA safety net |

## Testing

```bash
python -m pytest tests/ -v
```

Covers the grad-accumulation loss-scaling invariant, effective-batch formula, clip-gradient
no-op behavior, config defaults/validation, and the collator (shapes, `-100` label masking).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss is 0.0 | All labels are `-100` | Inspect a batch's labels; ensure the formatter emits the response text |
| Loss NaN early | LR too high | Drop `learning_rate` (try 2e-4), keep `max_grad_norm: 1.0` |
| Very slow on CPU | Big seq_len / model | Reduce `max_seq_length` to 256, keep batch size 1, stick to TinyLlama-class models |
| Out of memory (RAM) | FP32 1.1B model + long sequences | Close other apps; lower `max_seq_length`; try a smaller base model |
| Download fails at startup | No internet / HF Hub unreachable | Set `HF_ENDPOINT=https://hf-mirror.com` or pre-download the model |
| Ranks hang at start | Firewall blocking torchrun rendezvous | Allow the master port / use `--standalone` (already default) |
| Ranks diverge (CPU world>1) | DDP wrap is CUDA-gated | Expected limitation — run CPU single-process or move to GPU |
| Eval `KeyError`/'split' issues | Dataset lacks alpaca columns | Provide `text_field:` for plain-text datasets |

## Acknowledgments

- [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — the distribution primitives this project wires up by hand
- [HuggingFace Transformers / Datasets / PEFT](https://huggingface.co) — model hosting, tokenizer, LoRA implementation
- [TinyLlama](https://github.com/jzhang38/TinyLlama) — the default base model
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) — the default instruction dataset

## Roadmap ideas

- Mixed precision (bf16) on GPU; flash-attention
- Proper multi-node launches beyond `--standalone`
- Streaming/tokenized-pretokenized dataloading for larger corpora
