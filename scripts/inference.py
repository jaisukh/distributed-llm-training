"""
Interactive inference using a trained LoRA checkpoint.
Usage:
    python scripts/inference.py                          # use checkpoint-40
    python scripts/inference.py --checkpoint checkpoint-20  # use checkpoint-20
    python scripts/inference.py --prompt "Explain gravity"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT_DIR = Path("outputs/cpu-qlora-run1")


def load_model(checkpoint: str):
    """Load base model + LoRA adapters from checkpoint."""
    ckpt_path = CKPT_DIR / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading LoRA adapters from: {ckpt_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate a response from the model."""
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LoRA model")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-40",
                        help="Checkpoint name (default: checkpoint-40)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt. If omitted, runs interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling (default: 0.9)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)

    if args.prompt:
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'='*60}")
        response = generate(model, tokenizer, args.prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p)
        print(f"\n{response}\n")
    else:
        print("\nInteractive mode. Type 'quit' to exit, 'reset' to clear history.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not prompt:
                continue
            if prompt.lower() == "quit":
                print("Goodbye.")
                break
            if prompt.lower() == "reset":
                print("[Conversation reset]")
                continue

            print("Generating...")
            response = generate(model, tokenizer, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p)
            print(f"\n{response}\n")
            print("-" * 60)


if __name__ == "__main__":
    main()