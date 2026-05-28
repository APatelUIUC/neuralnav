"""Warm-start SFT for the verbalizer (AV): injected activation -> description.

Trains GPT-2-small to generate the SAE-derived target description from the
injected residual-stream activation. This is the supervised warm-start only
(FVE ~0.3-0.4 expected); the RL stage is what later makes explanations
reconstruction-faithful rather than SAE-label paraphrase.

Run (after data_gen.py has produced out/activations.npy + out/examples.jsonl):
    python train_av.py --epochs 3 --batch 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from inject import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    build_training_example,
    default_injection_scale,
    generate_description,
)

OUT = Path(__file__).resolve().parent / "out"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data():
    acts = np.load(OUT / "activations.npy")
    examples = [json.loads(l) for l in open(OUT / "examples.jsonl")]
    return acts, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)  # grad-accum micro-steps
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save", type=Path, default=OUT / "av")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    scale = default_injection_scale(model)

    acts, examples = load_data()
    acts = torch.tensor(acts, dtype=torch.float32)
    n = len(examples)
    n_val = max(1, int(n * args.val_frac))
    idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"{n} examples — {len(train_idx)} train / {len(val_idx)} val | scale {scale:.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        perm = np.random.default_rng(epoch).permutation(train_idx)
        running, seen = 0.0, 0
        opt.zero_grad()
        for j, i in enumerate(perm):
            emb, labels = build_training_example(
                model, tok, acts[i], examples[i]["target"], scale, device
            )
            loss = model(inputs_embeds=emb, labels=labels).loss
            (loss / args.batch).backward()
            running += loss.item()
            seen += 1
            if (j + 1) % args.batch == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if (j + 1) % 500 == 0:
                print(f"  epoch {epoch} step {j+1}/{len(perm)} loss {running/seen:.4f}")
                running, seen = 0.0, 0
        opt.step(); opt.zero_grad()

    # qualitative check on a few val examples
    print("\n--- val generations (greedy) ---")
    for i in val_idx[:6]:
        gen = generate_description(model, tok, acts[i], scale, device, max_new_tokens=20)
        print(f"  target: {examples[i]['target'][:70]!r}")
        print(f"  got   : {gen[:70]!r}\n")

    args.save.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    (args.save / "nla_meta.json").write_text(json.dumps(
        {"injection_scale": scale, "layer": 8, "d_model": 768,
         "prompt_prefix": PROMPT_PREFIX, "prompt_suffix": PROMPT_SUFFIX}, indent=2))
    print(f"saved AV -> {args.save}")


if __name__ == "__main__":
    main()
