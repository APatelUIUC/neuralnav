"""Warm-start the reconstructor (AR): description text -> activation vector.

AR = embedding + first N GPT-2 blocks + Linear(768,768) head, read at the last
token. Trained to reconstruct the (L2-normalized) activation that produced each
description. Reports FVE = 1 - MSE/MSE_predict_mean on a held-out split — the
standard NLA quality metric (0 = predict mean, 1 = perfect direction match).

Run (after data_gen.py):
    python train_ar.py --epochs 4 --ar-layers 9 --batch 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = Path(__file__).resolve().parent / "out"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AR(nn.Module):
    """Truncated GPT-2 backbone + linear head -> reconstructed activation."""

    def __init__(self, gpt2: GPT2LMHeadModel, n_layers: int):
        super().__init__()
        t = gpt2.transformer
        self.wte, self.wpe, self.drop = t.wte, t.wpe, t.drop
        self.blocks = nn.ModuleList(list(t.h[:n_layers]))
        self.ln_f = t.ln_f
        self.head = nn.Linear(768, 768)
        nn.init.eye_(self.head.weight)  # identity init: start near the raw resid
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids):
        seq = input_ids.shape[1]
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.wte(input_ids) + self.wpe(pos))
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.ln_f(h)
        return self.head(h[:, -1, :])  # [B, 768] from last token


def normed(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ar-layers", type=int, default=9)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save", type=Path, default=OUT / "ar.pt")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    ar = AR(gpt2, args.ar_layers).to(device)

    acts = torch.tensor(np.load(OUT / "activations.npy"), dtype=torch.float32)
    examples = [json.loads(l) for l in open(OUT / "examples.jsonl")]
    n = len(examples)
    n_val = max(1, int(n * args.val_frac))
    idx = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # predict-the-mean baseline (on normalized targets) over the val set
    targets_norm = normed(acts).to(device)
    mean_dir = normed(acts[train_idx].mean(0, keepdim=True)).to(device)
    base_mse = ((mean_dir - targets_norm[val_idx]) ** 2).sum(-1).mean().item()
    print(f"{n} examples — {len(train_idx)} train / {len(val_idx)} val | "
          f"predict-mean MSE {base_mse:.4f}")

    opt = torch.optim.AdamW(ar.parameters(), lr=args.lr)

    def encode(text):
        return tok(" " + text.strip(), return_tensors="pt",
                  truncation=True, max_length=48).input_ids.to(device)

    for epoch in range(args.epochs):
        ar.train()
        perm = np.random.default_rng(epoch).permutation(train_idx)
        running, seen = 0.0, 0
        opt.zero_grad()
        for j, i in enumerate(perm):
            pred = ar(encode(examples[i]["target"]))
            loss = ((normed(pred) - targets_norm[i]) ** 2).sum(-1).mean()
            (loss / args.batch).backward()
            running += loss.item(); seen += 1
            if (j + 1) % args.batch == 0:
                torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if (j + 1) % 500 == 0:
                print(f"  epoch {epoch} step {j+1}/{len(perm)} mse {running/seen:.4f}")
                running, seen = 0.0, 0
        opt.step(); opt.zero_grad()

        # val FVE
        ar.eval()
        with torch.no_grad():
            mses = []
            for i in val_idx:
                pred = ar(encode(examples[i]["target"]))
                mses.append(((normed(pred) - targets_norm[i]) ** 2).sum(-1).mean().item())
            val_mse = float(np.mean(mses))
        fve = 1 - val_mse / base_mse
        print(f"epoch {epoch}: val MSE {val_mse:.4f}  FVE {fve:.3f}")

    torch.save({"state_dict": ar.state_dict(), "ar_layers": args.ar_layers}, args.save)
    print(f"saved AR -> {args.save}")


if __name__ == "__main__":
    main()
