"""Warm-start data generation for the GPT-2 NLA, bootstrapped from SAE labels.

For each sampled (text, token) position we extract the layer-8 residual-stream
activation, run the existing Joseph-Bloom SAE to find its top features, and
synthesize a target description from those features' Neuronpedia labels. This
gives (activation -> description) pairs to warm-start both the verbalizer (AV)
and reconstructor (AR) — for free, grounded in real interpretability data, and
uniquely available because neuralnav already ships the SAE + labels.

Labels are fetched on-demand (only for features that actually appear) and cached,
so this needs ~hundreds of API calls, not the full 24k-feature space.

Output:
    out/activations.npy   float32 [N, 768]
    out/examples.jsonl     {"target": str, "top_features": [[idx, act], ...]}

Run (after convert_sae_weights.py --layers 8 has produced layer_8.npz):
    python data_gen.py --num-docs 400 --positions-per-doc 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

LAYER = 8
HIDDEN = 768
NP_API = "https://www.neuronpedia.org/api/feature/gpt2-small"
SAE_SET = f"{LAYER}-res-jb"

REPO = Path(__file__).resolve().parent.parent.parent
SAE_NPZ = REPO / "apps" / "server" / "data" / "sae_weights" / f"layer_{LAYER}.npz"
LABEL_CACHE = Path(__file__).resolve().parent / "out" / "label_cache.json"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- SAE (matches apps/server/app/sae.py math) ---

class SAE:
    def __init__(self, npz_path: Path):
        z = np.load(npz_path)
        self.W_enc = z["W_enc"].astype(np.float32)  # [768, 24576]
        self.b_enc = z["b_enc"].astype(np.float32)  # [24576]
        self.b_dec = z["b_dec"].astype(np.float32)  # [768]

    def top_features(self, x: np.ndarray, k: int):
        z = np.maximum((x - self.b_dec) @ self.W_enc + self.b_enc, 0.0)  # ReLU
        idx = np.argsort(z)[::-1][:k]
        idx = [int(i) for i in idx if z[i] > 0]
        return [(i, float(z[i])) for i in idx]


# --- Neuronpedia label fetch (on-demand, cached) ---

def load_cache() -> dict:
    if LABEL_CACHE.exists():
        return json.loads(LABEL_CACHE.read_text())
    return {}


def fetch_labels(indices: set[int], cache: dict, rps: float = 5.0) -> dict:
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "neuralnav-nla-datagen/0.1"
    interval = 1.0 / rps
    todo = [i for i in indices if str(i) not in cache]
    print(f"fetching {len(todo)} new labels (of {len(indices)} unique features) ...")
    for n, idx in enumerate(todo):
        try:
            t0 = time.monotonic()
            r = session.get(f"{NP_API}/{SAE_SET}/{idx}", timeout=30)
            if r.status_code == 404:
                cache[str(idx)] = ""
            else:
                r.raise_for_status()
                d = r.json()
                exps = d.get("explanations") or []
                cache[str(idx)] = (exps[0].get("description", "") if exps else "").strip()
            dt = time.monotonic() - t0
            if dt < interval:
                time.sleep(interval - dt)
        except Exception:
            cache[str(idx)] = cache.get(str(idx), "")
        if n and n % 100 == 0:
            LABEL_CACHE.write_text(json.dumps(cache))
            print(f"  ... {n}/{len(todo)}")
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LABEL_CACHE.write_text(json.dumps(cache))
    return cache


def synthesize_target(top_feats, cache: dict, max_parts: int = 3) -> str | None:
    """Join the top labeled features into a short description."""
    parts = []
    for idx, _act in top_feats:
        lbl = cache.get(str(idx), "")
        if lbl:
            parts.append(lbl)
        if len(parts) >= max_parts:
            break
    if not parts:
        return None
    return "; ".join(parts)


# --- corpus ---

SAMPLE_CORPUS = [
    "The Eiffel Tower, completed in 1889, remains one of the most visited monuments in the world.",
    "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "In the fourth quarter, the quarterback threw a touchdown pass to win the championship game.",
    "The barista steamed the milk carefully before pouring a delicate rosetta into the latte.",
    "Quicksort recursively partitions an array around a pivot, achieving average n log n performance.",
    "Beethoven continued composing symphonies even after he had completely lost his hearing.",
    "The volcano erupted violently, sending ash kilometers into the sky and lava down its slopes.",
    "Markets tumbled after the central bank unexpectedly raised interest rates by half a point.",
]


def load_corpus(path: str | None, n: int) -> list[str]:
    if path:
        lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        return lines[:n]
    # repeat the small built-in sample if no file given
    out = []
    while len(out) < n:
        out.extend(SAMPLE_CORPUS)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=None, help="text file, one doc per line")
    ap.add_argument("--num-docs", type=int, default=400)
    ap.add_argument("--positions-per-doc", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()

    if not SAE_NPZ.exists():
        raise SystemExit(f"SAE weights not found at {SAE_NPZ}\n"
                         f"Run: python ../convert_sae_weights.py --layers {LAYER} --skip-unembed")

    device = get_device()
    print(f"device: {device}")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    sae = SAE(SAE_NPZ)

    docs = load_corpus(args.corpus, args.num_docs)
    print(f"corpus: {len(docs)} docs")

    acts, feats_per = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for doc in docs:
            ids = tok(doc, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
            seq = ids.shape[1]
            if seq < 3:
                continue
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0]  # [seq, 768]
            # sample positions, skip position 0 (attention sink)
            choices = rng.choice(range(1, seq), size=min(args.positions_per_doc, seq - 1), replace=False)
            for t in choices:
                x = hs[t].float().cpu().numpy()
                acts.append(x)
                feats_per.append(sae.top_features(x, args.top_k))

    print(f"extracted {len(acts)} activations")

    # fetch labels for all features that appear
    needed = {idx for fp in feats_per for idx, _ in fp}
    cache = fetch_labels(needed, load_cache())

    # synthesize targets; keep only examples with at least one labeled feature
    args.out.mkdir(parents=True, exist_ok=True)
    kept_acts, n_examples = [], 0
    with open(args.out / "examples.jsonl", "w") as f:
        for x, fp in zip(acts, feats_per):
            target = synthesize_target(fp, cache)
            if target is None:
                continue
            kept_acts.append(x)
            f.write(json.dumps({"target": target, "top_features": fp}) + "\n")
            n_examples += 1

    np.save(args.out / "activations.npy", np.array(kept_acts, dtype=np.float32))
    cov = n_examples / max(len(acts), 1)
    print(f"kept {n_examples}/{len(acts)} examples (label coverage {cov:.0%})")
    print(f"saved -> {args.out}/activations.npy + examples.jsonl")
    # show a few
    for line in (args.out / "examples.jsonl").read_text().splitlines()[:4]:
        print("  ", json.loads(line)["target"][:90])


if __name__ == "__main__":
    main()
