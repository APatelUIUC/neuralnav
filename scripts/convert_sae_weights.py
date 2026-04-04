#!/usr/bin/env python3
"""
Phase 0 — Script 2: Convert SAE weights to float16 .npz files.

Downloads SAE weights from jbloom/GPT2-Small-SAEs-Reformatted on HuggingFace,
converts each layer's safetensors file to a compact float16 .npz archive, and
also extracts GPT-2's unembedding matrix W_U.

Usage:
    python convert_sae_weights.py
    python convert_sae_weights.py --output-dir ../apps/server/data/sae_weights --layers 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM


SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_KEYS = ["W_enc", "b_enc", "W_dec", "b_dec"]
EXPECTED_SHAPES = {
    "W_enc": (768, 24576),
    "b_enc": (24576,),
    "W_dec": (24576, 768),
    "b_dec": (768,),
}


# ---------------------------------------------------------------------------
# SAE weights
# ---------------------------------------------------------------------------

def convert_sae_layer(layer: int, output_dir: Path) -> None:
    """Download and convert a single SAE layer's weights."""

    filename = f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"

    print(f"  Downloading {filename} ...")
    local_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=filename,
    )

    tensors = load_file(local_path)

    # Validate keys and shapes
    for key in SAE_KEYS:
        if key not in tensors:
            raise KeyError(f"Missing key '{key}' in {filename}. Found: {list(tensors.keys())}")
        actual = tuple(tensors[key].shape)
        expected = EXPECTED_SHAPES[key]
        if actual != expected:
            raise ValueError(f"{key} shape mismatch: expected {expected}, got {actual}")

    # Convert to float16 numpy arrays
    arrays = {key: tensors[key].to(torch.float16).numpy() for key in SAE_KEYS}

    out_path = output_dir / f"layer_{layer}.npz"
    np.savez_compressed(str(out_path), **arrays)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Unembedding matrix
# ---------------------------------------------------------------------------

def extract_unembedding(output_dir: Path) -> None:
    """Extract and save GPT-2's unembedding matrix W_U."""

    print("Loading openai-community/gpt2 for unembedding matrix ...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    # lm_head.weight has shape (50257, 768) — we want (768, 50257)
    W_U = model.lm_head.weight.detach().T  # (768, 50257)
    assert W_U.shape == (768, 50257), f"Unexpected W_U shape: {W_U.shape}"

    W_U_fp16 = W_U.to(torch.float16).numpy()
    out_path = output_dir / "W_U.npy"
    np.save(str(out_path), W_U_fp16)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved W_U.npy ({size_mb:.1f} MB) — shape {W_U_fp16.shape}, dtype {W_U_fp16.dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SAE weights to float16 .npz files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "apps" / "server" / "data" / "sae_weights",
        help="Directory for output weight files (default: ../apps/server/data/sae_weights)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Layer indices to convert (default: 0-11)",
    )
    parser.add_argument(
        "--skip-unembed",
        action="store_true",
        help="Skip extraction of W_U unembedding matrix",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}\n")

    # Convert SAE layers
    for layer in tqdm(args.layers, desc="SAE layers"):
        try:
            convert_sae_layer(layer, args.output_dir)
        except Exception as e:
            print(f"\n  ERROR on layer {layer}: {e}", file=sys.stderr)
            print("  Continuing with remaining layers ...\n", file=sys.stderr)

    # Unembedding matrix
    if not args.skip_unembed:
        print()
        extract_unembedding(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
