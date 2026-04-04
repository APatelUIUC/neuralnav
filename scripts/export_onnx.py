#!/usr/bin/env python3
"""
Phase 0 — Script 1: Export GPT-2 Small to ONNX with residual stream outputs.

Standard HuggingFace ONNX export only exposes logits. This script creates a
custom wrapper that also exposes all 13 hidden states (embedding output + 12
transformer layer outputs) as named outputs `resid_0` through `resid_12`.

After export the model is quantized to float16 for smaller file size.

Usage:
    python export_onnx.py
    python export_onnx.py --output-dir ../models --seq-len 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class GPT2WithResiduals(nn.Module):
    """Thin wrapper around HuggingFace GPT2LMHeadModel that returns logits
    plus all 13 intermediate hidden states (the residual stream)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids, output_hidden_states=True)
        logits = outputs.logits                    # [batch, seq, vocab]
        hidden_states = outputs.hidden_states      # tuple of 13 tensors, each [batch, seq, 768]
        return (logits, *hidden_states)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(output_dir: Path, seq_len: int) -> Path:
    """Export GPT2WithResiduals to ONNX (float32)."""

    print("Loading openai-community/gpt2 ...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()

    wrapper = GPT2WithResiduals(model)
    wrapper.eval()

    dummy_input = torch.randint(0, 50257, (1, seq_len), dtype=torch.long)

    # Build output names: logits + resid_0 ... resid_12
    output_names = ["logits"] + [f"resid_{i}" for i in range(13)]

    # Dynamic axes for batch and seq_len on every output
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "seq_len"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch", 1: "seq_len"}

    onnx_path = output_dir / "gpt2-resid.onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset 17) -> {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        opset_version=17,
        input_names=["input_ids"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Quick sanity check on the saved graph
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and validated ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    return onnx_path


# ---------------------------------------------------------------------------
# Float16 quantisation
# ---------------------------------------------------------------------------

def quantize_fp16(onnx_path: Path) -> Path:
    """Convert an ONNX model to float16."""
    from onnxconverter_common import float16_converter

    fp16_path = onnx_path.with_name("gpt2-resid-fp16.onnx")
    print(f"Quantizing to float16 -> {fp16_path} ...")

    model = onnx.load(str(onnx_path))
    model_fp16 = float16_converter.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))

    print(f"FP16 model saved ({fp16_path.stat().st_size / 1e6:.1f} MB)")
    return fp16_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(onnx_path: Path, seq_len: int) -> bool:
    """Run a sample input through both PyTorch and ONNX Runtime and compare."""

    print("\nValidating ONNX vs PyTorch outputs ...")

    # PyTorch reference
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    text = "The quick brown fox jumps"
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"]

    with torch.no_grad():
        pt_out = model(input_ids, output_hidden_states=True)
    pt_logits = pt_out.logits.numpy()
    pt_hidden = [h.numpy() for h in pt_out.hidden_states]

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_ids": input_ids.numpy()})

    ort_logits = ort_out[0]
    ort_hidden = ort_out[1:]

    # Compare
    logits_close = np.allclose(pt_logits, ort_logits, atol=1e-4, rtol=1e-4)
    print(f"  Logits match:  {logits_close}  (max diff {np.max(np.abs(pt_logits - ort_logits)):.2e})")

    all_ok = logits_close
    for i, (pt_h, ort_h) in enumerate(zip(pt_hidden, ort_hidden)):
        close = np.allclose(pt_h, ort_h, atol=1e-4, rtol=1e-4)
        max_diff = np.max(np.abs(pt_h - ort_h))
        if not close:
            all_ok = False
        print(f"  resid_{i} match: {close}  (max diff {max_diff:.2e})")

    if all_ok:
        print("\nAll outputs match within tolerance.")
    else:
        print("\nWARNING: Some outputs exceeded tolerance (may still be acceptable).")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 Small to ONNX with residual stream outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Directory for output ONNX files (default: ../models)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for the dummy export input (default: 64)",
    )
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="Skip float16 quantisation step",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation step",
    )
    args = parser.parse_args()

    onnx_path = export_onnx(args.output_dir, args.seq_len)

    if not args.skip_fp16:
        quantize_fp16(onnx_path)

    if not args.skip_validate:
        validate(onnx_path, args.seq_len)

    print("\nDone.")


if __name__ == "__main__":
    main()
