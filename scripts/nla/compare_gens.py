"""Eyeball check: warm-start AV vs RL'd AV descriptions on held-out activations.

Does RL make explanations more meaningful, or just game the round-trip? Prints
the SAE-label target, the warm-start generation, and the RL'd generation side by
side for a handful of activations so we can judge by eye (precursor to the proper
independent-decoder steganography test).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from inject import generate_description

OUT = Path(__file__).resolve().parent / "out"
dev = torch.device("cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))

tok = GPT2TokenizerFast.from_pretrained(str(OUT / "av"))
acts = torch.tensor(np.load(OUT / "activations.npy"), dtype=torch.float32)
examples = [json.loads(l) for l in open(OUT / "examples.jsonl")]
scale = json.loads((OUT / "av" / "nla_meta.json").read_text())["injection_scale"]

ws = GPT2LMHeadModel.from_pretrained(str(OUT / "av")).to(dev).eval()
rl = GPT2LMHeadModel.from_pretrained(str(OUT / "av_rl")).to(dev).eval()

# held-out tail (val split used 5% = last ~300; sample a few)
for i in range(len(acts) - 8, len(acts)):
    a = acts[i]
    print(f"[{i}] SAE target : {examples[i]['target'][:85]}")
    print(f"     warm-start : {generate_description(ws, tok, a, scale, dev, 20)[:85]}")
    print(f"     RL'd       : {generate_description(rl, tok, a, scale, dev, 20)[:85]}")
    print()
