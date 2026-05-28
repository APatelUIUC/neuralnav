"""NLA (Natural Language Autoencoder) inference for the server.

Loads the trained verbalizer (AV) + reconstructor (AR) and exposes `explain()`:
inject an activation -> generate a natural-language description -> reconstruct
the activation from that text -> report the round-trip cosine (faithfulness).

torch/transformers are imported here only; the rest of the server (numpy SAE)
runs without them, and `load_nla` returns None gracefully if the models or the
deps are absent — so /explain stays disabled until the NLA is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _AR(nn.Module):
    """Truncated GPT-2 backbone + linear head (mirrors scripts/nla/train_ar.py)."""

    def __init__(self, gpt2: GPT2LMHeadModel, n_layers: int):
        super().__init__()
        t = gpt2.transformer
        self.wte, self.wpe, self.drop = t.wte, t.wpe, t.drop
        self.blocks = nn.ModuleList(list(t.h[:n_layers]))
        self.ln_f = t.ln_f
        self.head = nn.Linear(768, 768)

    def forward(self, input_ids):
        seq = input_ids.shape[1]
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.wte(input_ids) + self.wpe(pos))
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.ln_f(h)
        return self.head(h[:, -1, :])


class NLA:
    def __init__(self, av, tok, ar, meta, device):
        self.av, self.tok, self.ar, self.meta, self.device = av, tok, ar, meta, device


def load_nla(av_dir: Path, ar_path: Path) -> NLA | None:
    av_dir, ar_path = Path(av_dir), Path(ar_path)
    if not av_dir.exists() or not ar_path.exists():
        return None
    device = _device()
    meta = json.loads((av_dir / "nla_meta.json").read_text())
    tok = GPT2TokenizerFast.from_pretrained(str(av_dir))
    av = GPT2LMHeadModel.from_pretrained(str(av_dir)).to(device).eval()

    ckpt = torch.load(ar_path, map_location=device)
    ar = _AR(GPT2LMHeadModel.from_pretrained("gpt2"), ckpt["ar_layers"]).to(device).eval()
    ar.load_state_dict(ckpt["state_dict"])
    return NLA(av, tok, ar, meta, device)


@torch.no_grad()
def explain(nla: NLA, activation: list[float], max_new_tokens: int = 24) -> tuple[str, float]:
    d, tok, av, ar, meta = nla.device, nla.tok, nla.av, nla.ar, nla.meta
    dtype = next(av.parameters()).dtype
    act = torch.tensor(activation, dtype=torch.float32, device=d)

    # build prompt embeds: prefix + injected (normalized, scaled) activation + suffix
    wte = av.transformer.wte
    pre = tok(meta["prompt_prefix"], return_tensors="pt").input_ids.to(d)
    suf = tok(meta["prompt_suffix"], return_tensors="pt").input_ids.to(d)
    a = (act / (act.norm() + 1e-8) * meta["injection_scale"]).to(dtype).view(1, 1, -1)
    emb = torch.cat([wte(pre), a, wte(suf)], dim=1)

    out = av.generate(inputs_embeds=emb, max_new_tokens=max_new_tokens,
                      do_sample=False, pad_token_id=tok.eos_token_id)
    desc = tok.decode(out[0], skip_special_tokens=True).strip()

    # reconstruction faithfulness: AR(text) vs original activation, direction cosine
    ids = tok(" " + desc, return_tensors="pt", truncation=True, max_length=48).input_ids.to(d)
    recon = ar(ids)[0]
    cos = torch.nn.functional.cosine_similarity(
        act.unsqueeze(0), recon.unsqueeze(0)).item()
    return desc, float(cos)
