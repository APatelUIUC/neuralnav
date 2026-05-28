"""Activation injection for the GPT-2 NLA verbalizer (AV).

The verbalizer reads a residual-stream activation by splicing it into the input
embedding sequence as a single "token" at a fixed slot in a prompt, then
autoregressively decoding a natural-language description. This module builds the
`inputs_embeds` tensors for both generation and supervised training.

The injected vector is L2-normalized and rescaled to the model's typical token
embedding norm, so it stays in-distribution for the first LayerNorm. Only the
direction carries information (which is exactly what reconstruction/FVE scores),
so normalizing away the raw magnitude is safe and more robust than a guessed
scalar.
"""

from __future__ import annotations

import torch

PROMPT_PREFIX = "Activation:"
PROMPT_SUFFIX = " in plain English means:"


def default_injection_scale(model) -> float:
    """Mean L2 norm of the token-embedding rows — a sensible in-distribution scale."""
    wte = model.transformer.wte.weight  # [vocab, d]
    return float(wte.norm(dim=-1).mean().item())


def _prompt_embeds(model, tokenizer, activation, injection_scale, device, dtype):
    """prefix_emb + [injected activation] + suffix_emb  ->  [1, prompt_len, d]."""
    wte = model.transformer.wte
    prefix_ids = tokenizer(PROMPT_PREFIX, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(PROMPT_SUFFIX, return_tensors="pt").input_ids.to(device)

    prefix_emb = wte(prefix_ids)  # [1, p, d]
    suffix_emb = wte(suffix_ids)  # [1, s, d]

    act = activation.to(device=device, dtype=dtype).reshape(-1)
    act = act / (act.norm() + 1e-8) * injection_scale
    act = act.view(1, 1, -1)  # [1, 1, d]

    inputs_embeds = torch.cat([prefix_emb, act, suffix_emb], dim=1)
    return inputs_embeds


@torch.no_grad()
def generate_description(
    model,
    tokenizer,
    activation,
    injection_scale,
    device,
    max_new_tokens: int = 24,
) -> str:
    """Inject one activation and greedily decode a description."""
    model.eval()
    dtype = next(model.parameters()).dtype
    inputs_embeds = _prompt_embeds(model, tokenizer, activation, injection_scale, device, dtype)
    out = model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # With inputs_embeds, generate() returns only the newly generated token ids.
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def build_training_example(
    model,
    tokenizer,
    activation,
    target_text,
    injection_scale,
    device,
):
    """Build (inputs_embeds, labels) for one supervised AV example.

    Sequence = [prefix, injected activation, suffix, target tokens, eos].
    Loss is computed only on the target+eos tokens (prompt positions are -100).
    """
    dtype = next(model.parameters()).dtype
    wte = model.transformer.wte

    prompt_emb = _prompt_embeds(model, tokenizer, activation, injection_scale, device, dtype)
    prompt_len = prompt_emb.shape[1]

    target_ids = tokenizer(" " + target_text.strip(), return_tensors="pt").input_ids.to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    target_ids = torch.cat([target_ids, eos], dim=1)  # [1, t]
    target_emb = wte(target_ids)  # [1, t, d]

    inputs_embeds = torch.cat([prompt_emb, target_emb], dim=1)  # [1, prompt_len + t, d]

    labels = torch.full((1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
    labels[0, prompt_len:] = target_ids[0]  # supervise only the target span

    return inputs_embeds, labels
