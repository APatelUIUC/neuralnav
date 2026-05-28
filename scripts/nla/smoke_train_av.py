"""Smoke test: can a GPT-2 verbalizer learn vector-specific text from an
injected activation?

This is the riskiest unknown on the NLA critical path. We take a handful of
*real* GPT-2 residual-stream activations, assign each a distinct made-up label,
and overfit the verbalizer to emit the right label from the injected vector
alone. If train loss collapses and the model reproduces each label from its
vector, the injection-based training approach is viable and we can scale to real
warm-start data.

Run:
    python smoke_train_av.py            # ~1-2 min on M4 Max (MPS)
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from inject import build_training_example, default_injection_scale, generate_description

LAYER = 8          # residual stream after block 7 (~2/3 through); matches resid_8
STEPS = 250
LR = 1e-4

# Distinct (prompt -> made-up label) pairs. The label is arbitrary; the point is
# that each must be recoverable from that prompt's activation direction alone.
EXAMPLES = [
    ("The Eiffel Tower stands tall in the city of", "a famous landmark in Paris France"),
    ("Photosynthesis converts sunlight into chemical", "plant biology and energy conversion"),
    ("The quarterback threw a perfect spiral down the", "American football game action"),
    ("She poured the espresso and steamed the", "coffee preparation in a cafe"),
    ("The algorithm sorts the array in logarithmic", "computer science time complexity"),
    ("Beethoven composed his ninth symphony while", "classical music composition"),
    ("The volcano erupted, spewing molten lava and", "geological eruption and magma"),
    ("Investors watched the stock market plunge after the", "financial markets and trading"),
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract_activation(model, tokenizer, prompt, device):
    """Residual stream at LAYER, last token position."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    # hidden_states: tuple of 13 (embeddings + 12 blocks). [LAYER] = after block LAYER-1.
    return out.hidden_states[LAYER][0, -1, :].detach().clone()


def main():
    device = get_device()
    print(f"device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    scale = default_injection_scale(model)
    print(f"injection_scale (mean wte norm): {scale:.3f}")

    # Extract real activations once (frozen targets to verbalize).
    data = []
    for prompt, label in EXAMPLES:
        act = extract_activation(model, tokenizer, prompt, device)
        data.append((act, label))
    print(f"extracted {len(data)} activations at layer {LAYER}, dim={data[0][0].shape[0]}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    print("\n--- training (overfit) ---")
    for step in range(STEPS):
        total = 0.0
        opt.zero_grad()
        for act, label in data:
            inputs_embeds, labels = build_training_example(
                model, tokenizer, act, label, scale, device
            )
            loss = model(inputs_embeds=inputs_embeds, labels=labels).loss
            (loss / len(data)).backward()
            total += loss.item() / len(data)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  loss {total:.4f}")

    print("\n--- generations from injected activation (greedy) ---")
    correct = 0
    for (act, label), (prompt, _) in zip(data, EXAMPLES):
        gen = generate_description(model, tokenizer, act, scale, device, max_new_tokens=16)
        hit = label.split()[0].lower() in gen.lower()
        correct += hit
        print(f"  prompt : {prompt[:42]!r}")
        print(f"  target : {label!r}")
        print(f"  got    : {gen!r}  {'OK' if hit else 'x'}")
    print(f"\nrecovered {correct}/{len(data)} labels from activation alone")
    print("PASS — injection carries learnable signal" if total < 0.5 and correct >= len(data) * 0.6
          else "INCONCLUSIVE — inspect loss/generations above")


if __name__ == "__main__":
    main()
