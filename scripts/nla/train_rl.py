"""GRPO RL for the NLA — the stage that turns the SAE-bootstrapped warm-start
into a genuine NLA.

The verbalizer (AV, policy) samples K descriptions per activation; the
reconstructor (AR) scores each by how well it recovers the original activation
(reward = reconstruction cosine). GRPO normalizes rewards within each K-group to
get advantages; the AV is updated by policy gradient (+ KL to the frozen
warm-start as an anchor), and the AR is updated supervised on the fresh samples.

Reward is reconstruction faithfulness, NOT SAE-label match — this is what pulls
explanations beyond "SAE features in a sentence."

Usage:
    python train_rl.py --benchmark 20            # measure step time, report reward
    python train_rl.py --steps 400 --batch 8 --k 8 --save
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from inject import _prompt_embeds
from train_ar import AR, normed

OUT = Path(__file__).resolve().parent / "out"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seq_logprobs(model, prompt_emb, gen_ids, mask):
    """Per-token logprobs of gen_ids under `model`, given the prompt embeds.
    prompt_emb: [K, P, d]  gen_ids: [K, L]  mask: [K, L] -> token_logp [K, L]."""
    wte = model.transformer.wte
    full = torch.cat([prompt_emb, wte(gen_ids)], dim=1)  # [K, P+L, d]
    logits = model(inputs_embeds=full).logits           # [K, P+L, V]
    P = prompt_emb.shape[1]
    shift = logits[:, P - 1:-1, :]                       # predicts gen_ids[:, :L]
    logp = F.log_softmax(shift, dim=-1)
    return logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) * mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0, help="time this many steps, then exit")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8, help="activations per step")
    ap.add_argument("--k", type=int, default=8, help="samples per activation (GRPO group)")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ar-lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty to warm-start")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    if not (OUT / "av").exists() or not (OUT / "ar.pt").exists():
        raise SystemExit("warm-start models missing — run train_av.py and train_ar.py first")

    tok = GPT2TokenizerFast.from_pretrained(str(OUT / "av"))
    meta = json.loads((OUT / "av" / "nla_meta.json").read_text())
    scale = meta["injection_scale"]

    av = GPT2LMHeadModel.from_pretrained(str(OUT / "av")).to(device).train()
    ref = GPT2LMHeadModel.from_pretrained(str(OUT / "av")).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(OUT / "ar.pt", map_location=device)
    ar = AR(GPT2LMHeadModel.from_pretrained("gpt2"), ckpt["ar_layers"]).to(device).train()
    ar.load_state_dict(ckpt["state_dict"])

    acts = torch.tensor(np.load(OUT / "activations.npy"), dtype=torch.float32, device=device)
    dtype = next(av.parameters()).dtype

    opt_av = torch.optim.AdamW(av.parameters(), lr=args.lr)
    opt_ar = torch.optim.AdamW(ar.parameters(), lr=args.ar_lr)
    eos = tok.eos_token_id

    n_steps = args.benchmark or args.steps
    rng = np.random.default_rng(0)
    reward_hist, times = [], []

    for step in range(n_steps):
        t0 = time.monotonic()
        batch_idx = rng.choice(len(acts), size=args.batch, replace=False)
        pg_losses, kl_losses, ar_losses, step_rewards = [], [], [], []

        for ai in batch_idx:
            act = acts[ai]
            prompt = _prompt_embeds(av, tok, act, scale, device, dtype)   # [1, P, d]
            prompt_k = prompt.expand(args.k, -1, -1).contiguous()

            # 1. sample K completions
            with torch.no_grad():
                gen = av.generate(inputs_embeds=prompt_k, do_sample=True, temperature=args.temp,
                                  top_p=0.95, max_new_tokens=args.max_new,
                                  pad_token_id=eos)                        # [K, L]
            L = gen.shape[1]
            # mask: 1 up to & including first eos, else 0
            is_eos = (gen == eos)
            first_eos = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((args.k,), L - 1, device=device))
            ar_idx = torch.arange(L, device=device).unsqueeze(0)
            mask = (ar_idx <= first_eos.unsqueeze(1)).float()

            # 2. reward = reconstruction cosine (AR reads the generated text)
            recon = ar(gen)                                                # [K, 768]
            reward = F.cosine_similarity(recon, act.unsqueeze(0).expand(args.k, -1), dim=-1)
            adv = (reward - reward.mean()) / (reward.std() + 1e-6)

            # 3. policy gradient + KL to warm-start ref
            tok_lp = seq_logprobs(av, prompt_k, gen, mask)                 # [K, L] (grad)
            with torch.no_grad():
                tok_lp_ref = seq_logprobs(ref, prompt_k, gen, mask)
            seq_lp = tok_lp.sum(-1)
            pg = -(adv.detach() * seq_lp).mean()
            # k3 KL estimator (nonneg), masked
            log_ratio = (tok_lp_ref - tok_lp)
            kl = ((log_ratio.exp() - 1) - log_ratio) * mask
            kl = kl.sum(-1).mean()

            # 4. AR supervised on fresh samples (reconstruct activation from its text)
            ar_loss = ((normed(recon) - normed(act).unsqueeze(0)) ** 2).sum(-1).mean()

            pg_losses.append(pg); kl_losses.append(kl); ar_losses.append(ar_loss)
            step_rewards.append(reward.mean().item())

        av_loss = torch.stack(pg_losses).mean() + args.beta * torch.stack(kl_losses).mean()
        ar_loss = torch.stack(ar_losses).mean()

        opt_av.zero_grad(); av_loss.backward()
        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0); opt_av.step()
        opt_ar.zero_grad(); ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0); opt_ar.step()

        dt = time.monotonic() - t0
        times.append(dt); reward_hist.append(float(np.mean(step_rewards)))
        if step % max(1, n_steps // 20) == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  reward(cos) {reward_hist[-1]:.3f}  "
                  f"pg {torch.stack(pg_losses).mean().item():.3f}  "
                  f"kl {torch.stack(kl_losses).mean().item():.3f}  {dt:.2f}s")

    avg = float(np.mean(times[1:])) if len(times) > 1 else times[0]
    print(f"\navg step time: {avg:.2f}s  ({args.batch}x{args.k} rollouts/step, "
          f"~{args.batch*args.k*args.max_new/avg:.0f} tok/s)")
    print(f"reward: start {reward_hist[0]:.3f} -> end {reward_hist[-1]:.3f}")
    if args.benchmark:
        full = avg * 400
        print(f"projected 400-step run: {full/60:.0f} min ({full/3600:.1f} hr) on {device}")
    if args.save and not args.benchmark:
        av.save_pretrained(OUT / "av_rl"); tok.save_pretrained(OUT / "av_rl")
        (OUT / "av_rl" / "nla_meta.json").write_text(json.dumps(meta, indent=2))
        torch.save({"state_dict": ar.state_dict(), "ar_layers": ckpt["ar_layers"]}, OUT / "ar_rl.pt")
        print(f"saved RL models -> {OUT}/av_rl, ar_rl.pt")


if __name__ == "__main__":
    main()
