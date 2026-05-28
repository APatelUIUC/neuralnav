# NLA on GPT-2 — cross-machine handoff & log

This is the shared coordination doc between two Claude Code agents working on the
same repo, relayed by the human (Akash):
- **akashmac** — Apple M4 Max (MPS). Did the design, data-gen, and warm-start.
- **super** — Windows PC w/ RTX 4080 Super (CUDA, in WSL2). Runs the GPU-heavy **RL** stage.

If you are the **super** agent: read this whole file, do the "Tasks for super"
below, then append to the **LOG** at the bottom, commit, and push. `git pull`
before you start and before each push.

---

## What this project is

A **Natural Language Autoencoder (NLA)** for GPT-2-small, extending Akash's
existing SAE explorer (this repo). An NLA is two fine-tuned models:
- **AV (verbalizer)**: residual activation → English description. The activation
  is injected as one token embedding into a prompt (`inputs_embeds`), then it
  autoregresses a description. See `inject.py`.
- **AR (reconstructor)**: description text → activation vector (truncated GPT-2 +
  linear head). Reward/score = cosine(reconstructed, original).

Goal: an in-browser viewer where hovering a token shows the NLA's explanation
next to the SAE features, with reconstruction cosine as a faithfulness score.
The `/explain` server endpoint (mirrors `/decompose`) is already wired.

## Current state (from akashmac)

- Injection mechanism smoke-tested ✅ (AV learns vector→text).
- Dataset committed (no need to re-run data-gen): `out/activations.npy` (6000×768),
  `out/examples.jsonl` (SAE-label-derived targets), `out/label_cache.json`,
  `out/corpus.txt` (FineWeb sample).
- Warm-start AR trained → **FVE ≈ 0.09 (low)**. The SAE-feature-list targets are
  generic (many activations get similar "emphasis/intensity" descriptions), so the
  text underdetermines the activation direction. **This is the main thing to push on.**
- Big model files (`out/av/`, `*.pt`) are gitignored — regenerate them locally.

## Setup on super (do once, in WSL2 Ubuntu with CUDA)

```bash
cd ~ && gh repo clone APatelUIUC/neuralnav 2>/dev/null || (cd neuralnav && git fetch)
cd neuralnav && git checkout nla && git pull
cd scripts/nla
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers numpy requests
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
```
(All scripts auto-detect `cuda`/`mps`/`cpu` — no code changes needed.)

## Tasks for super

1. **Reproduce + improve warm-start** (fast on the 4080). Current AR FVE ≈ 0.09.
   Try to raise it before RL:
   ```bash
   python train_ar.py --epochs 8 --ar-layers 9 --batch 16 --lr 1e-4   # more epochs + higher LR
   python train_av.py --epochs 3 --batch 16
   ```
   Report the AR val FVE. If still <0.2, note it — RL may still work, and a
   teacher-summary warm-start (vs SAE labels) is the fallback lever.
2. **Benchmark RL step time:** `python train_rl.py --benchmark 20`  → record
   avg step time + projected full-run duration on the 4080.
3. **Run RL:** `python train_rl.py --steps 400 --batch 8 --k 8 --save`
   Watch `reward(cos)` rise. Saves to `out/av_rl/` + `out/ar_rl.pt`.
4. **Report:** the RL FVE/reward before→after, and whether explanations look
   meaningfully different from the SAE feature lists (the "is it just an SAE" test).

## Coordination protocol

- **Pull before you start and before each push.** Append to the LOG below with
  your machine name + UTC date. Keep entries short (numbers + decisions).
- Don't commit large models (gitignored). Commit code changes, metrics, and short
  result notes. Push the `nla` branch.
- Akash relays one-line nudges between the two agents.

---

## LOG

### 2026-05-28 — akashmac
- Pipeline built; injection smoke test passed; 6k FineWeb dataset committed.
- Warm-start AR FVE ≈ 0.09 (generic SAE-label targets). AV warm-start training.
- Handing the RL stage to super (4080). RL script: `train_rl.py` (GRPO, reward=recon cosine).
- Open question for super: does higher-LR / more-epoch warm-start lift FVE, and does RL push explanations beyond SAE-label paraphrase?

### 2026-05-28 (later) — akashmac
- Warm-start AV finished. Generations are SAE-label *style* but **generic/repetitive and
  often wrong** (e.g. target "authors collaborating" → got "names of individuals; names of
  individuals"). Confirms warm-start ≈ a weak SAE paraphraser. AR FVE ≈ 0.09.
- Validated `train_rl.py` runs end-to-end on MPS (no bugs). **Fixed the reward**: was raw
  cosine (~0.7, inflated by GPT-2 anisotropy); now **mean-centered** cosine → starts ~0,
  real dynamic range for GRPO. Pull to get this fix.
- MPS step time ~0.6s at batch2×k4×16tok (warmup ~7s). Full batch8×k8×24tok will be larger —
  **super: run `train_rl.py --benchmark 20` for the real CUDA number.**
- Likely levers for super, in order: (1) more/higher-LR warm-start, (2) RL with the fixed
  reward, (3) if FVE stays low, switch warm-start targets from SAE labels to teacher summaries.

### (super, append below)
