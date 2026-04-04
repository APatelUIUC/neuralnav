# How SAE Explorer Works

A complete technical walkthrough of the system — what it does, how every piece fits together, and what all the terminology means.

---

## The Big Picture

SAE Explorer lets you type a sentence, watch GPT-2 process it, and then peer inside the model to see *what concepts it's representing* at every layer and token position. You can then surgically remove individual concepts ("ablate" them) and see how the model's predictions change.

The key insight: neural networks represent concepts as directions in high-dimensional space. A Sparse Autoencoder (SAE) learns to decompose the model's internal activations into these directions, giving each one a human-readable label. This tool makes that decomposition interactive.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   YOUR BROWSER                       │
│                                                      │
│  ┌──────────────┐   ┌────────────┐   ┌───────────┐ │
│  │ GPT-2 Small  │   │ Tokenizer  │   │  React UI │ │
│  │ (ONNX, 623MB)│   │ (r50k_base)│   │           │ │
│  │              │   │            │   │ Heatmap   │ │
│  │ Runs full    │   │ Text → IDs │   │ Features  │ │
│  │ inference    │   │ IDs → Text │   │ Ablation  │ │
│  │ via WASM     │   │            │   │ Predictions│ │
│  └──────┬───────┘   └────────────┘   └─────┬─────┘ │
│         │                                    │       │
│         │ activations (768 floats)          │       │
│         └──────────────┬─────────────────────┘       │
└────────────────────────┼─────────────────────────────┘
                         │ HTTP POST (~6KB request)
                         ▼
┌─────────────────────────────────────────────────────┐
│              LOCAL SERVER (FastAPI, port 8080)        │
│                                                      │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ SAE Weights      │  │ Unembedding Matrix (W_U) │  │
│  │ 12 layers × 4    │  │ 768 × 50257             │  │
│  │ matrices each    │  │ Maps activations → token │  │
│  │ ~863MB total     │  │ predictions              │  │
│  └────────┬─────────┘  └────────────┬────────────┘  │
│           │                          │               │
│  ┌────────▼──────────────────────────▼────────────┐  │
│  │ /decompose — SAE forward pass → top features   │  │
│  │ /ablate — remove feature → logit diff          │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ Feature Labels (from Neuronpedia)              │  │
│  │ ~278,000 human-readable descriptions           │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## The Model

**GPT-2 Small** by OpenAI. Released in 2019, it's the smallest GPT-2 variant:

- **Parameters**: 124 million
- **Layers**: 12 transformer blocks
- **Hidden dimension**: 768 (each activation is a vector of 768 numbers)
- **Vocabulary**: 50,257 tokens (using BPE tokenization, called `r50k_base`)
- **Context window**: 1,024 tokens

We chose GPT-2 Small because it's the model with the best publicly available SAE weights (from Joseph Bloom's research). It's small enough to run in a browser but large enough to exhibit interesting interpretable behavior.

### The ONNX Export

We can't run PyTorch in a browser. Instead, we exported the model to **ONNX** (Open Neural Network Exchange), a portable format that runs via ONNX Runtime Web's WASM backend.

The standard HuggingFace ONNX export only outputs the final logits (token predictions). We needed the **residual stream** — the intermediate activations at every layer — because that's what the SAE decomposes. So we wrote a custom export (`scripts/export_onnx.py`) that manually walks through GPT-2's layers and exposes 14 outputs:

- `logits` — final token predictions (shape: batch × seq_len × 50257)
- `resid_0` through `resid_12` — the residual stream at 13 positions (shape: batch × seq_len × 768)

The model wouldn't trace correctly with HuggingFace's attention code (the SDPA and masking utilities break ONNX tracing in recent library versions), so we bypass `model.forward()` entirely and call each sub-module directly:

```python
hidden = wte(input_ids) + wpe(position_ids)  # embeddings
for block in transformer_blocks:
    hidden = hidden + block.attn(block.ln_1(hidden))   # attention
    hidden = hidden + block.mlp(block.ln_2(hidden))    # MLP
    all_hidden.append(hidden)                           # ← capture this
logits = lm_head(ln_f(hidden))                         # final prediction
```

### Quantization

The full float32 model is **623 MB**. We also generated a float16 version at **312 MB**, but ONNX Runtime Web couldn't run it — the `LayerNormalization` op gets mixed float32 inputs with float16 weights, which the WASM backend rejects. So we serve the float32 version. It's cached by the browser after the first download.

---

## What Runs Where

### In the Browser

- **ONNX Runtime Web** (WASM backend) — runs the full GPT-2 model. Inference takes ~1-3 seconds for a 20-token prompt on a modern laptop. Uses `navigator.hardwareConcurrency` threads.
- **gpt-tokenizer** (r50k_base encoding) — converts text to token IDs and back. Pure JavaScript, no server call needed.
- **React UI** — heatmap (canvas-based), feature panel, ablation diff, prediction bar. State managed with Zustand.
- **Activation norm computation** — for each of the 13 residual stream positions × each token, we compute the L2 norm (magnitude) of the 768-dim vector. This is what colors the heatmap.

### On Your Local Server

- **FastAPI** on port 8080 — two endpoints, stateless.
- **SAE weights** — 12 sets of encoder/decoder matrices loaded into RAM at startup (~863 MB as float16 numpy arrays, computed in float32 at request time).
- **Unembedding matrix** (W_U) — the model's final projection from 768-dim space to 50,257-dim vocabulary space. Used for logit lens computations.
- **Feature labels** — 278,000 human-readable descriptions from Neuronpedia, loaded from JSON files.
- **tiktoken** — GPT-2 tokenizer on the server side for decoding token IDs in logit effect computations.

---

## What Is the Residual Stream?

A transformer processes tokens through a series of layers. At each layer, the model reads the current state (a 768-dim vector per token), applies attention and MLP operations, and adds the result back. This "running total" is the **residual stream**.

```
Input: "The Eiffel Tower is in"
         ↓
    [Embedding] → resid_0: each token is now a 768-dim vector
         ↓
    [Layer 0: attention + MLP] → resid_1: vectors updated
         ↓
    [Layer 1: attention + MLP] → resid_2
         ↓
         ...
         ↓
    [Layer 11: attention + MLP] → resid_12
         ↓
    [LayerNorm → Unembedding] → logits → "Paris" at 4.2%, "the" at 8.1%, ...
```

The residual stream is where the model stores everything it "knows" at each point in processing. At early layers (resid_0, resid_1), representations are close to the raw token embeddings. By later layers (resid_10, resid_11), they encode complex relationships like "this token should predict a French city."

The heatmap shows the **L2 norm** (magnitude) of each residual stream vector. Higher norm generally means more "information" or "processing" has accumulated at that position.

### The Attention Sink

You'll notice the first token always has much higher norms than everything else. This is called an **attention sink** — a well-documented phenomenon where the causal attention mask forces every token to be able to attend to position 0, so it becomes a "garbage dump" for attention weight when heads don't have meaningful targets. The heatmap normalizes its color scale from tokens 1 onward so this doesn't wash out the visualization.

---

## What Is a Sparse Autoencoder (SAE)?

A neural network's 768-dim residual stream vectors are dense and uninterpretable — every dimension participates in representing every concept simultaneously (**superposition**). You can't just look at dimension 42 and say "that's the France dimension."

A **Sparse Autoencoder** learns to decompose these dense vectors into a much larger set of **sparse, interpretable directions**. It's trained on millions of activations from the model and learns:

- An **encoder** that maps 768 dims → 24,576 features (with ReLU, so most are zero)
- A **decoder** that maps 24,576 features → 768 dims (reconstructing the original)

The key property: for any given input, only ~50-200 features are active (nonzero). Each feature corresponds to a specific, human-interpretable concept.

### The Math

Given a residual stream vector `x` (768-dim):

```
Encode:  features = ReLU((x - b_dec) @ W_enc + b_enc)
                    ↑                    ↑         ↑
                    sparse!         (768×24576)  (24576,)
                    (24576-dim, mostly zeros)

Decode:  x_hat = features @ W_dec + b_dec
                             ↑        ↑
                        (24576×768)  (768,)
```

- `b_dec` is subtracted first (centering the input)
- `W_enc` (768 × 24,576) projects into the overcomplete feature space
- `b_enc` (24,576) is added before ReLU
- ReLU zeros out negative values, enforcing sparsity
- `W_dec` (24,576 × 768) reconstructs the activation
- The reconstruction score (cosine similarity between `x` and `x_hat`) tells you how well the SAE captures the activation — typically 0.90-0.99

### The Weight Files

We use SAEs trained by **Joseph Bloom** from the repo `jbloom/GPT2-Small-SAEs-Reformatted` on HuggingFace. There's one SAE per layer, each trained on that layer's `hook_resid_pre` (the residual stream before the layer processes it).

Per layer, the weights are:
- `W_enc`: 768 × 24,576 (encoder matrix)
- `b_enc`: 24,576 (encoder bias)
- `W_dec`: 24,576 × 768 (decoder matrix)
- `b_dec`: 768 (decoder bias / centering)

Stored as float16: ~75 MB per layer, ~900 MB total for all 12 layers.

---

## What Is a Feature?

When you click a cell in the heatmap, the browser sends that 768-dim activation vector to the server. The server runs the SAE encoder and finds the **top 20 features** with the highest activation values.

Each feature has:

- **Index** — its position in the 24,576-dim feature space (e.g., #3131)
- **Activation** — how strongly it fired on this input (e.g., 4.82). Higher = this concept is more strongly represented.
- **Label** — a human-readable description from Neuronpedia, generated by GPT-3.5/4 analyzing what inputs cause the feature to fire (e.g., "mentions of locations away from home and interactions with people from different places")
- **Logit effects** — what this feature does to the model's predictions. Computed via: `effect = activation × (W_dec[feature] @ W_U)`. This tells you which tokens this feature promotes (e.g., "+Paris") and suppresses (e.g., "-London").

### Where Do the Labels Come From?

**Neuronpedia** (neuronpedia.org) is a community platform that catalogs SAE features. For each feature, they ran automated interpretability: feed many texts through the model, find which inputs activate the feature most, then ask GPT-3.5/4 to describe the pattern. We bulk-downloaded all ~278,000 labels via their export API.

---

## What Is Ablation?

**Ablation** means surgically removing one feature's contribution from the residual stream. It answers: "What would the model predict if this concept wasn't represented here?"

### The Math

Each feature contributes a specific direction to the residual stream:

```
contribution = feature_activation × W_dec[feature_index]
               ↑                     ↑
               scalar (e.g., 4.82)   768-dim direction
```

Ablation subtracts this:

```
modified_activation = original_activation - contribution
```

This gives us a new 768-dim vector representing "what the residual stream would look like without this feature."

### What Happens After Ablation

The server computes a **logit lens approximation** of how predictions change. Instead of re-running the entire model from that layer forward (which would require partial inference), we project the *difference* through the unembedding matrix:

```
delta_logits = -feature_activation × (W_dec[feature_index] @ W_U)
```

This 50,257-dim vector tells us how every token's probability changes. The browser adds this to the original logits and re-runs softmax to show the new top predictions.

This is an approximation — it ignores the nonlinear effects of subsequent layers processing the modified activation. But it's fast, requires no extra inference, and is surprisingly accurate for strong features.

---

## The Data Flow, Step by Step

### 1. User Types a Prompt

"The Eiffel Tower is in"

### 2. Browser: Tokenization

The `gpt-tokenizer` library (r50k_base BPE encoding) converts this to token IDs:

```
"The" → 464
" Eiffel" → 46863
" Tower" → 8765
" is" → 318
" in" → 287
```

### 3. Browser: ONNX Inference

The token IDs are passed to the ONNX model as a `BigInt64Array` tensor. ONNX Runtime Web runs the full model via WebAssembly, returning:

- `logits`: Float32Array of 5 × 50,257 values (predictions for each position)
- `resid_0` through `resid_12`: thirteen Float32Arrays of 5 × 768 values each

### 4. Browser: Norm Computation + Heatmap

For each (layer, token) pair, compute the L2 norm of the 768-dim vector:

```
norm = sqrt(sum(v[i]² for i in 0..767))
```

This produces a 13 × 5 grid of scalars that color the heatmap using a viridis color scale.

### 5. Browser: Show Predictions

The logits for the last token position (index 4, " in") are extracted and softmaxed to show the top-8 predicted next tokens with probabilities.

### 6. User Clicks a Heatmap Cell

Say they click (token=4 " in", layer=8). The browser extracts the 768-dim vector from `resid_8` at position 4 and sends it to the server:

```json
POST /decompose
{ "activation": [0.123, -0.456, ...768 floats...], "layer": 8 }
```

### 7. Server: SAE Decomposition

The server:
1. Loads layer 8's SAE weights
2. Centers: `x_centered = activation - b_dec`
3. Encodes: `features = ReLU(x_centered @ W_enc + b_enc)` → 24,576-dim sparse vector
4. Finds top 20 nonzero features by activation value
5. For each: computes logit effects via `activation × (W_dec[i] @ W_U)`
6. Looks up human labels from the Neuronpedia cache
7. Computes reconstruction score (cosine similarity)
8. Returns ~10KB JSON response

### 8. User Clicks "Ablate" on a Feature

The browser sends the same activation plus the feature to remove:

```json
POST /ablate
{ "activation": [...], "layer": 8, "feature_index": 3131, "feature_activation": 4.82 }
```

### 9. Server: Ablation

1. `modified = activation - 4.82 × W_dec[3131]`
2. `delta_logits = -4.82 × (W_dec[3131] @ W_U)` → 50,257-dim vector
3. Returns modified activation + full logit delta + top promoted/demoted tokens

### 10. Browser: Show Modified Predictions

The browser adds `delta_logits` to the original logits and re-softmaxes, showing the new top-8 predictions alongside the originals.

---

## Prerequisites and Dependencies

### Phase 0: Data Preparation (one-time, on your machine)

These scripts run locally to produce the artifacts the app needs.

**Python packages** (in `scripts/.venv`):
- `torch` — PyTorch, for loading GPT-2 and tracing the ONNX export
- `transformers` — HuggingFace, for the pretrained GPT-2 model
- `onnx` — ONNX model format library
- `onnxruntime` — for validating the exported model
- `onnxconverter-common` — for float16 quantization (attempted, not used in final)
- `safetensors` — for reading SAE weight files
- `huggingface-hub` — for downloading SAE weights from HuggingFace

**Artifacts produced**:
| File | Size | What |
|------|------|------|
| `models/gpt2-resid.onnx` | 623 MB | GPT-2 with 14 outputs (float32) |
| `apps/server/data/sae_weights/layer_0-11.npz` | 863 MB total | SAE weights (float16) |
| `apps/server/data/sae_weights/W_U.npy` | 74 MB | Unembedding matrix (float16) |
| `apps/server/data/feature_labels/layer_0-11.json` | ~12 MB total | 278K labels from Neuronpedia |

### Frontend (browser)

**npm packages**:
- `onnxruntime-web` — runs ONNX models in the browser via WASM
- `gpt-tokenizer` — GPT-2 tokenizer (r50k_base) in pure JavaScript
- `react` + `react-dom` v19 — UI framework
- `zustand` — lightweight state management
- `tailwindcss` v4 — styling
- `vite` — dev server and bundler

### Backend (local server)

**Python packages** (in `apps/server/.venv`):
- `fastapi` + `uvicorn` — web framework and ASGI server
- `numpy` — matrix multiplication for SAE forward passes
- `tiktoken` — GPT-2 tokenizer for decoding token IDs in logit effects
- `pydantic` — request/response validation

---

## Scaling to Larger Models

This architecture could work for larger models, but each piece gets harder:

### The Browser Inference Problem

GPT-2 Small (124M params, 623MB ONNX) is near the limit of what runs comfortably in a browser. For larger models:

| Model | Params | Estimated ONNX Size | Browser Feasible? |
|-------|--------|--------------------|--------------------|
| GPT-2 Small | 124M | 623 MB | Yes (current) |
| GPT-2 Medium | 355M | ~1.5 GB | Marginal — long download, ~5-10s inference |
| GPT-2 Large | 774M | ~3 GB | No — too much memory for most browsers |
| Llama 3.2 1B | 1.2B | ~5 GB | No |
| Llama 3.1 8B | 8B | ~32 GB | Absolutely not |

For models beyond GPT-2 Medium, you'd move inference to the server too (or use WebGPU with aggressive quantization like int4).

### The SAE Availability Problem

SAEs must be trained specifically for a model and hook point. Currently available:

- **GPT-2 Small**: Joseph Bloom's SAEs (what we use) — well-tested, good labels
- **Gemma 2 2B / 9B / 27B**: Google DeepMind released SAEs via Gemma Scope
- **Llama 3.1 8B**: Some community SAEs exist but fewer labels
- **Claude**: Anthropic published SAEs for an older Claude model but weights aren't public

For a new model, you'd need to either find existing SAEs or train your own (requires significant GPU compute — days on A100s for a single layer).

### The Server Memory Problem

SAE weights scale with model dimension and dictionary size:

```
Per layer: 2 × hidden_dim × dictionary_size × 2 bytes (float16)

GPT-2 Small (768 × 24,576):    ~75 MB/layer, ~900 MB total
Gemma 2 2B (2,304 × 16,384):   ~144 MB/layer, ~3.7 GB total (26 layers)
Llama 8B (4,096 × 65,536):     ~1 GB/layer, ~32 GB total (32 layers)
```

For large models, you'd need lazy loading (only load requested layers), GPU acceleration for the matmuls, or a more powerful server.

### What Would Change Architecturally

For a hypothetical "Llama 8B SAE Explorer":

1. **Model inference**: Moves to server (GPU). Browser becomes a pure UI client.
2. **SAE weights**: Loaded on-demand per layer, possibly on GPU for fast matmul.
3. **Heatmap data**: Server computes norms and sends them as a small JSON grid, not raw activations.
4. **Feature decomposition**: Same architecture, just bigger matrices.
5. **Cost**: A GPU server (e.g., A10G on AWS) at ~$1/hr instead of a $14/mo CPU instance.

The fundamental UX and concepts remain identical — type a prompt, see the heatmap, click to decompose, ablate to experiment. Only the plumbing changes.
