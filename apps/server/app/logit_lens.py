"""Logit effect computation for SAE features."""

import numpy as np
import tiktoken
from numpy.typing import NDArray

# Module-level tokenizer instance (threadsafe for reads)
_enc = tiktoken.get_encoding("gpt2")


def compute_logit_effects(
    feature_indices: NDArray[np.intp],
    activations: NDArray[np.float32],
    W_dec: NDArray[np.float16],
    W_U: NDArray[np.float16],
    top_n: int = 5,
) -> list[dict]:
    """Compute the logit-space effect for each feature.

    For each feature:
        logit_effect = activation * (W_dec[feature_index] @ W_U)

    Args:
        feature_indices: Array of feature indices.
        activations: Corresponding activation values.
        W_dec: Decoder weight matrix (24576, 768).
        W_U: Unembedding matrix (768, 50257).
        top_n: Number of top positive/negative tokens to return.

    Returns:
        List of dicts with top_positive_logits and top_negative_logits per feature.
    """
    W_dec_f32 = W_dec.astype(np.float32)
    W_U_f32 = W_U.astype(np.float32)
    results = []

    for idx, act in zip(feature_indices, activations):
        logit_effect = float(act) * (W_dec_f32[idx] @ W_U_f32)

        top_pos_indices = np.argpartition(logit_effect, -top_n)[-top_n:]
        top_pos_indices = top_pos_indices[np.argsort(logit_effect[top_pos_indices])[::-1]]

        top_neg_indices = np.argpartition(logit_effect, top_n)[:top_n]
        top_neg_indices = top_neg_indices[np.argsort(logit_effect[top_neg_indices])]

        results.append(
            {
                "top_positive_logits": [
                    {"token": _decode_token(int(i)), "logit": round(float(logit_effect[i]), 4)}
                    for i in top_pos_indices
                ],
                "top_negative_logits": [
                    {"token": _decode_token(int(i)), "logit": round(float(logit_effect[i]), 4)}
                    for i in top_neg_indices
                ],
            }
        )

    return results


def compute_logit_diff(
    feature_index: int,
    feature_activation: float,
    W_dec: NDArray[np.float16],
    W_U: NDArray[np.float16],
    original_logits: NDArray[np.float32] | None = None,
    top_n: int = 5,
) -> dict:
    """Compute the logit difference caused by ablating a single feature.

    delta_logits = -feature_activation * (W_dec[feature_index] @ W_U)

    Args:
        feature_index: Index of the ablated feature.
        feature_activation: Activation value being removed.
        W_dec: Decoder weight matrix (24576, 768).
        W_U: Unembedding matrix (768, 50257).
        original_logits: Optional original logit vector for top-token comparison.
        top_n: Number of promoted/demoted tokens to return.

    Returns:
        Dict with tokens_promoted, tokens_demoted, and optional top token info.
    """
    delta = -feature_activation * (W_dec[feature_index].astype(np.float32) @ W_U.astype(np.float32))

    promoted_indices = np.argpartition(delta, -top_n)[-top_n:]
    promoted_indices = promoted_indices[np.argsort(delta[promoted_indices])[::-1]]

    demoted_indices = np.argpartition(delta, top_n)[:top_n]
    demoted_indices = demoted_indices[np.argsort(delta[demoted_indices])]

    result: dict = {
        "tokens_promoted": [
            {"token": _decode_token(int(i)), "logit_change": round(float(delta[i]), 4)}
            for i in promoted_indices
        ],
        "tokens_demoted": [
            {"token": _decode_token(int(i)), "logit_change": round(float(delta[i]), 4)}
            for i in demoted_indices
        ],
    }

    if original_logits is not None:
        original_top = int(np.argmax(original_logits))
        new_logits = original_logits + delta
        new_top = int(np.argmax(new_logits))
        result["original_top_token"] = _decode_token(original_top)
        result["new_top_token"] = _decode_token(new_top)

    return result


def _decode_token(token_id: int) -> str:
    """Decode a single token ID to its string representation."""
    try:
        return _enc.decode([token_id])
    except Exception:
        return f"<token_{token_id}>"
