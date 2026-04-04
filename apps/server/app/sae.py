"""SAE weight loading and forward pass logic."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class SAEWeights:
    """Sparse Autoencoder weights for a single layer."""

    W_enc: NDArray[np.float16]  # (768, 24576)
    b_enc: NDArray[np.float16]  # (24576,)
    W_dec: NDArray[np.float16]  # (24576, 768)
    b_dec: NDArray[np.float16]  # (768,)


def load_layer_weights(path: Path) -> SAEWeights:
    """Load SAE weights from a single .npz file."""
    data = np.load(path)
    return SAEWeights(
        W_enc=data["W_enc"].astype(np.float16),
        b_enc=data["b_enc"].astype(np.float16),
        W_dec=data["W_dec"].astype(np.float16),
        b_dec=data["b_dec"].astype(np.float16),
    )


def load_all_weights(data_dir: str) -> tuple[dict[int, SAEWeights], NDArray[np.float16]]:
    """Load all 12 layer SAE weights and the unembedding matrix.

    Returns:
        Tuple of (layer_weights dict, W_U matrix).
    """
    base = Path(data_dir)
    weights_dir = base / "sae_weights"

    layer_weights: dict[int, SAEWeights] = {}
    for layer_idx in range(12):
        path = weights_dir / f"layer_{layer_idx}.npz"
        if path.exists():
            layer_weights[layer_idx] = load_layer_weights(path)

    w_u_path = weights_dir / "W_U.npy"
    W_U = np.load(w_u_path).astype(np.float16) if w_u_path.exists() else np.zeros(
        (768, 50257), dtype=np.float16
    )

    return layer_weights, W_U


def encode(x: NDArray[np.float32], weights: SAEWeights) -> NDArray[np.float32]:
    """SAE forward pass: encode activation into sparse feature space.

    z = ReLU((x - b_dec) @ W_enc + b_enc)

    Args:
        x: Input activation vector (768,).
        weights: SAE weights for the target layer.

    Returns:
        Sparse activation vector (24576,).
    """
    x_centered = x.astype(np.float32) - weights.b_dec.astype(np.float32)
    z = x_centered @ weights.W_enc.astype(np.float32) + weights.b_enc.astype(np.float32)
    np.maximum(z, 0, out=z)
    return z


def decode(z: NDArray[np.float32], weights: SAEWeights) -> NDArray[np.float32]:
    """SAE decode: reconstruct activation from sparse features.

    x_hat = z @ W_dec + b_dec

    Args:
        z: Sparse activation vector (24576,).
        weights: SAE weights for the target layer.

    Returns:
        Reconstructed activation vector (768,).
    """
    return z @ weights.W_dec.astype(np.float32) + weights.b_dec.astype(np.float32)


def top_k_features(
    z: NDArray[np.float32], k: int = 20
) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    """Get indices and values of the top-k nonzero activations.

    Uses np.argpartition for efficient partial sorting.

    Args:
        z: Sparse activation vector (24576,).
        k: Number of top features to return.

    Returns:
        Tuple of (indices, values), both of length <= k.
    """
    nonzero_mask = z > 0
    num_active = int(np.sum(nonzero_mask))

    if num_active == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

    k = min(k, num_active)

    # argpartition is O(n) vs O(n log n) for full sort
    top_indices = np.argpartition(z, -k)[-k:]
    # Sort the top-k by value descending
    sorted_order = np.argsort(z[top_indices])[::-1]
    top_indices = top_indices[sorted_order]

    return top_indices, z[top_indices]
