"""Generate mock SAE weights and test server endpoints."""

import json
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
WEIGHTS_DIR = DATA_DIR / "sae_weights"
LABELS_DIR = DATA_DIR / "feature_labels"


def create_mock_weights():
    """Create small mock SAE weights for testing."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for layer in range(12):
        W_enc = rng.standard_normal((768, 24576)).astype(np.float16) * 0.01
        b_enc = np.zeros(24576, dtype=np.float16)
        W_dec = rng.standard_normal((24576, 768)).astype(np.float16) * 0.01
        b_dec = np.zeros(768, dtype=np.float16)

        np.savez(
            WEIGHTS_DIR / f"layer_{layer}.npz",
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
        )
        print(f"  Created layer_{layer}.npz")

        # Mock labels
        labels = {str(i): f"Test feature {i} for layer {layer}" for i in range(100)}
        with open(LABELS_DIR / f"layer_{layer}.json", "w") as f:
            json.dump(labels, f)

    # Unembedding matrix
    W_U = rng.standard_normal((768, 50257)).astype(np.float16) * 0.01
    np.save(WEIGHTS_DIR / "W_U.npy", W_U)
    print("  Created W_U.npy")


def test_endpoints():
    """Test the server endpoints with httpx."""
    import httpx

    base = "http://127.0.0.1:8080"

    # Health check
    r = httpx.get(f"{base}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("  /health OK")

    # Decompose
    rng = np.random.default_rng(123)
    activation = rng.standard_normal(768).tolist()

    r = httpx.post(
        f"{base}/decompose",
        json={"activation": activation, "layer": 5},
        timeout=30,
    )
    assert r.status_code == 200, f"Decompose failed: {r.status_code} {r.text}"
    data = r.json()
    assert "features" in data
    assert "reconstruction_score" in data
    assert "total_features_active" in data
    print(f"  /decompose OK — {len(data['features'])} features, "
          f"reconstruction: {data['reconstruction_score']:.4f}, "
          f"active: {data['total_features_active']}")

    if data["features"]:
        f = data["features"][0]
        print(f"    Top feature: #{f['index']} ({f['label']}) activation={f['activation']}")
        if f["top_positive_logits"]:
            print(f"    Top positive logit: {f['top_positive_logits'][0]}")

        # Ablate
        r = httpx.post(
            f"{base}/ablate",
            json={
                "activation": activation,
                "layer": 5,
                "feature_index": f["index"],
                "feature_activation": f["activation"],
            },
            timeout=30,
        )
        assert r.status_code == 200, f"Ablate failed: {r.status_code} {r.text}"
        ablate_data = r.json()
        assert "modified_activation" in ablate_data
        assert "logit_diff_approx" in ablate_data
        assert len(ablate_data["modified_activation"]) == 768
        diff = ablate_data["logit_diff_approx"]
        print(f"  /ablate OK — promoted: {len(diff.get('tokens_promoted', []))}, "
              f"demoted: {len(diff.get('tokens_demoted', []))}")

    # Edge case: all zeros
    r = httpx.post(
        f"{base}/decompose",
        json={"activation": [0.0] * 768, "layer": 0},
        timeout=30,
    )
    assert r.status_code == 200
    zero_data = r.json()
    print(f"  /decompose (zeros) OK — features: {len(zero_data['features'])}, "
          f"active: {zero_data['total_features_active']}")

    # Validation: wrong size
    r = httpx.post(
        f"{base}/decompose",
        json={"activation": [1.0] * 10, "layer": 0},
        timeout=10,
    )
    assert r.status_code == 422, f"Expected 422 for wrong size, got {r.status_code}"
    print("  /decompose validation (wrong size) OK — 422")

    # Validation: layer out of range
    r = httpx.post(
        f"{base}/decompose",
        json={"activation": [0.0] * 768, "layer": 99},
        timeout=10,
    )
    assert r.status_code == 422, f"Expected 422 for bad layer, got {r.status_code}"
    print("  /decompose validation (bad layer) OK — 422")

    print("\nAll tests passed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create-weights":
        print("Creating mock weights...")
        create_mock_weights()
        print("Done.")
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing endpoints...")
        test_endpoints()
    else:
        print("Usage: python test_server.py [create-weights|test]")
