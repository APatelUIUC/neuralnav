"""Feature label lookup from pre-cached JSON files."""

import json
from pathlib import Path


_label_cache: dict[int, dict[int, str]] = {}


def load_labels(data_dir: str) -> None:
    """Load all feature label files into memory.

    Expects files at data_dir/feature_labels/layer_{i}.json,
    each containing a JSON object mapping feature index (string) to label string.
    """
    base = Path(data_dir) / "feature_labels"
    _label_cache.clear()

    for layer_idx in range(12):
        path = base / f"layer_{layer_idx}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _label_cache[layer_idx] = {int(k): v for k, v in raw.items()}
        else:
            _label_cache[layer_idx] = {}


def get_label(layer: int, feature_index: int) -> str:
    """Look up a human-readable label for a feature.

    Args:
        layer: Layer index (0-11).
        feature_index: Feature index within that layer.

    Returns:
        Label string, or a fallback like "Feature 1234".
    """
    layer_labels = _label_cache.get(layer, {})
    return layer_labels.get(feature_index, f"Feature {feature_index}")
