#!/usr/bin/env python3
"""
Phase 0 — Script 3: Pre-cache feature labels from Neuronpedia API.

Fetches human-readable descriptions for SAE features from the Neuronpedia API
and saves them as per-layer JSON files. The script is resume-capable: it skips
features that have already been fetched.

Usage:
    python prefetch_feature_labels.py
    python prefetch_feature_labels.py --layers 0 1 --top-k 500 --rps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


API_BASE = "https://www.neuronpedia.org/api/feature/gpt2-small"
DEFAULT_TOP_K = 2000
DEFAULT_RPS = 5  # requests per second


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_feature(layer: int, index: int, session: requests.Session) -> dict | None:
    """Fetch a single feature's metadata from Neuronpedia.
    Returns a dict with at least 'index' and 'description', or None on failure."""

    url = f"{API_BASE}/{layer}-res-jb/{index}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    description = data.get("explanation", data.get("description", ""))
    return {
        "index": index,
        "description": description,
    }


def load_existing(path: Path) -> dict[int, dict]:
    """Load already-fetched features from a JSON file, keyed by index."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        features = json.load(f)
    return {feat["index"]: feat for feat in features}


def save_features(path: Path, features: dict[int, dict]) -> None:
    """Save features dict (keyed by index) to a sorted JSON list."""
    sorted_list = [features[k] for k in sorted(features.keys())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_list, f, indent=2, ensure_ascii=False)


def fetch_layer(
    layer: int,
    indices: list[int],
    output_dir: Path,
    rps: float,
    max_retries: int = 5,
) -> None:
    """Fetch features for one layer with rate limiting and exponential backoff."""

    out_path = output_dir / f"layer_{layer}.json"
    existing = load_existing(out_path)
    to_fetch = [i for i in indices if i not in existing]

    if not to_fetch:
        print(f"  Layer {layer}: all {len(indices)} features already cached, skipping.")
        return

    print(f"  Layer {layer}: {len(existing)} cached, {len(to_fetch)} remaining")

    interval = 1.0 / rps
    session = requests.Session()
    session.headers["User-Agent"] = "sae-explorer-prefetch/0.1"

    failures = 0
    pbar = tqdm(to_fetch, desc=f"  L{layer}", leave=False)

    for index in pbar:
        retries = 0
        while retries <= max_retries:
            try:
                t0 = time.monotonic()
                feat = fetch_feature(layer, index, session)
                if feat is not None:
                    existing[feat["index"]] = feat

                # Respect rate limit
                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                break

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 404:
                    # Feature doesn't exist, skip
                    break
                if status == 429 or status >= 500:
                    retries += 1
                    backoff = min(2 ** retries, 60)
                    pbar.set_postfix({"retry": retries, "backoff": f"{backoff}s"})
                    time.sleep(backoff)
                else:
                    retries += 1
                    time.sleep(2 ** retries)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                retries += 1
                backoff = min(2 ** retries, 60)
                time.sleep(backoff)

            except Exception as e:
                failures += 1
                pbar.set_postfix({"errors": failures})
                break
        else:
            failures += 1

        # Periodic save every 100 features
        if len(existing) % 100 == 0:
            save_features(out_path, existing)

    # Final save
    save_features(out_path, existing)
    print(f"  Layer {layer}: saved {len(existing)} features to {out_path.name}"
          f" ({failures} failures)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-cache feature labels from Neuronpedia")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "apps" / "server" / "data" / "feature_labels",
        help="Directory for output JSON files (default: ../apps/server/data/feature_labels)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Layer indices to fetch (default: 0-11)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of feature indices to fetch per layer (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help=f"Max requests per second (default: {DEFAULT_RPS})",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit feature indices to fetch (overrides --top-k)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Rate limit: {args.rps} req/sec\n")

    indices = args.indices if args.indices is not None else list(range(args.top_k))

    for layer in args.layers:
        fetch_layer(layer, indices, args.output_dir, args.rps)

    print("\nDone.")


if __name__ == "__main__":
    main()
