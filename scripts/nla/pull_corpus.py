"""Stream a small, diverse corpus from FineWeb for NLA warm-start data-gen.

Writes one short doc per line (first ~40 words) to out/corpus.txt — short because
data_gen truncates to 64 tokens and we want varied contexts, not long articles.
Falls back to WikiText-103 if FineWeb streaming is unavailable.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "out" / "corpus.txt"


def shorten(text: str, max_words: int = 40) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:max_words])


def stream(n: int):
    try:
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
        key = "text"
        src = "FineWeb sample-10BT"
    except Exception as e:
        print(f"FineWeb unavailable ({e}); falling back to WikiText-103")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
        key = "text"
        src = "WikiText-103"
    print(f"streaming from {src} ...")
    seen = 0
    for ex in ds:
        t = shorten(ex.get(key, ""))
        if len(t.split()) >= 12:  # skip stubs
            yield t
            seen += 1
            if seen >= n:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, doc in enumerate(stream(args.n)):
            f.write(doc + "\n")
            if i and i % 500 == 0:
                print(f"  {i} docs")
    n = sum(1 for _ in open(OUT))
    print(f"wrote {n} docs -> {OUT}")


if __name__ == "__main__":
    main()
