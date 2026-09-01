"""
Search a FAISS index built by embed_index.py.

Embeds your query with the SAME model used to build the index, then
retrieves the top-k most similar chunks by cosine similarity.

This is the piece worth testing carefully before wiring up the LLM --
if retrieval doesn't surface the right chunks for a question, no amount
of generator prompting will fix the answer.

Usage:
    python search_index.py --index data/index/idx_300_60 --query "What is the placement rate for computer science?"
    python search_index.py --index data/index/idx_300_60 --query "Who leads the physics department?" --k 5
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser(description="Search a FAISS chunk index.")
    ap.add_argument("--index", required=True, help="Directory containing index.faiss + metadata.jsonl")
    ap.add_argument("--query", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=3, help="Number of results to return")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="Must match the model used to build the index")
    ap.add_argument("--threshold", type=float, default=None,
                     help="If set, drop results with cosine similarity below this (0-1). "
                          "Useful for testing calibrated abstention thresholds.")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers faiss-cpu")
    try:
        import faiss
    except ImportError:
        raise SystemExit("faiss-cpu is not installed. Run: pip install sentence-transformers faiss-cpu")

    index_dir = Path(args.index)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    metadata = load_metadata(index_dir / "metadata.jsonl")

    print(f"Loaded index: {index.ntotal} vectors, {len(metadata)} metadata records")
    if index.ntotal != len(metadata):
        print("  [warn] index size and metadata count don't match -- rebuild the index")

    model = SentenceTransformer(args.model)
    query_vec = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    scores, indices = index.search(query_vec, args.k)
    scores, indices = scores[0], indices[0]

    print(f"\nQuery: {args.query}\n")

    shown = 0
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        if args.threshold is not None and score < args.threshold:
            continue
        rec = metadata[idx]
        shown += 1
        print(f"--- #{shown}  score={score:.3f}  {rec['source_url']}")
        print(rec["text"][:400])
        print()

    if shown == 0:
        if args.threshold is not None:
            print(f"No results above threshold {args.threshold} -- this is what a "
                  f"calibrated-abstention system would treat as 'not enough evidence'.")
        else:
            print("No results found.")


if __name__ == "__main__":
    main()
