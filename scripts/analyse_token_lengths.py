"""
Task 1 -- MiniLM token-length / truncation analysis
------------------------------------------------------
Measures how the SentenceTransformer("all-MiniLM-L6-v2") tokenizer
ACTUALLY tokenizes each stored chunk -- not the tiktoken (cl100k_base)
counts chunk.py used to size chunks. tiktoken (BPE) and MiniLM's tokenizer
(WordPiece, bert-base-uncased vocab) are different tokenizers and do not
produce the same token counts for the same text, so this must be measured
directly against the real embedding pipeline, not assumed.

Does NOT modify the original chunk files -- read-only.

For each corpus (300/60, 600/100), reports:
    total chunks, min, mean, median, p90, p95, max MiniLM token length
    the model's effective max_seq_length
    count and percentage of chunks exceeding max_seq_length
    estimated average % of content lost among truncated chunks

Saves:
    results/token_length_summary.csv     -- one row per corpus (the headline numbers)
    results/token_length_details_<name>.csv  -- one row per chunk (full distribution,
                                                 for later dissertation analysis/plots)
    results/token_length_summary.json    -- same summary, machine-readable

Usage:
    python analyse_token_lengths.py
    python analyse_token_lengths.py --chunks data/chunks/chunks_300_60.jsonl data/chunks/chunks_600_100.jsonl
    python analyse_token_lengths.py --out-dir results
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def analyse_one(chunks_path: Path, tokenizer, max_len: int) -> tuple[dict, list[dict]]:
    chunks = load_chunks(chunks_path)
    n = len(chunks)

    # If the chunk file declares its own target size (chunk_minilm.py output
    # does; the older tiktoken-based files don't), report against that too --
    # this is the number that actually matters for "did chunking respect its
    # own design target", separate from the model's hard ceiling.
    declared_target = chunks[0].get("chunk_size_setting") if chunks else None

    per_chunk = []
    for c in chunks:
        # add_special_tokens=True to match exactly what SentenceTransformer
        # feeds the model at inference time ([CLS] ... [SEP])
        ids = tokenizer.encode(c["text"], add_special_tokens=True)
        wp_len = len(ids)
        per_chunk.append({
            "chunk_id": c["chunk_id"],
            "source_url": c["source_url"],
            "chunk_index": c["chunk_index"],
            "stored_token_count": c.get("token_count"),
            "minilm_wordpiece_count": wp_len,
            "truncated": wp_len > max_len,
        })

    lengths = np.array([r["minilm_wordpiece_count"] for r in per_chunk])
    truncated_mask = lengths > max_len
    n_truncated = int(truncated_mask.sum())

    if n_truncated > 0:
        loss_pct = ((lengths[truncated_mask] - max_len) / lengths[truncated_mask]) * 100
        avg_loss_pct = float(loss_pct.mean())
    else:
        avg_loss_pct = 0.0

    summary = {
        "corpus_file": str(chunks_path),
        "n_chunks": n,
        "min_wordpieces": int(lengths.min()) if n else 0,
        "mean_wordpieces": float(lengths.mean()) if n else 0.0,
        "median_wordpieces": float(np.median(lengths)) if n else 0.0,
        "p90_wordpieces": float(np.percentile(lengths, 90)) if n else 0.0,
        "p95_wordpieces": float(np.percentile(lengths, 95)) if n else 0.0,
        "max_wordpieces": int(lengths.max()) if n else 0,
        "model_max_seq_length": max_len,
        "n_truncated": n_truncated,
        "pct_truncated": (100 * n_truncated / n) if n else 0.0,
        "avg_pct_content_lost_among_truncated": avg_loss_pct,
        "declared_target_size": declared_target,
    }

    if declared_target is not None:
        over_target_mask = lengths > declared_target
        n_over_target = int(over_target_mask.sum())
        summary["n_over_declared_target"] = n_over_target
        summary["pct_over_declared_target"] = (100 * n_over_target / n) if n else 0.0

    return summary, per_chunk


def main():
    ap = argparse.ArgumentParser(description="MiniLM token-length/truncation analysis (Task 1).")
    ap.add_argument("--chunks", nargs="+",
                     default=["data/chunks/chunks_300_60.jsonl", "data/chunks/chunks_600_100.jsonl"],
                     help="One or more chunk .jsonl files to analyse")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers")

    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer
    max_len = model.get_max_seq_length()

    print(f"Model: {args.model}")
    print(f"Tokenizer: {type(tokenizer).__name__}")
    print(f"Effective max_seq_length: {max_len} word-pieces (incl. special tokens)\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for chunks_path_str in args.chunks:
        chunks_path = Path(chunks_path_str)
        if not chunks_path.exists():
            print(f"[skip] not found: {chunks_path}")
            continue

        print(f"=== {chunks_path} ===")
        summary, per_chunk = analyse_one(chunks_path, tokenizer, max_len)
        all_summaries.append(summary)

        print(f"  n_chunks:               {summary['n_chunks']}")
        print(f"  min wordpieces:         {summary['min_wordpieces']}")
        print(f"  mean wordpieces:        {summary['mean_wordpieces']:.1f}")
        print(f"  median wordpieces:      {summary['median_wordpieces']:.1f}")
        print(f"  p90 wordpieces:         {summary['p90_wordpieces']:.1f}")
        print(f"  p95 wordpieces:         {summary['p95_wordpieces']:.1f}")
        print(f"  max wordpieces:         {summary['max_wordpieces']}")
        if summary.get("declared_target_size") is not None:
            print(f"  chunks > {summary['declared_target_size']} (declared target): "
                  f"{summary['n_over_declared_target']} ({summary['pct_over_declared_target']:.1f}%)")
        print(f"  chunks > {max_len} (hard model limit):     {summary['n_truncated']} "
              f"({summary['pct_truncated']:.1f}%)")
        if summary["n_truncated"]:
            print(f"  avg % content lost among truncated: "
                  f"{summary['avg_pct_content_lost_among_truncated']:.1f}%")
        print()

        # Per-chunk detail CSV, for plotting/analysis in the dissertation
        detail_name = f"token_length_details_{chunks_path.stem}.csv"
        detail_path = out_dir / detail_name
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_chunk[0].keys()) if per_chunk else [])
            writer.writeheader()
            writer.writerows(per_chunk)
        print(f"  Saved per-chunk detail -> {detail_path}")
        print()

    # Combined summary CSV + JSON
    summary_csv = out_dir / "token_length_summary.csv"
    if all_summaries:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)

    summary_json = out_dir / "token_length_summary.json"
    summary_json.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print(f"Summary saved -> {summary_csv}")
    print(f"Summary saved -> {summary_json}")


if __name__ == "__main__":
    main()
