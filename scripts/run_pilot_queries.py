"""
Runs the fixed pilot questions against a FAISS index and SAVES the top-5
results (chunk ID, URL, score) to CSV -- not terminal-only, per the
project's convention of recording experimental results as files.

4 of these 5 questions were actually run earlier in this project's manual
testing (fee structure, physics placement, CS department head, Antarctica
out-of-domain). The 5th (admissions) was NOT previously tested -- it is
added here as a new question for category diversity (fees/placements/
department-faculty/out-of-domain were covered; admissions was not), and is
labelled as such in the output rather than misrepresented as reused.

Usage:
    python run_pilot_queries.py --index data/index/idx_128_25_minilm
    python run_pilot_queries.py --index data/index/idx_240_40_minilm
"""

import argparse
import csv
import json
import time
from pathlib import Path

PILOT_QUESTIONS = [
    {"id": "fee_structure", "question": "What is the fee structure for engineering?",
     "category": "fees", "provenance": "previously tested"},
    {"id": "physics_placement", "question": "What is the placement rate for physics department?",
     "category": "placements", "provenance": "previously tested"},
    {"id": "cs_department_head", "question": "Who is the head of the computer science department?",
     "category": "department/faculty", "provenance": "previously tested"},
    {"id": "antarctica_weather", "question": "What is the weather like in Antarctica?",
     "category": "out-of-domain", "provenance": "previously tested"},
    {"id": "admissions_new", "question": "What are the admission requirements for GITAM?",
     "category": "admissions",
     "provenance": "NEW -- verified before use against cleaned corpus: "
                    "https://www.gitam.edu/faqs (28,619 chars, confirmed contains GAT exam / "
                    "eligibility / merit-scholarship content answering this question)",
     "expected_source_url": "https://www.gitam.edu/faqs"},
]


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser(description="Run fixed pilot questions against a FAISS index, save results to CSV.")
    ap.add_argument("--index", required=True, help="Directory containing index.faiss + metadata.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <index>/pilot_query_results.csv)")
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
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    print(f"Index: {index_dir}  |  {index.ntotal} vectors  |  "
          f"config: {manifest.get('chunk_size_setting')}/{manifest.get('chunk_overlap_setting')}")

    model = SentenceTransformer(args.model)

    rows = []
    for pq in PILOT_QUESTIONS:
        t0 = time.time()
        query_vec = model.encode([pq["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, indices = index.search(query_vec, args.k)
        latency_ms = (time.time() - t0) * 1000
        scores, indices = scores[0], indices[0]

        print(f"\n--- {pq['id']} [{pq['category']}, {pq['provenance']}] "
              f"({latency_ms:.0f}ms) ---")
        print(f"Q: {pq['question']}")

        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:
                continue
            rec = metadata[idx]
            print(f"  #{rank}  score={score:.3f}  {rec['source_url']}  ({rec['chunk_id']})")
            rows.append({
                "question_id": pq["id"],
                "question": pq["question"],
                "category": pq["category"],
                "provenance": pq["provenance"],
                "expected_source_url": pq.get("expected_source_url"),
                "rank": rank,
                "chunk_id": rec["chunk_id"],
                "source_url": rec["source_url"],
                "matched_expected_source": rec["source_url"] == pq.get("expected_source_url"),
                "page_title": rec.get("page_title"),
                "score": float(score),
                "latency_ms": round(latency_ms, 1),
                "index_config": f"{manifest.get('chunk_size_setting')}/{manifest.get('chunk_overlap_setting')}",
            })

    out_path = Path(args.out) if args.out else index_dir / "pilot_query_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
