"""
Step 3 -- Formal retrieval evaluation.

Runs all 45 frozen, split-assigned questions against BOTH indexes
(idx_128_25_minilm, idx_240_40_minilm), retrieves top-5 for each, and
computes Hit@1/3/5 and MRR for the 36 ANSWERABLE questions only (per
instructions -- unanswerable questions are retrieved and logged for
observation, but do not enter the Hit@k/MRR calculation).

Multi-source questions (FAC-02, POL-02: two source URLs from a cross-page
comparison) are scored two ways:
    - "at least one source found in top-5" (lenient)
    - "all sources found in top-5"          (strict)
both saved separately, since neither alone fully captures whether
retrieval actually supports the cross-page answer.

Does NOT tune the 0.4 threshold, does NOT touch Mistral, does NOT modify
chunking/embeddings/FAISS/the evaluation questions based on results --
purely measures and reports.

Outputs:
    retrieval_results_detailed.csv   -- one row per (question, config, rank),
                                         i.e. up to 45 * 2 * 5 = 450 rows
    retrieval_metrics_summary.csv    -- Hit@1/3/5, MRR broken down by
                                         config / split / category
    retrieval_multisource_summary.csv -- at-least-one vs all-sources-found
                                          for the 2 multi-source questions

Usage:
    python run_formal_evaluation.py
"""

import argparse
import csv
import json
import time
from pathlib import Path


def load_metadata(index_dir: Path) -> list[dict]:
    records = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_source_urls(row) -> list[str]:
    raw = row["expected_source_url"].strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(";") if u.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="evaluation_questions_split.csv")
    ap.add_argument("--indexes", nargs="+", default=[
        "data/index/idx_128_25_minilm",
        "data/index/idx_240_40_minilm",
    ])
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--detailed-out", default="retrieval_results_detailed.csv")
    ap.add_argument("--metrics-out", default="retrieval_metrics_summary.csv")
    ap.add_argument("--multisource-out", default="retrieval_multisource_summary.csv")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers faiss-cpu")
    try:
        import faiss
    except ImportError:
        raise SystemExit("faiss-cpu is not installed. Run: pip install sentence-transformers faiss-cpu")

    with open(args.questions, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    print(f"Loaded {len(questions)} questions from {args.questions}")

    model = SentenceTransformer(args.model)

    detailed_rows = []
    # per-question, per-config: rank of first correct hit (None if not found in top-k)
    first_hit_rank: dict[tuple[str, str], int | None] = {}
    # per-question, per-config: which of the expected sources (if multiple) were found
    sources_found: dict[tuple[str, str], set] = {}

    for index_path in args.indexes:
        index_dir = Path(index_path)
        config_name = index_dir.name  # e.g. "idx_128_25_minilm"

        print(f"\n{'='*70}\nEvaluating against: {config_name}\n{'='*70}")

        index = faiss.read_index(str(index_dir / "index.faiss"))
        metadata = load_metadata(index_dir)
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        chunk_config = f"{manifest.get('chunk_size_setting')}/{manifest.get('chunk_overlap_setting')}"

        for qi, row in enumerate(questions, start=1):
            qid = row["question_id"]
            question_text = row["question"]
            expected_urls = get_source_urls(row)

            t0 = time.time()
            query_vec = model.encode([question_text], normalize_embeddings=True,
                                      convert_to_numpy=True).astype("float32")
            scores, indices = index.search(query_vec, args.k)
            latency_ms = (time.time() - t0) * 1000
            scores, indices = scores[0], indices[0]

            key = (qid, config_name)
            first_hit_rank[key] = None
            sources_found[key] = set()

            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx == -1:
                    continue
                rec = metadata[idx]
                retrieved_url = rec["source_url"]
                source_match = retrieved_url in expected_urls

                if source_match:
                    sources_found[key].add(retrieved_url)
                    if first_hit_rank[key] is None:
                        first_hit_rank[key] = rank

                detailed_rows.append({
                    "question_id": qid,
                    "question": question_text,
                    "category": row["category"],
                    "split": row["split"],
                    "configuration": chunk_config,
                    "index_dir": config_name,
                    "answerable": row["answerable"],
                    "expected_source_urls": " ; ".join(expected_urls) if expected_urls else "",
                    "retrieved_rank": rank,
                    "chunk_id": rec["chunk_id"],
                    "retrieved_url": retrieved_url,
                    "similarity_score": float(score),
                    "retrieved_text": rec["text"],
                    "source_match": source_match,
                    "retrieval_latency_ms": round(latency_ms, 2),
                })

            if qi % 10 == 0 or qi == len(questions):
                print(f"  {qi}/{len(questions)} questions processed...")

    # Save detailed results
    with open(args.detailed_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"\nDetailed results saved -> {args.detailed_out} ({len(detailed_rows)} rows)")

    # --- Metrics computation ---
    q_by_id = {r["question_id"]: r for r in questions}
    configs = sorted({r["index_dir"] for r in detailed_rows})

    def compute_metrics(qids: list[str], config_name: str) -> dict:
        """Hit@1/3/5 and MRR over the given question IDs, for ANSWERABLE questions only."""
        answerable_qids = [qid for qid in qids if q_by_id[qid]["answerable"] == "TRUE"]
        n = len(answerable_qids)
        if n == 0:
            return {"n": 0, "hit@1": None, "hit@3": None, "hit@5": None, "mrr": None}

        hit1 = hit3 = hit5 = 0
        rr_sum = 0.0
        for qid in answerable_qids:
            rank = first_hit_rank.get((qid, config_name))
            if rank is not None:
                rr_sum += 1.0 / rank
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
        return {
            "n": n,
            "hit@1": hit1 / n,
            "hit@3": hit3 / n,
            "hit@5": hit5 / n,
            "mrr": rr_sum / n,
        }

    all_qids = [r["question_id"] for r in questions]
    metrics_rows = []

    for config_name in configs:
        # Overall
        m = compute_metrics(all_qids, config_name)
        metrics_rows.append({"config": config_name, "scope_type": "overall", "scope": "all", **m})

        # By split
        for split_val in ["calibration", "final_test"]:
            qids = [r["question_id"] for r in questions if r["split"] == split_val]
            m = compute_metrics(qids, config_name)
            metrics_rows.append({"config": config_name, "scope_type": "split", "scope": split_val, **m})

        # By category
        categories = sorted({r["category"] for r in questions})
        for cat in categories:
            qids = [r["question_id"] for r in questions if r["category"] == cat]
            m = compute_metrics(qids, config_name)
            metrics_rows.append({"config": config_name, "scope_type": "category", "scope": cat, **m})

    with open(args.metrics_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "scope_type", "scope", "n", "hit@1", "hit@3", "hit@5", "mrr"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Metrics summary saved -> {args.metrics_out}")

    # --- Multi-source questions (at-least-one vs all-sources-found) ---
    multisource_qids = [r["question_id"] for r in questions
                         if r["answerable"] == "TRUE" and len(get_source_urls(r)) > 1]
    multisource_rows = []
    for qid in multisource_qids:
        expected = set(get_source_urls(q_by_id[qid]))
        for config_name in configs:
            found = sources_found.get((qid, config_name), set())
            multisource_rows.append({
                "question_id": qid,
                "config": config_name,
                "expected_source_count": len(expected),
                "sources_found_count": len(found),
                "at_least_one_found": len(found) >= 1,
                "all_sources_found": found == expected,
            })
    if multisource_rows:
        with open(args.multisource_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(multisource_rows[0].keys()))
            writer.writeheader()
            writer.writerows(multisource_rows)
        print(f"Multi-source summary saved -> {args.multisource_out} ({len(multisource_qids)} question(s): {multisource_qids})")

    # --- Print headline summary to terminal ---
    print(f"\n{'='*70}\nHEADLINE METRICS (overall, answerable questions only)\n{'='*70}")
    for row in metrics_rows:
        if row["scope_type"] == "overall":
            print(f"\n{row['config']}  (n={row['n']} answerable questions)")
            print(f"  Hit@1: {row['hit@1']:.3f}  Hit@3: {row['hit@3']:.3f}  "
                  f"Hit@5: {row['hit@5']:.3f}  MRR: {row['mrr']:.3f}")

    print(f"\n{'='*70}\nLATENCY SUMMARY\n{'='*70}")
    import statistics
    for config_name in configs:
        latencies = [r["retrieval_latency_ms"] for r in detailed_rows
                     if r["index_dir"] == config_name and r["retrieved_rank"] == 1]
        if latencies:
            print(f"\n{config_name}  (n={len(latencies)} queries)")
            print(f"  mean: {statistics.mean(latencies):.1f}ms  "
                  f"median: {statistics.median(latencies):.1f}ms  "
                  f"min: {min(latencies):.1f}ms  max: {max(latencies):.1f}ms")

    print(f"\nDone. See {args.detailed_out}, {args.metrics_out}, {args.multisource_out} "
          f"for full breakdowns (per-split, per-category included).")


if __name__ == "__main__":
    main()
