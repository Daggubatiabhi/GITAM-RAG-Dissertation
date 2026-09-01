"""
Threshold calibration -- CALIBRATION SPLIT ONLY, idx_128_25_minilm ONLY.

Reuses the already-frozen retrieval_results_detailed.csv (retrieval itself
is NOT re-run) joined against evaluation_questions_split.csv. The 30
final-test questions are never touched by this script.

Two distinct concepts, per spec:
    question_answerable   -- ground truth from the dataset itself: does this
                              question have a real answer in the corpus at all?
    retrieval_supported    -- did the ACTUAL top-5 retrieval (unthresholded)
                              contain the full required evidence for this
                              specific question, regardless of any threshold?

retrieval_supported is computed once per question (independent of threshold):
    - answerable=FALSE (out-of-domain OR in-domain-unsupported) -> ALWAYS False,
      by definition, regardless of what got retrieved.
    - answerable=TRUE, single required source -> True iff that source appears
      anywhere in top-5 (source_match=True on any of its retrieved rows).
    - answerable=TRUE, multiple required sources (FAC-02/POL-02 style) -> True
      ONLY if ALL required sources appear in top-5. Partial retrieval (some
      but not all sources) = NOT supported, per spec.

Then for each threshold tau, the per-query SYSTEM DECISION is:
    accept (attempt to answer) if top-1 (best) retrieved score >= tau
    refuse (abstain)           if top-1 (best) retrieved score <  tau
(equivalent to "does anything in top-5 clear tau", since FAISS returns
results score-sorted descending).

Confusion matrix per threshold, over the 15 calibration questions:
    True Accept  (TA): accept AND retrieval_supported=True
    False Accept (FA): accept AND retrieval_supported=False  <- hallucination risk
    True Refusal (TR): refuse AND retrieval_supported=False
    False Refusal(FR): refuse AND retrieval_supported=True   <- coverage loss

Usage:
    python calibrate_threshold.py
"""

import argparse
import csv
from pathlib import Path


def get_expected_urls(row) -> set:
    raw = row["expected_source_urls"].strip()
    if not raw:
        return set()
    return {u.strip() for u in raw.split(";") if u.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="evaluation_questions_split.csv")
    ap.add_argument("--detailed-results", default="retrieval_results_detailed.csv")
    ap.add_argument("--config", default="idx_128_25_minilm",
                     help="Which index's results to use (must match index_dir in the detailed CSV)")
    ap.add_argument("--thresholds", nargs="+", type=float,
                     default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    ap.add_argument("--out", default="threshold_calibration_sweep.csv")
    args = ap.parse_args()

    with open(args.split_file, encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    calib_qids = {r["question_id"] for r in split_rows if r["split"] == "calibration"}
    print(f"Calibration set: {len(calib_qids)} questions")

    with open(args.detailed_results, encoding="utf-8") as f:
        detail_rows = list(csv.DictReader(f))
    detail_rows = [r for r in detail_rows
                   if r["index_dir"] == args.config and r["question_id"] in calib_qids]
    if not detail_rows:
        raise SystemExit(f"No detailed results found for config={args.config} "
                          f"restricted to the calibration set -- check --config matches "
                          f"the index_dir values in {args.detailed_results}.")

    # Group rows per question
    by_qid: dict[str, list[dict]] = {}
    for r in detail_rows:
        by_qid.setdefault(r["question_id"], []).append(r)

    # Per-question: top1 score, retrieval_supported ground truth
    per_question = {}
    for qid, rows in by_qid.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["retrieved_rank"]))
        top1_score = float(rows_sorted[0]["similarity_score"])
        answerable = rows_sorted[0]["answerable"] == "TRUE"
        category = rows_sorted[0]["category"]
        question_text = rows_sorted[0]["question"]

        if not answerable:
            supported = False
        else:
            expected_urls = get_expected_urls(rows_sorted[0])
            matched_urls = {r["retrieved_url"] for r in rows if r["source_match"] == "True"}
            supported = expected_urls.issubset(matched_urls) and len(expected_urls) > 0

        per_question[qid] = {
            "question_id": qid,
            "question": question_text,
            "category": category,
            "question_answerable": answerable,
            "retrieval_supported": supported,
            "top1_score": top1_score,
        }

    n_total = len(per_question)
    n_supported = sum(1 for q in per_question.values() if q["retrieval_supported"])
    n_unsupported = n_total - n_supported
    print(f"Of {n_total} calibration questions: {n_supported} retrieval_supported=True, "
          f"{n_unsupported} retrieval_supported=False")
    print(f"  (question_answerable=True: {sum(1 for q in per_question.values() if q['question_answerable'])}, "
          f"question_answerable=False: {sum(1 for q in per_question.values() if not q['question_answerable'])})\n")

    # Per-question detail table (useful for manual inspection)
    print(f"{'qid':10} {'answerable':11} {'supported':10} {'top1_score':11} category")
    for qid in sorted(per_question.keys()):
        q = per_question[qid]
        print(f"{qid:10} {str(q['question_answerable']):11} {str(q['retrieval_supported']):10} "
              f"{q['top1_score']:.3f}       {q['category']}")

    # --- Threshold sweep ---
    sweep_rows = []
    for tau in args.thresholds:
        ta = fa = tr = fr = 0
        for q in per_question.values():
            accept = q["top1_score"] >= tau
            supported = q["retrieval_supported"]
            if accept and supported:
                ta += 1
            elif accept and not supported:
                fa += 1
            elif not accept and not supported:
                tr += 1
            else:
                fr += 1

        precision = ta / (ta + fa) if (ta + fa) > 0 else None
        recall = ta / (ta + fr) if (ta + fr) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) > 0 else None)
        specificity = tr / (tr + fa) if (tr + fa) > 0 else None
        balanced_acc = ((recall + specificity) / 2
                         if recall is not None and specificity is not None else None)
        coverage = (ta + fa) / n_total if n_total > 0 else None
        far = fa / (fa + tr) if (fa + tr) > 0 else None  # false acceptance rate
        frr = fr / (fr + ta) if (fr + ta) > 0 else None  # false refusal rate

        sweep_rows.append({
            "threshold": tau,
            "true_accept": ta, "false_accept": fa,
            "true_refusal": tr, "false_refusal": fr,
            "precision": precision, "recall": recall, "f1": f1,
            "balanced_accuracy": balanced_acc,
            "answer_coverage": coverage,
            "false_acceptance_rate": far,
            "false_refusal_rate": frr,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    print(f"\n{'='*100}")
    print(f"THRESHOLD SWEEP (calibration set only, n={n_total}, config={args.config})")
    print(f"{'='*100}")
    header = f"{'tau':>5} {'TA':>4} {'FA':>4} {'TR':>4} {'FR':>4} {'Prec':>6} {'Recall':>7} {'F1':>6} {'BalAcc':>7} {'Coverage':>9} {'FAR':>6} {'FRR':>6}"
    print(header)
    for r in sweep_rows:
        def fmt(x):
            return f"{x:.3f}" if x is not None else "  n/a"
        print(f"{r['threshold']:>5.2f} {r['true_accept']:>4} {r['false_accept']:>4} "
              f"{r['true_refusal']:>4} {r['false_refusal']:>4} "
              f"{fmt(r['precision']):>6} {fmt(r['recall']):>7} {fmt(r['f1']):>6} "
              f"{fmt(r['balanced_accuracy']):>7} {fmt(r['answer_coverage']):>9} "
              f"{fmt(r['false_acceptance_rate']):>6} {fmt(r['false_refusal_rate']):>6}")

    print(f"\nSaved full sweep -> {args.out}")
    print(f"\nNOTE: this is calibration-set-only analysis. Final-test questions were NOT used.")
    print(f"No threshold has been selected/applied -- this is descriptive output for review.")


if __name__ == "__main__":
    main()
