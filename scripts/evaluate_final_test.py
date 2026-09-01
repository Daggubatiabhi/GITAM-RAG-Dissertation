"""
Final-test threshold evaluation -- FINAL-TEST SPLIT ONLY (30 questions),
idx_128_25_minilm ONLY, two PRE-DECLARED thresholds (0.70 primary, 0.65
secondary sensitivity). No sweep. No threshold selection happens here --
both thresholds were fixed before this script ever looked at final-test
data, per the calibration-set-only selection already done.

retrieval_supported is computed with the exact same rule as calibration:
    - answerable=FALSE -> always False
    - answerable=TRUE, single source -> True iff that source in top-5
    - answerable=TRUE, multi-source -> True iff ALL sources in top-5
      (partial = not supported)

Reuses the already-frozen retrieval_results_detailed.csv -- retrieval
itself is not re-run, no re-querying of FAISS.

Usage:
    python evaluate_final_test.py
"""

import argparse
import csv


def get_expected_urls(row) -> set:
    raw = row["expected_source_urls"].strip()
    if not raw:
        return set()
    return {u.strip() for u in raw.split(";") if u.strip()}


def confusion_and_metrics(questions: list[dict], tau: float) -> dict:
    ta = fa = tr = fr = 0
    for q in questions:
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

    n = len(questions)
    precision = ta / (ta + fa) if (ta + fa) > 0 else None
    recall = ta / (ta + fr) if (ta + fr) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    specificity = tr / (tr + fa) if (tr + fa) > 0 else None
    balanced_acc = ((recall + specificity) / 2
                     if recall is not None and specificity is not None else None)
    coverage = (ta + fa) / n if n > 0 else None
    far = fa / (fa + tr) if (fa + tr) > 0 else None
    frr = fr / (fr + ta) if (fr + ta) > 0 else None

    return {
        "threshold": tau, "n": n,
        "true_accept": ta, "false_accept": fa, "true_refusal": tr, "false_refusal": fr,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": balanced_acc, "answer_coverage": coverage,
        "false_acceptance_rate": far, "false_refusal_rate": frr,
    }


def fmt(x):
    return f"{x:.3f}" if x is not None else "n/a"


def print_metrics_block(label: str, m: dict):
    print(f"\n{label}  (n={m['n']})")
    print(f"  TA={m['true_accept']}  FA={m['false_accept']}  "
          f"TR={m['true_refusal']}  FR={m['false_refusal']}")
    print(f"  Precision={fmt(m['precision'])}  Recall={fmt(m['recall'])}  F1={fmt(m['f1'])}  "
          f"BalAcc={fmt(m['balanced_accuracy'])}")
    print(f"  Coverage={fmt(m['answer_coverage'])}  FAR={fmt(m['false_acceptance_rate'])}  "
          f"FRR={fmt(m['false_refusal_rate'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="evaluation_questions_split.csv")
    ap.add_argument("--detailed-results", default="retrieval_results_detailed.csv")
    ap.add_argument("--config", default="idx_128_25_minilm")
    ap.add_argument("--primary-threshold", type=float, default=0.70)
    ap.add_argument("--secondary-threshold", type=float, default=0.65)
    ap.add_argument("--summary-out", default="final_test_threshold_summary.csv")
    ap.add_argument("--detail-out", default="final_test_threshold_detail.csv")
    args = ap.parse_args()

    with open(args.split_file, encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    test_qids = {r["question_id"] for r in split_rows if r["split"] == "final_test"}
    neg_type_by_qid = {r["question_id"]: r["negative_type"] for r in split_rows}
    print(f"Final-test set: {len(test_qids)} questions")

    with open(args.detailed_results, encoding="utf-8") as f:
        detail_rows = list(csv.DictReader(f))
    detail_rows = [r for r in detail_rows
                   if r["index_dir"] == args.config and r["question_id"] in test_qids]
    if not detail_rows:
        raise SystemExit(f"No detailed results found for config={args.config} restricted "
                          f"to the final-test set -- check --config value.")

    by_qid: dict[str, list[dict]] = {}
    for r in detail_rows:
        by_qid.setdefault(r["question_id"], []).append(r)

    questions = []
    for qid, rows in by_qid.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["retrieved_rank"]))
        top1_score = float(rows_sorted[0]["similarity_score"])
        answerable = rows_sorted[0]["answerable"] == "TRUE"
        category = rows_sorted[0]["category"]
        question_text = rows_sorted[0]["question"]
        neg_type = neg_type_by_qid.get(qid, "")

        if not answerable:
            supported = False
        else:
            expected_urls = get_expected_urls(rows_sorted[0])
            matched_urls = {r["retrieved_url"] for r in rows if r["source_match"] == "True"}
            supported = expected_urls.issubset(matched_urls) and len(expected_urls) > 0

        questions.append({
            "question_id": qid, "question": question_text, "category": category,
            "negative_type": neg_type, "question_answerable": answerable,
            "retrieval_supported": supported, "top1_score": top1_score,
        })

    n_total = len(questions)
    n_supported = sum(1 for q in questions if q["retrieval_supported"])
    print(f"Of {n_total} final-test questions: {n_supported} retrieval_supported=True, "
          f"{n_total - n_supported} retrieval_supported=False\n")

    # --- Subgroups ---
    supported_answerable = [q for q in questions if q["question_answerable"] and q["retrieval_supported"]]
    missed_answerable = [q for q in questions if q["question_answerable"] and not q["retrieval_supported"]]
    ood = [q for q in questions if q["negative_type"] == "out_of_domain"]
    idu = [q for q in questions if q["negative_type"] == "in_domain_unsupported"]

    print(f"Subgroup sizes: supported_answerable={len(supported_answerable)}, "
          f"missed_answerable={len(missed_answerable)}, out_of_domain={len(ood)}, "
          f"in_domain_unsupported={len(idu)}")

    thresholds = [("PRIMARY", args.primary_threshold), ("SECONDARY (sensitivity)", args.secondary_threshold)]

    summary_rows = []
    for label, tau in thresholds:
        print(f"\n{'='*70}\n{label}: tau = {tau}\n{'='*70}")

        m_overall = confusion_and_metrics(questions, tau)
        print_metrics_block("OVERALL (all 30 final-test questions)", m_overall)
        summary_rows.append({"threshold_label": label, "subgroup": "overall", **m_overall})

        for sub_label, sub_qs in [
            ("supported_answerable", supported_answerable),
            ("missed_answerable (evidence not retrieved)", missed_answerable),
            ("out_of_domain", ood),
            ("in_domain_unsupported", idu),
        ]:
            if not sub_qs:
                continue
            m_sub = confusion_and_metrics(sub_qs, tau)
            print_metrics_block(f"  Subgroup: {sub_label}", m_sub)
            summary_rows.append({"threshold_label": label, "subgroup": sub_label, **m_sub})

        # Explicitly call out false accepts among in_domain_unsupported
        idu_false_accepts = [q for q in idu if q["top1_score"] >= tau]
        print(f"\n  *** False accepts among in_domain_unsupported at tau={tau}: "
              f"{len(idu_false_accepts)} of {len(idu)} ***")
        for q in idu_false_accepts:
            print(f"    {q['question_id']}  score={q['top1_score']:.3f}  {q['question'][:70]}")

    # --- Save detail CSV (one row per question, with decisions at both thresholds) ---
    detail_out_rows = []
    for q in questions:
        row = dict(q)
        for label, tau in thresholds:
            key = f"decision_{label.split()[0].lower()}_{tau}"
            accept = q["top1_score"] >= tau
            if accept and q["retrieval_supported"]:
                outcome = "TA"
            elif accept and not q["retrieval_supported"]:
                outcome = "FA"
            elif not accept and not q["retrieval_supported"]:
                outcome = "TR"
            else:
                outcome = "FR"
            row[key] = outcome
        detail_out_rows.append(row)

    with open(args.detail_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_out_rows)

    with open(args.summary_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n\nSaved -> {args.summary_out}")
    print(f"Saved -> {args.detail_out}")
    print(f"\nNOTE: Both thresholds were pre-declared from calibration-only analysis. "
          f"No retuning has occurred based on these final-test results.")


if __name__ == "__main__":
    main()
