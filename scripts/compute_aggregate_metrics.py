"""
Final aggregate comparison: Mode A (No-RAG) vs Mode B (Standard RAG) vs
Mode C (Thresholded RAG), computed from the completed
generation_scoring_working_copy.csv.

PRIMARY metrics are computed ONLY over rows where exclusion_flag is blank
(81 of 90 rows -- excludes LDR-03's 3 rows, the documented conflict case).
LDR-03 is reported separately afterward, never folded into the primary
numbers.

For each mode, reports:
    - mean factual_correctness (0-2 scale; always numeric in this dataset)
    - mean groundedness_faithfulness (0-2 scale; N/A rows excluded from the
      mean but counted separately -- N/A is expected for all of Mode A)
    - refusal_behaviour accuracy: correct / (correct + incorrect), among
      rows where it's not N/A (all of Mode A is N/A -- no refusal protocol)
    - mean hallucination_claim_count (always numeric, all modes)
    - mean citation_correctness (0-2 scale; N/A rows excluded from the mean
      but counted -- N/A expected for Mode A and for any refused B/C row)
    - refusal rate (fraction of rows where model_refused=True)

Usage:
    python compute_aggregate_metrics.py
"""

import argparse
import csv
import statistics


def safe_mean(values: list) -> float | None:
    numeric = [float(v) for v in values if v not in ("", "N/A", None)]
    return statistics.mean(numeric) if numeric else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working-copy", default="generation_scoring_working_copy.csv")
    ap.add_argument("--out", default="final_aggregate_results.csv")
    args = ap.parse_args()

    with open(args.working_copy, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unscored = [r for r in rows if not r["factual_correctness"].strip()]
    if unscored:
        print(f"WARNING: {len(unscored)} row(s) still unscored -- results below are INCOMPLETE.")
        for r in unscored:
            print(f"  missing: {r['question_id']} / {r['mode']}")
        print()

    primary = [r for r in rows if not r["exclusion_flag"].strip()]
    excluded = [r for r in rows if r["exclusion_flag"].strip()]

    print(f"Total rows: {len(rows)}  |  Primary (exclusion_flag blank): {len(primary)}  "
          f"|  Excluded: {len(excluded)}\n")

    modes = ["A_no_rag", "B_standard_rag", "C_thresholded_rag"]
    mode_labels = {"A_no_rag": "Mode A (No-RAG)", "B_standard_rag": "Mode B (Standard RAG)",
                    "C_thresholded_rag": "Mode C (Thresholded RAG)"}

    results = []
    for mode in modes:
        mode_rows = [r for r in primary if r["mode"] == mode]
        n = len(mode_rows)

        factual_vals = [r["factual_correctness"] for r in mode_rows]
        mean_factual = safe_mean(factual_vals)

        ground_vals = [r["groundedness_faithfulness"] for r in mode_rows]
        mean_ground = safe_mean(ground_vals)
        n_ground_na = sum(1 for v in ground_vals if v.strip() == "N/A")

        refusal_vals = [r["refusal_behaviour"].strip().lower() for r in mode_rows]
        n_correct = sum(1 for v in refusal_vals if v == "correct")
        n_incorrect = sum(1 for v in refusal_vals if v == "incorrect")
        n_refusal_na = sum(1 for v in refusal_vals if v == "n/a")
        refusal_accuracy = (n_correct / (n_correct + n_incorrect)
                             if (n_correct + n_incorrect) > 0 else None)

        halluc_vals = [r["hallucination_claim_count"] for r in mode_rows]
        mean_halluc = safe_mean(halluc_vals)

        citation_vals = [r["citation_correctness"] for r in mode_rows]
        mean_citation = safe_mean(citation_vals)
        n_citation_na = sum(1 for v in citation_vals if v.strip() == "N/A")

        refused_vals = [r["model_refused"].strip().lower() for r in mode_rows]
        refusal_rate = sum(1 for v in refused_vals if v == "true") / n if n > 0 else None

        results.append({
            "mode": mode_labels[mode], "n": n,
            "mean_factual_correctness": mean_factual,
            "mean_groundedness": mean_ground, "groundedness_na_count": n_ground_na,
            "refusal_accuracy": refusal_accuracy,
            "refusal_correct_count": n_correct, "refusal_incorrect_count": n_incorrect,
            "refusal_na_count": n_refusal_na,
            "mean_hallucination_count": mean_halluc,
            "mean_citation_correctness": mean_citation, "citation_na_count": n_citation_na,
            "refusal_rate": refusal_rate,
        })

    # --- Print formatted comparison table ---
    print("=" * 100)
    print("PRIMARY AGGREGATE RESULTS (exclusion_flag blank only, n=%d rows, %d questions)"
          % (len(primary), len(primary) // 3))
    print("=" * 100)

    def fmt(x, decimals=3):
        return f"{x:.{decimals}f}" if x is not None else "n/a"

    for r in results:
        print(f"\n{r['mode']}  (n={r['n']})")
        print(f"  Mean factual correctness (0-2):     {fmt(r['mean_factual_correctness'])}")
        print(f"  Mean groundedness (0-2):             {fmt(r['mean_groundedness'])}"
              f"  ({r['groundedness_na_count']} N/A)")
        print(f"  Refusal behaviour accuracy:          {fmt(r['refusal_accuracy'])}"
              f"  (correct={r['refusal_correct_count']}, incorrect={r['refusal_incorrect_count']}, "
              f"N/A={r['refusal_na_count']})")
        print(f"  Mean hallucination claim count:      {fmt(r['mean_hallucination_count'])}")
        print(f"  Mean citation correctness (0-2):     {fmt(r['mean_citation_correctness'])}"
              f"  ({r['citation_na_count']} N/A)")
        print(f"  Refusal rate (fraction that refused): {fmt(r['refusal_rate'])}")

    # --- Save CSV ---
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n\nSaved -> {args.out}")

    # --- Excluded (LDR-03) reported separately ---
    if excluded:
        print(f"\n{'='*100}\nEXCLUDED FROM PRIMARY METRICS: {excluded[0]['question_id']} "
              f"({len(excluded)} rows, exclusion_flag={excluded[0]['exclusion_flag']})\n{'='*100}")
        for r in excluded:
            print(f"\n  {r['question_id']} / {r['mode']}")
            print(f"    factual={r['factual_correctness']}  groundedness={r['groundedness_faithfulness']}  "
                  f"refusal={r['refusal_behaviour']}  hallucination={r['hallucination_claim_count']}  "
                  f"citation={r['citation_correctness']}")


if __name__ == "__main__":
    main()
