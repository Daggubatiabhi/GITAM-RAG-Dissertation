"""
Final analysis package -- builds every table requested for the results
write-up, computed strictly from the 87 PRIMARY rows (exclusion_flag
blank; LDR-03's 3 rows reported separately, never folded into any
primary number). Produces numbers only -- no interpretation/discussion.

Data sources:
    generation_scoring_working_copy.csv  -- scores, category, exclusion_flag,
                                             model_refused, latency fields
                                             (carried through from the raw log)
    threshold_calibration_sweep.csv       -- calibration-set threshold sweep
    final_test_threshold_summary.csv      -- final-test tau=0.70/0.65 results

Outputs (all saved as CSV, also printed):
    1. final_pkg_aggregate_table.csv        -- Mode A/B/C, n per metric
    2. final_pkg_per_category_factual.csv   -- factual correctness by category x mode
    3. final_pkg_outcome_counts.csv         -- correct/refusal/false-refusal/false-accept counts, B & C
    4. final_pkg_latency.csv                -- median/p95 latency by mode
    5. final_pkg_calibration_vs_final.csv   -- calibration vs final-test threshold comparison
    6. final_pkg_qualitative_cases.csv      -- the 6 representative cases, full text

Usage:
    python build_final_analysis_package.py
"""

import argparse
import csv
import statistics


def safe_mean(values: list):
    numeric = [float(v) for v in values if v not in ("", "N/A", None)]
    return statistics.mean(numeric) if numeric else None


def percentile(values: list, pct: float):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


MODES = ["A_no_rag", "B_standard_rag", "C_thresholded_rag"]
MODE_LABELS = {"A_no_rag": "Mode A (No-RAG)", "B_standard_rag": "Mode B (Standard RAG)",
               "C_thresholded_rag": "Mode C (Thresholded RAG)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working-copy", default="generation_scoring_working_copy.csv")
    ap.add_argument("--calib-sweep", default="threshold_calibration_sweep.csv")
    ap.add_argument("--final-test-thresholds", default="final_test_threshold_summary.csv")
    args = ap.parse_args()

    with open(args.working_copy, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unscored = [r for r in rows if not r["factual_correctness"].strip()]
    if unscored:
        print(f"WARNING: {len(unscored)} unscored row(s) -- results INCOMPLETE.")
        for r in unscored:
            print(f"  {r['question_id']} / {r['mode']}")
        print()

    primary = [r for r in rows if not r["exclusion_flag"].strip()]
    excluded = [r for r in rows if r["exclusion_flag"].strip()]

    # =====================================================================
    # TABLE 1: Aggregate table, n reported per metric
    # =====================================================================
    print("=" * 100)
    print(f"TABLE 1: PRIMARY AGGREGATE RESULTS (n={len(primary)} rows / {len(primary)//3} questions, "
          f"NOTE: 'factual correctness' = performance on this frozen GITAM domain-QA set only)")
    print("=" * 100)

    agg_rows = []
    for mode in MODES:
        mr = [r for r in primary if r["mode"] == mode]
        n_total = len(mr)

        factual_vals = [r["factual_correctness"] for r in mr]
        n_factual = sum(1 for v in factual_vals if v not in ("", "N/A"))
        mean_factual = safe_mean(factual_vals)

        ground_vals = [r["groundedness_faithfulness"] for r in mr]
        n_ground = sum(1 for v in ground_vals if v not in ("", "N/A"))
        mean_ground = safe_mean(ground_vals)

        refusal_vals = [r["refusal_behaviour"].strip().lower() for r in mr]
        n_correct = sum(1 for v in refusal_vals if v == "correct")
        n_incorrect = sum(1 for v in refusal_vals if v == "incorrect")
        n_refusal_scored = n_correct + n_incorrect
        refusal_acc = n_correct / n_refusal_scored if n_refusal_scored > 0 else None

        halluc_vals = [r["hallucination_claim_count"] for r in mr]
        n_halluc = sum(1 for v in halluc_vals if v not in ("", "N/A"))
        mean_halluc = safe_mean(halluc_vals)

        citation_vals = [r["citation_correctness"] for r in mr]
        n_citation = sum(1 for v in citation_vals if v not in ("", "N/A"))
        mean_citation = safe_mean(citation_vals)

        refused_vals = [r["model_refused"].strip().lower() for r in mr]
        n_refused = sum(1 for v in refused_vals if v == "true")
        refusal_rate = n_refused / n_total if n_total > 0 else None

        agg_rows.append({
            "mode": MODE_LABELS[mode], "n_total": n_total,
            "mean_factual_correctness": mean_factual, "n_factual": n_factual,
            "mean_groundedness": mean_ground, "n_groundedness": n_ground,
            "refusal_behaviour_accuracy": refusal_acc, "n_refusal_scored": n_refusal_scored,
            "mean_hallucination_count": mean_halluc, "n_hallucination": n_halluc,
            "mean_citation_correctness": mean_citation, "n_citation": n_citation,
            "refusal_rate": refusal_rate, "n_refusal_rate": n_total,
        })

    def fmt(x, d=3):
        return f"{x:.{d}f}" if x is not None else "n/a"

    for r in agg_rows:
        print(f"\n{r['mode']}  (n_total={r['n_total']})")
        print(f"  Factual correctness (0-2, this GITAM QA set): {fmt(r['mean_factual_correctness'])}  (n={r['n_factual']})")
        print(f"  Groundedness (0-2):                            {fmt(r['mean_groundedness'])}  (n={r['n_groundedness']})")
        print(f"  Refusal-behaviour accuracy:                    {fmt(r['refusal_behaviour_accuracy'])}  (n={r['n_refusal_scored']})")
        print(f"  Hallucination claim count (mean):              {fmt(r['mean_hallucination_count'])}  (n={r['n_hallucination']})")
        print(f"  Citation correctness (0-2):                    {fmt(r['mean_citation_correctness'])}  (n={r['n_citation']})")
        print(f"  Refusal rate:                                  {fmt(r['refusal_rate'])}  (n={r['n_refusal_rate']})")

    with open("final_pkg_aggregate_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # =====================================================================
    # TABLE 2: Per-category factual correctness, all 3 modes
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 2: PER-CATEGORY FACTUAL CORRECTNESS (mean, 0-2 scale)\n{'='*100}")
    categories = sorted({r["category"] for r in primary})
    cat_rows = []
    header = f"{'Category':35}" + "".join(f"{MODE_LABELS[m]:28}" for m in MODES)
    print(header)
    for cat in categories:
        row_out = {"category": cat}
        line = f"{cat:35}"
        for mode in MODES:
            vals = [r["factual_correctness"] for r in primary if r["category"] == cat and r["mode"] == mode]
            m = safe_mean(vals)
            n = len(vals)
            row_out[f"{mode}_mean"] = m
            row_out[f"{mode}_n"] = n
            line += f"{fmt(m):>8} (n={n:<2})          "
        print(line)
        cat_rows.append(row_out)

    with open("final_pkg_per_category_factual.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader()
        w.writerows(cat_rows)

    # =====================================================================
    # TABLE 3: Outcome-type counts for B and C
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 3: OUTCOME-TYPE COUNTS (Modes B and C)\n{'='*100}")
    print("Categories (mutually exclusive, partition all rows per mode):")
    print("  appropriate_answer_decision = answered (not refused) AND refusal_behaviour=correct")
    print("                                 (the DECISION to answer was appropriate --")
    print("                                  NOT a claim that factual_correctness=2;")
    print("                                  answer quality is reported separately in Table 1/2)")
    print("  correct_refusal  = refused AND refusal_behaviour=correct")
    print("  false_refusal    = refused AND refusal_behaviour=incorrect")
    print("  false_accept     = answered (not refused) AND refusal_behaviour=incorrect (unsupported answer)")

    outcome_rows = []
    for mode in ["B_standard_rag", "C_thresholded_rag"]:
        mr = [r for r in primary if r["mode"] == mode]
        appropriate_answer_decision = [r for r in mr if r["model_refused"].strip().lower() == "false"
                           and r["refusal_behaviour"].strip().lower() == "correct"]
        correct_refusal = [r for r in mr if r["model_refused"].strip().lower() == "true"
                            and r["refusal_behaviour"].strip().lower() == "correct"]
        false_refusal = [r for r in mr if r["model_refused"].strip().lower() == "true"
                          and r["refusal_behaviour"].strip().lower() == "incorrect"]
        false_accept = [r for r in mr if r["model_refused"].strip().lower() == "false"
                         and r["refusal_behaviour"].strip().lower() == "incorrect"]

        print(f"\n{MODE_LABELS[mode]}  (n={len(mr)})")
        print(f"  appropriate_answer_decision: {len(appropriate_answer_decision)}   {[r['question_id'] for r in appropriate_answer_decision]}")
        print(f"  correct_refusal:             {len(correct_refusal)}   {[r['question_id'] for r in correct_refusal]}")
        print(f"  false_refusal:                {len(false_refusal)}   {[r['question_id'] for r in false_refusal]}")
        print(f"  false_accept:                 {len(false_accept)}   {[r['question_id'] for r in false_accept]}")

        outcome_rows.append({
            "mode": MODE_LABELS[mode], "n": len(mr),
            "appropriate_answer_decision_n": len(appropriate_answer_decision),
            "correct_refusal_n": len(correct_refusal),
            "false_refusal_n": len(false_refusal),
            "false_accept_n": len(false_accept),
            "appropriate_answer_decision_ids": ";".join(r["question_id"] for r in appropriate_answer_decision),
            "correct_refusal_ids": ";".join(r["question_id"] for r in correct_refusal),
            "false_refusal_ids": ";".join(r["question_id"] for r in false_refusal),
            "false_accept_ids": ";".join(r["question_id"] for r in false_accept),
        })

    with open("final_pkg_outcome_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(outcome_rows[0].keys()))
        w.writeheader()
        w.writerows(outcome_rows)

    # =====================================================================
    # TABLE 4: Latency (median, p95) by mode
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 4: LATENCY (median, p95, milliseconds)\n{'='*100}")
    lat_rows = []
    for mode in MODES:
        mr = [r for r in primary if r["mode"] == mode]
        gen_lat = [float(r["generation_latency_ms"]) for r in mr if r.get("generation_latency_ms")]
        ret_lat = [float(r["retrieval_latency_ms"]) for r in mr if r.get("retrieval_latency_ms")]
        tot_lat = [float(r["total_latency_ms"]) for r in mr if r.get("total_latency_ms")]

        row_out = {
            "mode": MODE_LABELS[mode], "n": len(mr),
            "gen_median_ms": statistics.median(gen_lat) if gen_lat else None,
            "gen_p95_ms": percentile(gen_lat, 0.95) if gen_lat else None,
            "retrieval_median_ms": statistics.median(ret_lat) if ret_lat else None,
            "retrieval_p95_ms": percentile(ret_lat, 0.95) if ret_lat else None,
            "total_median_ms": statistics.median(tot_lat) if tot_lat else None,
            "total_p95_ms": percentile(tot_lat, 0.95) if tot_lat else None,
        }
        print(f"\n{row_out['mode']}  (n={row_out['n']})")
        print(f"  Generation:  median={fmt(row_out['gen_median_ms'],1)}ms  p95={fmt(row_out['gen_p95_ms'],1)}ms")
        print(f"  Retrieval:   median={fmt(row_out['retrieval_median_ms'],1)}ms  p95={fmt(row_out['retrieval_p95_ms'],1)}ms")
        print(f"  Total:       median={fmt(row_out['total_median_ms'],1)}ms  p95={fmt(row_out['total_p95_ms'],1)}ms")
        lat_rows.append(row_out)

    with open("final_pkg_latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lat_rows[0].keys()))
        w.writeheader()
        w.writerows(lat_rows)

    # =====================================================================
    # TABLE 4b: SUPPLEMENTAL -- generation latency, Mistral-INVOKED rows only
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 4b: SUPPLEMENTAL -- GENERATION LATENCY, MISTRAL-INVOKED ROWS ONLY\n{'='*100}")
    print("(Excludes Mode C rows where the threshold gate failed and Mistral was never called --")
    print(" those log generation_latency_ms=0 and would otherwise understate true per-call cost,")
    print(" especially for Mode C. Table 4 above is UNCHANGED and still reports all-query latency.)")

    lat4b_rows = []
    for mode in MODES:
        mr = [r for r in primary if r["mode"] == mode]
        invoked = [r for r in mr if r.get("generation_latency_ms") and float(r["generation_latency_ms"]) > 0]
        gen_lat = [float(r["generation_latency_ms"]) for r in invoked]

        row_out = {
            "mode": MODE_LABELS[mode], "n_total": len(mr), "n_invoked": len(invoked),
            "gen_median_ms_invoked_only": statistics.median(gen_lat) if gen_lat else None,
            "gen_p95_ms_invoked_only": percentile(gen_lat, 0.95) if gen_lat else None,
        }
        print(f"\n{row_out['mode']}  (invoked {row_out['n_invoked']} of {row_out['n_total']} rows)")
        print(f"  Generation (invoked only): median={fmt(row_out['gen_median_ms_invoked_only'],1)}ms  "
              f"p95={fmt(row_out['gen_p95_ms_invoked_only'],1)}ms")
        lat4b_rows.append(row_out)

    with open("final_pkg_latency_invoked_only.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lat4b_rows[0].keys()))
        w.writeheader()
        w.writerows(lat4b_rows)

    # =====================================================================
    # TABLE 5: Calibration vs final-test threshold comparison
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 5: CALIBRATION vs FINAL-TEST THRESHOLD COMPARISON\n{'='*100}")
    calib_vs_final_rows = []
    try:
        with open(args.calib_sweep, encoding="utf-8") as f:
            calib_sweep = list(csv.DictReader(f))
        calib_070 = next((r for r in calib_sweep if abs(float(r["threshold"]) - 0.70) < 1e-6), None)
        calib_065 = next((r for r in calib_sweep if abs(float(r["threshold"]) - 0.65) < 1e-6), None)

        with open(args.final_test_thresholds, encoding="utf-8") as f:
            final_rows = list(csv.DictReader(f))
        final_070 = next((r for r in final_rows if r["threshold_label"] == "PRIMARY" and r["subgroup"] == "overall"), None)
        final_065 = next((r for r in final_rows if "SECONDARY" in r["threshold_label"] and r["subgroup"] == "overall"), None)

        for label, calib, final in [("tau=0.70", calib_070, final_070), ("tau=0.65", calib_065, final_065)]:
            if calib and final:
                row_out = {
                    "threshold": label,
                    "calib_precision": calib["precision"], "final_precision": final["precision"],
                    "calib_recall": calib["recall"], "final_recall": final["recall"],
                    "calib_f1": calib["f1"], "final_f1": final["f1"],
                    "calib_far": calib["false_acceptance_rate"], "final_far": final["false_acceptance_rate"],
                    "calib_frr": calib["false_refusal_rate"], "final_frr": final["false_refusal_rate"],
                    "calib_coverage": calib["answer_coverage"], "final_coverage": final["answer_coverage"],
                }
                print(f"\n{label}")
                print(f"  {'Metric':12} {'Calibration':>12} {'Final-test':>12}")
                for metric in ["precision", "recall", "f1", "false_acceptance_rate", "false_refusal_rate", "answer_coverage"]:
                    print(f"  {metric:12} {calib[metric]:>12} {final[metric]:>12}")
                calib_vs_final_rows.append(row_out)
    except FileNotFoundError as e:
        print(f"  Could not load threshold comparison files: {e}")

    if calib_vs_final_rows:
        with open("final_pkg_calibration_vs_final.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(calib_vs_final_rows[0].keys()))
            w.writeheader()
            w.writerows(calib_vs_final_rows)

    # =====================================================================
    # TABLE 6: Qualitative cases -- 6 representative examples, one per category
    # =====================================================================
    print(f"\n\n{'='*100}\nTABLE 6: QUALITATIVE CASES (extracted, not interpreted)\n{'='*100}")
    by_key = {(r["question_id"], r["mode"]): r for r in rows}  # use ALL rows so LDR-03 is reachable

    case_specs = [
        ("No-RAG hallucination", "ADM-06", "A_no_rag"),
        ("Successful RAG grounding", "ADM-07", "B_standard_rag"),
        ("Generator self-refusal (gate passed, model still refused)", "OOD-09", "C_thresholded_rag"),
        ("Threshold false refusal", "POL-04", "C_thresholded_rag"),
        ("Entity substitution", "OOD-08", "A_no_rag"),
        ("Corpus/source ambiguity (excluded from primary aggregates)", "LDR-03", "B_standard_rag"),
    ]

    qual_rows = []
    for label, qid, mode in case_specs:
        r = by_key.get((qid, mode))
        if r is None:
            print(f"\n[{label}] {qid}/{mode}: NOT FOUND")
            continue
        print(f"\n--- {label}: {qid} / {mode} ---")
        print(f"  Question: {r['question']}")
        print(f"  Generated: {r['generated_answer'][:250]}")
        print(f"  Scores: factual={r['factual_correctness']} groundedness={r['groundedness_faithfulness']} "
              f"refusal={r['refusal_behaviour']} hallucination={r['hallucination_claim_count']} "
              f"citation={r['citation_correctness']}")
        print(f"  Scorer note: {r['scorer_notes'][:300]}")
        qual_rows.append({
            "category_illustrated": label, "question_id": qid, "mode": mode,
            "question": r["question"], "generated_answer": r["generated_answer"],
            "factual_correctness": r["factual_correctness"],
            "groundedness_faithfulness": r["groundedness_faithfulness"],
            "refusal_behaviour": r["refusal_behaviour"],
            "hallucination_claim_count": r["hallucination_claim_count"],
            "citation_correctness": r["citation_correctness"],
            "scorer_notes": r["scorer_notes"],
        })

    if qual_rows:
        with open("final_pkg_qualitative_cases.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(qual_rows[0].keys()))
            w.writeheader()
            w.writerows(qual_rows)
    else:
        print("\nWARNING: none of the hardcoded qualitative case IDs were found in this data -- "
              "final_pkg_qualitative_cases.csv NOT written. Check --working-copy points to the "
              "real 90-row file.")

    # =====================================================================
    # LDR-03 (excluded) reported separately
    # =====================================================================
    print(f"\n\n{'='*100}\nEXCLUDED FROM ALL PRIMARY TABLES ABOVE: {excluded[0]['question_id'] if excluded else 'none'} "
          f"({len(excluded)} rows)\n{'='*100}")
    for r in excluded:
        print(f"\n  {r['question_id']} / {r['mode']}")
        print(f"    factual={r['factual_correctness']}  groundedness={r['groundedness_faithfulness']}  "
              f"refusal={r['refusal_behaviour']}  hallucination={r['hallucination_claim_count']}  "
              f"citation={r['citation_correctness']}")

    print(f"\n\nAll 6 CSV files saved. No interpretation/discussion included -- numbers only, for review.")


if __name__ == "__main__":
    main()
