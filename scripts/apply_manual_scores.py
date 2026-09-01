"""
Merges manual_scores.csv (accumulated, append-only record of decided
scores) into generation_scoring_template_prefilled.csv to produce
generation_scoring_working_copy.csv.

Idempotent and safe to re-run: always regenerates the working copy fresh
from the template + the current manual_scores.csv, rather than editing
any file in place. The template and the frozen raw execution log
(generation_final_test_log.csv) are never written to.

LDR-03's exclusion_flag=conflict_case is carried through unchanged from
the template (this script does not alter exclusion_flag) -- remember to
exclude those rows from primary aggregate metrics when computing summary
statistics later, regardless of what scores they end up with.

Usage:
    python apply_manual_scores.py
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="generation_scoring_template_prefilled.csv")
    ap.add_argument("--scores", default="manual_scores.csv")
    ap.add_argument("--out", default="generation_scoring_working_copy.csv")
    args = ap.parse_args()

    with open(args.template, encoding="utf-8") as f:
        template_rows = list(csv.DictReader(f))
        fieldnames = list(template_rows[0].keys())

    with open(args.scores, encoding="utf-8") as f:
        score_rows = list(csv.DictReader(f))
    scores_by_key = {(r["question_id"], r["mode"]): r for r in score_rows}

    score_fields = ["factual_correctness", "groundedness_faithfulness", "refusal_behaviour",
                     "hallucination_claim_count", "citation_correctness", "scorer_notes"]

    n_scored = 0
    n_conflict_case_scored = 0
    out_rows = []
    for r in template_rows:
        key = (r["question_id"], r["mode"])
        out_row = dict(r)
        if key in scores_by_key:
            score_row = scores_by_key[key]
            for field in score_fields:
                out_row[field] = score_row.get(field, "")
            n_scored += 1
            if r["exclusion_flag"].strip():
                n_conflict_case_scored += 1
        out_rows.append(out_row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Working copy saved -> {args.out}")
    print(f"Scored so far: {n_scored} of {len(out_rows)} rows ({n_scored/len(out_rows)*100:.0f}%)")
    if n_conflict_case_scored:
        print(f"  ({n_conflict_case_scored} of those carry exclusion_flag -- scored normally, "
              f"but remember to exclude from primary aggregate metrics)")
    print(f"\nTemplate ({args.template}) and frozen execution log are UNCHANGED.")


if __name__ == "__main__":
    main()
