"""
Generates the blank rubric-scoring template from the raw execution log.
Per generation_rubric.md: execution facts (what the system did) and
evaluation judgments (how it was scored) are kept in SEPARATE files. This
script only READS the execution log -- it never modifies it.

Usage:
    python prepare_scoring_template.py
    python prepare_scoring_template.py --input generation_final_test_log.csv
"""

import argparse
import csv

RUBRIC_COLUMNS = [
    "factual_correctness",
    "groundedness_faithfulness",
    "refusal_behaviour",
    "hallucination_claim_count",
    "hallucination_claim_notes",
    "citation_correctness",
    "scorer_notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generation_final_test_log.csv")
    ap.add_argument("--out", default="generation_scoring_template.csv")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for col in RUBRIC_COLUMNS:
            r[col] = ""

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_excluded = sum(1 for r in rows if r.get("exclusion_flag"))
    print(f"Scoring template saved -> {args.out}  ({len(rows)} rows)")
    if n_excluded:
        print(f"  {n_excluded} row(s) carry an exclusion_flag (e.g. LDR-03/conflict_case) -- "
              f"still score them normally, but remember to exclude them when computing "
              f"primary aggregate metrics later.")
    print(f"\nFill in these columns by hand, referencing evaluation_questions_frozen.csv "
          f"(expected_answer/evidence_text) and generation_rubric.md (scale definitions):")
    for col in RUBRIC_COLUMNS:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
