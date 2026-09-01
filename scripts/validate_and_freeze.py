"""
Step 1 -- Validate and freeze the evaluation dataset.

Checks every criterion specified before treating the dataset as frozen:
    - exactly 45 total questions
    - 36 answerable / 9 unanswerable
    - 3 out_of_domain / 6 in_domain_unsupported
    - no duplicate question_ids or duplicate question text
    - every answerable question has a non-empty expected_answer, evidence_text,
      and expected_source_url
    - every unanswerable question uses expected_answer == INSUFFICIENT_EVIDENCE

If ALL checks pass: writes an exact copy to evaluation_questions_frozen.csv
and computes its SHA-256 hash, saved to evaluation_questions_frozen.sha256
(also printed). This hash is what proves later the dataset was not modified
after experiments started -- re-hashing the file at any future point and
comparing against this saved value detects any change, however small.

If ANY check fails: does NOT write the frozen file or hash, and reports
exactly what failed so it can be fixed first.

Usage:
    python validate_and_freeze.py
    python validate_and_freeze.py --input evaluation_questions.csv
"""

import argparse
import csv
import hashlib
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="evaluation_questions.csv")
    ap.add_argument("--frozen-out", default="evaluation_questions_frozen.csv")
    ap.add_argument("--hash-out", default="evaluation_questions_frozen.sha256")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors = []
    warnings = []

    # --- Total count ---
    if len(rows) != 45:
        errors.append(f"Expected 45 total questions, found {len(rows)}")

    # --- Answerable / unanswerable counts ---
    answerable = [r for r in rows if r["answerable"] == "TRUE"]
    unanswerable = [r for r in rows if r["answerable"] == "FALSE"]
    if len(answerable) != 36:
        errors.append(f"Expected 36 answerable questions, found {len(answerable)}")
    if len(unanswerable) != 9:
        errors.append(f"Expected 9 unanswerable questions, found {len(unanswerable)}")

    other_answerable_values = {r["answerable"] for r in rows} - {"TRUE", "FALSE"}
    if other_answerable_values:
        errors.append(f"Unexpected values in 'answerable' column: {other_answerable_values}")

    # --- Negative type counts ---
    ood = [r for r in unanswerable if r["negative_type"] == "out_of_domain"]
    idu = [r for r in unanswerable if r["negative_type"] == "in_domain_unsupported"]
    if len(ood) != 3:
        errors.append(f"Expected 3 out_of_domain questions, found {len(ood)}")
    if len(idu) != 6:
        errors.append(f"Expected 6 in_domain_unsupported questions, found {len(idu)}")

    bad_neg_types = [r["question_id"] for r in unanswerable
                      if r["negative_type"] not in ("out_of_domain", "in_domain_unsupported")]
    if bad_neg_types:
        errors.append(f"Unanswerable questions with invalid negative_type: {bad_neg_types}")

    non_empty_neg_type_on_answerable = [r["question_id"] for r in answerable if r["negative_type"]]
    if non_empty_neg_type_on_answerable:
        warnings.append(f"Answerable questions with non-blank negative_type "
                         f"(expected blank): {non_empty_neg_type_on_answerable}")

    # --- Duplicate IDs / questions ---
    ids = [r["question_id"] for r in rows]
    dup_ids = {i for i in ids if ids.count(i) > 1}
    if dup_ids:
        errors.append(f"Duplicate question_id(s): {dup_ids}")

    questions_text = [r["question"].strip().lower() for r in rows]
    dup_questions = {q for q in questions_text if questions_text.count(q) > 1}
    if dup_questions:
        errors.append(f"Duplicate question text (case-insensitive): {dup_questions}")

    # --- Answerable completeness ---
    incomplete_answerable = []
    for r in answerable:
        missing = []
        if not r["expected_answer"].strip():
            missing.append("expected_answer")
        if not r["evidence_text"].strip():
            missing.append("evidence_text")
        if not r["expected_source_url"].strip():
            missing.append("expected_source_url")
        if missing:
            incomplete_answerable.append((r["question_id"], missing))
    if incomplete_answerable:
        errors.append(f"Answerable questions missing required fields: {incomplete_answerable}")

    # --- Unanswerable expected_answer convention ---
    bad_neg_answer = [r["question_id"] for r in unanswerable
                       if r["expected_answer"].strip() != "INSUFFICIENT_EVIDENCE"]
    if bad_neg_answer:
        errors.append(f"Unanswerable questions NOT using expected_answer=INSUFFICIENT_EVIDENCE: {bad_neg_answer}")

    # --- Category counts (informational, matches the 8+1 category plan) ---
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)

    # --- Report ---
    print("=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"\nTotal questions: {len(rows)}")
    print(f"Answerable: {len(answerable)}  |  Unanswerable: {len(unanswerable)}")
    print(f"  out_of_domain: {len(ood)}  |  in_domain_unsupported: {len(idu)}")
    print(f"\nCategory breakdown:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat}: {n}")

    if warnings:
        print(f"\n--- WARNINGS ({len(warnings)}) ---")
        for w in warnings:
            print(f"  [warn] {w}")

    if errors:
        print(f"\n--- FAILED: {len(errors)} error(s) found ---")
        for e in errors:
            print(f"  [FAIL] {e}")
        print(f"\nDataset NOT frozen. Fix the errors above and re-run.")
        return

    print(f"\n--- ALL CHECKS PASSED ---")

    # Write frozen copy (byte-identical content, same columns/order)
    with open(args.input, encoding="utf-8") as f:
        content = f.read()
    Path(args.frozen_out).write_text(content, encoding="utf-8")

    # Compute SHA-256 of the frozen file
    sha256 = hashlib.sha256(Path(args.frozen_out).read_bytes()).hexdigest()
    Path(args.hash_out).write_text(sha256 + "\n", encoding="utf-8")

    print(f"\nFrozen copy saved -> {args.frozen_out}")
    print(f"SHA-256: {sha256}")
    print(f"Hash saved -> {args.hash_out}")
    print(f"\nTo verify the dataset hasn't changed later, run:")
    print(f'  python -c "import hashlib; print(hashlib.sha256(open(\'{args.frozen_out}\',\'rb\').read()).hexdigest())"')
    print(f"and compare against the value in {args.hash_out}.")


if __name__ == "__main__":
    main()
