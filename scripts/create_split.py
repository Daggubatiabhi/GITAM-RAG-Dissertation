"""
Step 2 -- Create the fixed calibration/final-test split.

Target allocation (computed to sum exactly to 15/30 while keeping every
category represented proportionally):
    5-question categories (admissions, fees, programmes, complex):    2 calib / 3 test
    4-question categories (placements, leadership, facilities, policies): 1 calib / 3 test
    out-of-domain (9 total = 3 out_of_domain + 6 in_domain_unsupported): 3 calib (1+2) / 6 test (2+4)
    Total: 15 calib / 30 test

Grouping rule: questions that share a source URL are grouped (via shared-URL
graph) and kept in the SAME split where this doesn't conflict with the
category targets above. Category/negative-type balance is treated as the
harder constraint per the brief ("preserve diversity... keep same-source
together WHERE PRACTICAL") -- so a group spanning multiple categories is
deliberately split by category need instead, and reported explicitly as
an exception rather than silently overridden.

Deterministic: no randomness. Groups/questions processed in a fixed
(question_id, alphabetical) order, so re-running produces an identical
split every time -- required since the split must be frozen BEFORE seeing
any retrieval results.

Usage:
    python create_split.py
    python create_split.py --input evaluation_questions_frozen.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CALIB_TARGET_BY_CATEGORY = {
    "admissions/eligibility": 2,
    "fees/scholarships": 2,
    "programmes/course details": 2,
    "complex/multi-sentence": 2,
    "placements/careers": 1,
    "leadership/faculty": 1,
    "campus facilities/services": 1,
    "policies/student support": 1,
    # out-of-domain handled separately via negative_type sub-targets below
}
CALIB_TARGET_OOD_OUT_OF_DOMAIN = 1
CALIB_TARGET_OOD_IN_DOMAIN_UNSUPPORTED = 2


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def get_source_urls(row) -> list[str]:
    raw = row["expected_source_url"].strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(";") if u.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="evaluation_questions_frozen.csv")
    ap.add_argument("--out", default="evaluation_questions_split.csv")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row_by_id = {r["question_id"]: r for r in rows}

    # Build shared-source-URL groups
    uf = UnionFind([r["question_id"] for r in rows])
    url_to_first_qid: dict[str, str] = {}
    for r in rows:
        for url in get_source_urls(r):
            if url in url_to_first_qid:
                uf.union(r["question_id"], url_to_first_qid[url])
            else:
                url_to_first_qid[url] = r["question_id"]

    groups: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(row_by_id.keys()):
        groups[uf.find(qid)].append(qid)
    group_list = [sorted(members) for members in groups.values()]
    group_list.sort(key=lambda g: g[0])  # deterministic order

    # Identify groups spanning >1 category -- these are the "where practical"
    # exceptions where grouping and category-balance conflict.
    cross_category_groups = []
    single_category_groups = []
    for g in group_list:
        cats = {row_by_id[qid]["category"] for qid in g}
        if len(cats) > 1:
            cross_category_groups.append(g)
        else:
            single_category_groups.append(g)

    # Running remaining calibration targets per category
    remaining_calib = dict(CALIB_TARGET_BY_CATEGORY)
    remaining_calib["out-of-domain"] = (CALIB_TARGET_OOD_OUT_OF_DOMAIN
                                         + CALIB_TARGET_OOD_IN_DOMAIN_UNSUPPORTED)
    remaining_calib_ood_type = {
        "out_of_domain": CALIB_TARGET_OOD_OUT_OF_DOMAIN,
        "in_domain_unsupported": CALIB_TARGET_OOD_IN_DOMAIN_UNSUPPORTED,
    }

    split_assignment: dict[str, str] = {}
    exceptions_log = []

    # Pass 1: single-category groups of size > 1 -- try to keep together
    for g in single_category_groups:
        if len(g) == 1:
            continue
        cat = row_by_id[g[0]]["category"]
        if cat == "out-of-domain":
            continue  # handled individually in pass 3 (negative_type sub-targets)
        if remaining_calib.get(cat, 0) >= len(g):
            for qid in g:
                split_assignment[qid] = "calibration"
            remaining_calib[cat] -= len(g)
        else:
            for qid in g:
                split_assignment[qid] = "final_test"

    # Pass 2: cross-category groups -- explicitly split by each member's own
    # category target rather than forcing the whole group into one split.
    for g in cross_category_groups:
        cats_in_group = sorted({row_by_id[qid]["category"] for qid in g})
        exceptions_log.append(
            f"Group {g} spans categories {cats_in_group} (shares a source URL) -- "
            f"split by individual category target rather than kept together, "
            f"since category balance is the harder constraint."
        )
        for qid in g:
            cat = row_by_id[qid]["category"]
            if remaining_calib.get(cat, 0) > 0:
                split_assignment[qid] = "calibration"
                remaining_calib[cat] -= 1
            else:
                split_assignment[qid] = "final_test"

    # Pass 3: remaining singleton groups (including all out-of-domain
    # questions, handled via negative_type sub-targets)
    for g in group_list:
        for qid in g:
            if qid in split_assignment:
                continue
            row = row_by_id[qid]
            cat = row["category"]
            if cat == "out-of-domain":
                neg_type = row["negative_type"]
                if remaining_calib_ood_type.get(neg_type, 0) > 0:
                    split_assignment[qid] = "calibration"
                    remaining_calib_ood_type[neg_type] -= 1
                else:
                    split_assignment[qid] = "final_test"
            else:
                if remaining_calib.get(cat, 0) > 0:
                    split_assignment[qid] = "calibration"
                    remaining_calib[cat] -= 1
                else:
                    split_assignment[qid] = "final_test"

    # Write output with split column added
    fieldnames = list(rows[0].keys()) + ["split"]
    for r in rows:
        r["split"] = split_assignment[r["question_id"]]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Report ---
    calib = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "final_test"]

    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    print(f"\nCalibration: {len(calib)}  |  Final test: {len(test)}")

    from collections import Counter
    print(f"\nCalibration by category:")
    for cat, n in sorted(Counter(r["category"] for r in calib).items()):
        print(f"  {cat}: {n}")
    print(f"\nFinal test by category:")
    for cat, n in sorted(Counter(r["category"] for r in test).items()):
        print(f"  {cat}: {n}")

    calib_ood = [r for r in calib if r["category"] == "out-of-domain"]
    test_ood = [r for r in test if r["category"] == "out-of-domain"]
    print(f"\nCalibration out-of-domain by negative_type:")
    for nt, n in sorted(Counter(r["negative_type"] for r in calib_ood).items()):
        print(f"  {nt}: {n}")
    print(f"Final test out-of-domain by negative_type:")
    for nt, n in sorted(Counter(r["negative_type"] for r in test_ood).items()):
        print(f"  {nt}: {n}")

    if exceptions_log:
        print(f"\n--- Source-grouping exceptions ({len(exceptions_log)}) ---")
        for e in exceptions_log:
            print(f"  {e}")

    multi_member_groups_kept = [g for g in single_category_groups if len(g) > 1]
    if multi_member_groups_kept:
        print(f"\n--- Same-source groups kept together ({len(multi_member_groups_kept)}) ---")
        for g in multi_member_groups_kept:
            splits_in_group = {split_assignment[qid] for qid in g}
            status = "OK (same split)" if len(splits_in_group) == 1 else "SPLIT (target conflict)"
            print(f"  {g} -> {status}")

    print(f"\nSaved -> {args.out}")
    print(f"\nThis split is now FROZEN -- do not regenerate after seeing retrieval results.")


if __name__ == "__main__":
    main()
