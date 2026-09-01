"""
Verifies that a specific candidate page (a) exists in the crawled/cleaned
corpus and (b) has substantive, non-thin content -- before it gets locked
in as a pilot query's ground-truth source.

Usage:
    python verify_pilot_candidate.py --url https://www.gitam.edu/faqs
"""

import argparse
import glob
import json

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--clean-dir", default="data/clean")
args = ap.parse_args()

found = None
for f in glob.glob(f"{args.clean_dir}/*.json"):
    rec = json.load(open(f, encoding="utf-8"))
    if rec["url"] == args.url:
        found = rec
        break

if not found:
    print(f"NOT FOUND in {args.clean_dir}: {args.url}")
    print("This URL was not crawled/cleaned -- cannot be used as a pilot query source.")
else:
    print(f"FOUND: {args.url}")
    print(f"Cleaned length: {found['clean_text_length']} chars")
    print(f"\nFull cleaned text:\n{'-'*60}")
    print(found["clean_text"])
    print(f"{'-'*60}")
    if found["clean_text_length"] < 400:
        print("\n[warn] This is a THIN page (<400 chars) -- may not contain enough "
              "detail to be a robust ground-truth source. Check the content above.")
    else:
        print("\nContent length looks substantive. Read the text above and confirm "
              "it genuinely answers the intended question before using this as a pilot source.")
