"""
Show the FULL text of specific short pages, so we can judge whether they're
genuinely thin content (fine) or another extraction gap (needs fixing).

Usage:
    python inspect_short_pages.py
"""
import json

URLS_TO_CHECK = [
    "https://www.gitam.edu/discipline/computer-science",
    "https://www.gitam.edu/discipline/pharmacy",
    "https://www.gitam.edu/gimsr/leadership",
    "https://www.gitam.edu/academics/academic-policies",
    "https://www.gitam.edu/vdc/careers",
]

import glob
for f in glob.glob("data/clean/*.json"):
    rec = json.load(open(f, encoding="utf-8"))
    if rec["url"] in URLS_TO_CHECK:
        print(f"=== {rec['url']}  ({rec['clean_text_length']} chars) ===")
        print(rec["clean_text"])
        print("\n" + "=" * 60 + "\n")
