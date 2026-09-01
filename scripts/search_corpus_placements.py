"""
Searches the FROZEN cleaned corpus (data/clean) for a genuinely distinct
careers/placements source -- a dedicated hub page, not another department/
course page (we already have three of those for this category).

Two search strategies, both reported:
    1. URL path contains a careers/placements-related keyword
    2. Page text contains related terms, regardless of URL

Usage:
    python search_corpus_placements.py
"""

import argparse
import glob
import json

URL_KEYWORDS = [
    "career", "placement", "training", "employab", "recruit", "gcgc",
]
TEXT_KEYWORDS = [
    "Career Guidance Centre", "Career Guidance Center", "GCGC",
    "placement drive", "placement cell", "training and placement",
    "employability", "recruiters visit", "campus recruitment",
]

ap = argparse.ArgumentParser()
ap.add_argument("--clean-dir", default="data/clean")
ap.add_argument("--exclude-urls", nargs="*", default=[
    "https://www.gitam.edu/faqs",
    "https://www.gitam.edu/visakhapatnam/computer-science/course/master-of-computer-applications",
    "https://www.gitam.edu/bengaluru/artificial-intelligence-and-data-science/course/btech-computer-science-and-engineering-artificial-intelligence-and-machine-learning",
    "https://www.gitam.edu/hyderabad/aerospace-engineering/course/mtech-aerospace-engineering",
], help="URLs already used elsewhere in the evaluation set -- excluded so results show genuinely new candidates")
ap.add_argument("--snippet-chars", type=int, default=200)
args = ap.parse_args()

url_matches = []
text_matches = []

for f in glob.glob(f"{args.clean_dir}/*.json"):
    rec = json.load(open(f, encoding="utf-8"))
    url = rec["url"]
    text = rec["clean_text"]

    if url in args.exclude_urls:
        continue

    url_lower = url.lower()
    if any(kw in url_lower for kw in URL_KEYWORDS):
        url_matches.append((url, rec["clean_text_length"]))

    for kw in TEXT_KEYWORDS:
        idx = text.find(kw)
        if idx != -1:
            start = max(0, idx - 60)
            end = min(len(text), idx + args.snippet_chars)
            snippet = text[start:end].replace("\n", " ")
            text_matches.append((url, rec["clean_text_length"], kw, snippet))
            break

print(f"=== URL-pattern matches ===")
print(f"{len(url_matches)} found\n")
for url, length in sorted(url_matches, key=lambda x: -x[1]):
    print(f"  [{length:5d} chars]  {url}")

print(f"\n=== Text-content matches ===")
print(f"{len(text_matches)} found\n")
for url, length, kw, snippet in sorted(text_matches, key=lambda x: -x[1]):
    print(f"  [{length:5d} chars]  {url}")
    print(f"    matched \"{kw}\": ...{snippet}...")
    print()
