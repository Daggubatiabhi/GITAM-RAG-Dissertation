"""
Searches the FROZEN cleaned corpus (data/clean) for student-support/policy
content distinct from the residential-life pages already used elsewhere
in this evaluation set.

Usage:
    python search_corpus_policies.py
"""

import argparse
import glob
import json

URL_KEYWORDS = [
    "policy", "policies", "grievance", "discipline", "disciplinary",
    "conduct", "attendance", "leave", "counsel", "wellness", "mental-health",
    "anti-ragging", "ragging", "regulation",
]
TEXT_KEYWORDS = [
    "Grievance Redressal", "Code of Conduct", "Disciplinary Committee",
    "attendance policy", "leave policy", "student support", "counselling",
    "mental health", "anti-ragging committee", "Ombudsperson",
]

ap = argparse.ArgumentParser()
ap.add_argument("--clean-dir", default="data/clean")
ap.add_argument("--exclude-urls", nargs="*", default=[
    "https://www.gitam.edu/faqs",
    "https://www.gitam.edu/visakhapatnam/campus-life/residential-life",
    "https://www.gitam.edu/bengaluru/campus-life/residential-life",
    "https://www.gitam.edu/hyderabad/campus-life/transport-facilities",
    "https://www.gitam.edu/library/library-rules-regulations",
])
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
