"""
Searches the FROZEN cleaned corpus (data/clean) for admissions-related
content on pages OTHER than /faqs -- for evaluation-question source
diversity. Does not touch chunking/embeddings/FAISS/retrieval in any way;
this is a plain-text search over the cleaned page JSON files.

Two search strategies, both reported:
    1. URL path contains an admissions-related keyword (e.g. "admission")
    2. Page text contains an admissions-related keyword, regardless of URL

Excludes https://www.gitam.edu/faqs by default (that's the page we
already have covered) so what's left shows genuine alternative sources.

Usage:
    python search_corpus_admissions.py
"""

import argparse
import glob
import json
import re

URL_KEYWORDS = ["admission", "eligibility", "apply", "how-to-apply", "entrance"]
TEXT_KEYWORDS = [
    "GAT", "GITAM Admission Test", "eligibility criteria", "entrance exam",
    "admission process", "admission requirements", "eligible to apply",
    "qualifying mark", "application fee", "admission test",
]

ap = argparse.ArgumentParser()
ap.add_argument("--clean-dir", default="data/clean")
ap.add_argument("--exclude-url", default="https://www.gitam.edu/faqs")
ap.add_argument("--snippet-chars", type=int, default=200)
args = ap.parse_args()

url_matches = []
text_matches = []

for f in glob.glob(f"{args.clean_dir}/*.json"):
    rec = json.load(open(f, encoding="utf-8"))
    url = rec["url"]
    text = rec["clean_text"]

    if url == args.exclude_url:
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
            break  # one match per page is enough to flag it as a candidate

print(f"=== URL-pattern matches (path contains an admissions keyword) ===")
print(f"{len(url_matches)} found\n")
for url, length in sorted(url_matches, key=lambda x: -x[1]):
    print(f"  [{length:5d} chars]  {url}")

print(f"\n=== Text-content matches (page mentions admissions terms) ===")
print(f"{len(text_matches)} found\n")
for url, length, kw, snippet in sorted(text_matches, key=lambda x: -x[1]):
    print(f"  [{length:5d} chars]  {url}")
    print(f"    matched \"{kw}\": ...{snippet}...")
    print()
