"""
Verifies ABSENCE of specific facts across the ENTIRE frozen corpus
(data/clean) -- not just the one page originally suspected of lacking
them. This matters because a fact could plausibly be phrased on a
different, unexpected page even if it's missing from the obvious one.

For each candidate, searches all cleaned pages for a set of keywords/
phrases. Reports every match found (with snippet + URL) so a surprising
hit isn't silently missed. "Verified absent" means NONE of the keywords
for that candidate matched anywhere in the corpus -- this is evidence of
absence via a reasonable targeted search, not an absolute guarantee
(different phrasing elsewhere is always possible).

Usage:
    python verify_absence.py
"""

import glob
import json

CANDIDATES = [
    {
        "id": "OOD-U1",
        "description": "GIMSR Anti-Ragging Committee disciplinary action against a first-time offender",
        "keywords": ["first-time offender", "first offence", "first time offender",
                     "repeat offender", "first offense"],
    },
    {
        "id": "OOD-U2",
        "description": "GITAM Research Admissions portal PhD application deadline for 2026-27",
        "keywords": ["researchadmissions", "PhD application deadline 2026",
                     "Apply Before 2026", "research admissions 2026-27"],
    },
    {
        "id": "OOD-U3",
        "description": "Password reset process on apply.gitam.edu application portal",
        "keywords": ["apply.gitam.edu", "forgot password", "reset your password",
                     "password reset"],
    },
    {
        "id": "OOD-U4",
        "description": "Individual subject-wise minimum marks (not aggregate) for Architecture UG admission",
        "keywords": ["minimum marks in Mathematics", "individual subject minimum",
                     "subject-wise eligibility", "minimum marks in Physics for Architecture"],
    },
    {
        "id": "OOD-U5",
        "description": "Exact date of the GAT 2027 entrance exam",
        "keywords": ["GAT 2027", "2027 exam date", "GAT UG 2027", "2027 entrance exam date"],
    },
    {
        "id": "OOD-U6",
        "description": "Exact rupee scholarship output from the Scholarship Calculator for a ₹9 lakh combined income",
        "keywords": ["9 lakh", "\u20b99,00,000", "9,00,000 combined income",
                     "scholarship amount for", "calculated scholarship"],
    },
]


def main():
    files = glob.glob("data/clean/*.json")
    print(f"Searching {len(files)} cleaned pages for {len(CANDIDATES)} candidates...\n")

    corpus = []
    for f in files:
        rec = json.load(open(f, encoding="utf-8"))
        corpus.append((rec["url"], rec["clean_text"]))

    for cand in CANDIDATES:
        print(f"=== {cand['id']}: {cand['description']} ===")
        hits = []
        for url, text in corpus:
            text_lower = text.lower()
            for kw in cand["keywords"]:
                idx = text_lower.find(kw.lower())
                if idx != -1:
                    start = max(0, idx - 60)
                    end = min(len(text), idx + 150)
                    snippet = text[start:end].replace("\n", " ")
                    hits.append((url, kw, snippet))

        if hits:
            print(f"  [FOUND {len(hits)} match(es) -- NOT confirmed absent]")
            for url, kw, snippet in hits:
                print(f"    {url}")
                print(f"      matched \"{kw}\": ...{snippet}...")
        else:
            print(f"  [NO MATCHES -- verified absent by targeted search across full corpus]")
        print()


if __name__ == "__main__":
    main()
