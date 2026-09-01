"""
Diagnostic pass on two specific pilot cases -- NO changes to chunking,
embeddings, FAISS, top-k, or the 0.4 threshold are made based on this.
Purely inspects what's actually happening, to distinguish:

    answer missing from chunks entirely  -> chunking-stage failure
    answer exists in a chunk, ranks low  -> embedding/retrieval-stage failure
    answer chunk ranks highly            -> (would be a generation-stage
                                              question, not applicable yet --
                                              no generator is wired up)

Case 1 -- CS department-head query:
    Shows the FULL TEXT of the top-5 retrieved chunks for both 128/25 and
    240/40. Checks for a structural leadership marker phrase ("Head of the
    Department" / "Department Leadership") -- the same phrase pattern
    confirmed present on other verified department pages (e.g. physics)
    earlier in this project. Presence of the marker is a signal to go read
    the printed text, not proof by itself that a name is actually there.

Case 2 -- Admissions query:
    1. Scans metadata.jsonl (same order as the FAISS index, so this is
       chunking-stage ground truth, independent of any search) for chunks
       from https://www.gitam.edu/faqs containing the verified answer
       phrase ("GAT (GITAM Admission Test)" or "appear for the GAT").
       Confirms these chunks exist in BOTH corpora before doing anything else.
    2. Runs an EXHAUSTIVE search (k = index.ntotal, not an arbitrary cutoff
       like 50/100) to find the EXACT rank and similarity score of each
       matched chunk for the admissions query -- not just whether it's in
       top-5.

Saves all results to CSV. Read-only against existing indexes -- builds
nothing, changes nothing.

Usage:
    python diagnose_pilot_cases.py
"""

import csv
import json
from pathlib import Path

CS_HEAD_QUERY = "Who is the head of the computer science department?"
LEADERSHIP_MARKERS = ["head of the department", "department leadership"]

ADMISSIONS_QUERY = "What are the admission requirements for GITAM?"
ADMISSIONS_EXPECTED_URL = "https://www.gitam.edu/faqs"
ADMISSIONS_KEY_PHRASES = ["gat (gitam admission test)", "appear for the gat"]

INDEXES = [
    {"name": "128/25", "dir": "data/index/idx_128_25_minilm"},
    {"name": "240/40", "dir": "data/index/idx_240_40_minilm"},
]


def load_metadata(index_dir: Path) -> list[dict]:
    records = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_marker(text: str, markers: list[str]) -> str | None:
    lower = text.lower()
    for m in markers:
        if m in lower:
            return m
    return None


def diagnose_department_head(index, metadata, model, config_name: str, rows: list[dict]):
    print(f"\n{'='*70}")
    print(f"CASE 1: CS department-head query -- config {config_name}")
    print(f"{'='*70}")

    query_vec = model.encode([CS_HEAD_QUERY], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(query_vec, 5)
    scores, indices = scores[0], indices[0]

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx == -1:
            continue
        rec = metadata[idx]
        marker = find_marker(rec["text"], LEADERSHIP_MARKERS)

        print(f"\n--- rank {rank}  score={score:.3f}  {rec['source_url']}  ({rec['chunk_id']})")
        print(f"contains leadership marker: {'YES (\"' + marker + '\")' if marker else 'NO'}")
        print(f"full text:\n{rec['text']}")

        rows.append({
            "diagnostic_type": "cs_department_head_top5",
            "query": CS_HEAD_QUERY,
            "config": config_name,
            "rank": rank,
            "chunk_id": rec["chunk_id"],
            "source_url": rec["source_url"],
            "score": float(score),
            "contains_leadership_marker": bool(marker),
            "matched_marker": marker or "",
            "text": rec["text"],
        })


def diagnose_admissions(index, metadata, model, config_name: str, rows: list[dict]):
    print(f"\n{'='*70}")
    print(f"CASE 2: Admissions query -- config {config_name}")
    print(f"{'='*70}")

    # Step 1: chunking-stage ground truth -- does the expected answer exist
    # as a chunk at all, independent of any search/ranking?
    matches = []
    for i, rec in enumerate(metadata):
        if rec["source_url"] == ADMISSIONS_EXPECTED_URL:
            phrase = find_marker(rec["text"], ADMISSIONS_KEY_PHRASES)
            if phrase:
                matches.append((i, rec, phrase))

    if not matches:
        print(f"[FAIL] No chunk from {ADMISSIONS_EXPECTED_URL} contains the expected "
              f"answer phrase in config {config_name} -- this would indicate a "
              f"CHUNKING-STAGE failure (answer lost before embedding).")
        rows.append({
            "diagnostic_type": "admissions_expected_chunk_rank",
            "query": ADMISSIONS_QUERY,
            "config": config_name,
            "rank": None,
            "chunk_id": None,
            "source_url": ADMISSIONS_EXPECTED_URL,
            "score": None,
            "contains_leadership_marker": None,
            "matched_marker": "NOT FOUND IN ANY CHUNK",
            "text": "",
        })
        return

    print(f"Confirmed: {len(matches)} chunk(s) from {ADMISSIONS_EXPECTED_URL} "
          f"contain the expected answer phrase (chunking stage OK).")

    # Step 2: EXHAUSTIVE search (k = ntotal) for the exact rank/score of
    # each matched chunk -- not an arbitrary top-N cutoff.
    query_vec = model.encode([ADMISSIONS_QUERY], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    k = index.ntotal
    scores, indices = index.search(query_vec, k)
    scores, indices = scores[0], indices[0]

    for i, rec, phrase in matches:
        pos = list(indices).index(i)  # exact position in the full ranked list
        rank = pos + 1
        score = float(scores[pos])

        verdict = ""
        if rank <= 5:
            verdict = "in top-5 (unexpected given earlier pilot run -- recheck)"
        elif rank <= 50:
            verdict = "ranks moderately low -- retrieval-stage weakness"
        else:
            verdict = "ranks very low relative to corpus size -- clear retrieval-stage failure"

        print(f"\nChunk {rec['chunk_id']}  (matched phrase: \"{phrase}\")")
        print(f"  Rank {rank} of {k}  |  score={score:.3f}  |  {verdict}")
        print(f"  full text:\n{rec['text']}")

        rows.append({
            "diagnostic_type": "admissions_expected_chunk_rank",
            "query": ADMISSIONS_QUERY,
            "config": config_name,
            "rank": rank,
            "chunk_id": rec["chunk_id"],
            "source_url": rec["source_url"],
            "score": score,
            "contains_leadership_marker": None,
            "matched_marker": phrase,
            "text": rec["text"],
        })


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers faiss-cpu")
    try:
        import faiss
    except ImportError:
        raise SystemExit("faiss-cpu is not installed. Run: pip install sentence-transformers faiss-cpu")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    rows: list[dict] = []

    for cfg in INDEXES:
        index_dir = Path(cfg["dir"])
        index = faiss.read_index(str(index_dir / "index.faiss"))
        metadata = load_metadata(index_dir)

        diagnose_department_head(index, metadata, model, cfg["name"], rows)
        diagnose_admissions(index, metadata, model, cfg["name"], rows)

    out_path = Path("results_minilm") / "diagnostic_pilot_cases.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*70}")
    print(f"Saved diagnostic output -> {out_path}")
    print("No changes made to chunking, embeddings, FAISS, top-k, or threshold.")


if __name__ == "__main__":
    main()
