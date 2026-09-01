"""
MiniLM-tokenizer-aligned chunking (revised approach, v2)
-------------------------------------------------------
v1 of this script sized chunks by SUMMING each sentence's independently-
measured token count. That additive approximation is not reliable: WordPiece
tokenization of a joined multi-sentence string is not guaranteed to equal
the sum of each sentence's isolated token count (boundary/merge effects),
and this let a 240-target chunk reach 257 tokens in practice -- over the
model's 256-token hard limit. v2 fixes this at the root: every packing
decision re-tokenizes the ACTUAL joined candidate text and checks its real
length, exactly the way split_long_unit() already did correctly in v1.

--size now means the FULL inference-time token budget, i.e. it includes
the model's special tokens ([CLS]/[SEP], 2 tokens for this model). A
"--size 128" chunk is guaranteed <= 128 tokens total once you run
tokenizer.encode(text, add_special_tokens=True) on it -- not 128 content
tokens plus 2 more on top. Internally this reserves 2 tokens of the budget
for specials (content budget = size - 2).

Guarantees enforced and verified (see the PASS/FAIL summary printed at the
end, and cross-checked again independently by analyse_token_lengths.py):
    - every chunk's real full-token count (with specials) <= --size
    - every chunk's real full-token count (with specials) <= model max_seq_length

Does NOT touch or overwrite the original tiktoken-based chunk files.

Usage:
    python chunk_minilm.py --clean data/clean --out data/chunks/chunks_128_25_minilm.jsonl --size 128 --overlap 25
    python chunk_minilm.py --clean data/clean --out data/chunks/chunks_240_40_minilm.jsonl --size 240 --overlap 40
"""

import argparse
import json
import re
from pathlib import Path

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SPECIAL_TOKEN_OVERHEAD = 2  # [CLS] + [SEP] for this model


def split_paragraphs_and_sentences(text: str) -> list[str]:
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s for s in SENTENCE_SPLIT.split(para) if s.strip())
    return sentences


def token_len(tokenizer, text_or_units) -> int:
    """TRUE token count of the actual joined text -- never an additive sum
    of independently-measured parts. Accepts a string or a list of strings
    (joined with spaces first)."""
    text = text_or_units if isinstance(text_or_units, str) else " ".join(text_or_units)
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_long_unit(text: str, tokenizer, content_budget: int) -> list[str]:
    """Splits a single overlong 'sentence' (e.g. an unpunctuated stat table
    or list block with heavy numeric/currency content, which tokenizes into
    many more wordpieces per word than ordinary prose) into word-level
    pieces, each verified by real re-tokenization of the actual joined
    candidate at every step."""
    words = text.split(" ")
    pieces, current = [], []

    for w in words:
        candidate = current + [w]
        if current and token_len(tokenizer, candidate) > content_budget:
            pieces.append(" ".join(current))
            current = [w]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))

    # Rare fallback: a single "word" alone still exceeds budget (e.g. a
    # long unbroken numeric string) -- hard-slice at the token-id level.
    # The main() loop independently re-verifies every final chunk anyway,
    # so if this ever drifts, it will be reported, not silently hidden.
    final_pieces = []
    for p in pieces:
        if token_len(tokenizer, p) > content_budget:
            ids = tokenizer.encode(p, add_special_tokens=False)
            for i in range(0, len(ids), content_budget):
                final_pieces.append(tokenizer.decode(ids[i:i + content_budget], skip_special_tokens=True))
        else:
            final_pieces.append(p)
    return final_pieces


def build_atomic_units(text: str, tokenizer, content_budget: int) -> list[str]:
    """Sentence-split, then further split any unit that alone exceeds the
    content budget -- verified via real joined-text tokenization -- so the
    packing step never has to force an oversized unit in."""
    units = []
    for s in split_paragraphs_and_sentences(text):
        if token_len(tokenizer, s) > content_budget:
            units.extend(split_long_unit(s, tokenizer, content_budget))
        else:
            units.append(s)
    return units


def pack_chunks(units: list[str], tokenizer, content_budget: int, overlap: int) -> list[str]:
    """Greedy packing where EVERY decision is based on the true joined-text
    token count of the actual candidate chunk -- not a sum of independently
    measured parts. This is what guarantees the final output respects the
    budget, since it's the same measurement the embedding model uses."""
    chunks = []
    current: list[str] = []

    for unit in units:
        candidate = current + [unit]
        if current and token_len(tokenizer, candidate) > content_budget:
            chunks.append(" ".join(current))

            # Build overlap by trimming from the back until the REAL joined
            # length of the retained tail fits the overlap budget.
            overlap_units: list[str] = []
            for u in reversed(current):
                candidate_overlap = [u] + overlap_units
                if token_len(tokenizer, candidate_overlap) > overlap:
                    break
                overlap_units = candidate_overlap

            # The overlap tail was only checked against the OVERLAP budget in
            # isolation -- but it also has to leave room for the incoming
            # unit within content_budget. If overlap_units + unit together
            # don't fit (a near-full-budget unit landing right after a
            # near-full overlap tail), shrink the overlap further, dropping
            # the OLDEST retained sentences first, until it does. The
            # incoming unit itself is guaranteed <= content_budget on its
            # own (build_atomic_units enforces this), so this loop is
            # guaranteed to terminate with a valid candidate.
            candidate = overlap_units + [unit]
            while overlap_units and token_len(tokenizer, candidate) > content_budget:
                overlap_units = overlap_units[1:]
                candidate = overlap_units + [unit]

            current = overlap_units

        current = candidate

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    ap = argparse.ArgumentParser(description="MiniLM-tokenizer-aligned chunking (v2, root-cause fixed).")
    ap.add_argument("--clean", default="data/clean", help="Directory of cleaned page JSON files")
    ap.add_argument("--out", required=True, help="Output JSONL path for this chunk setting")
    ap.add_argument("--size", type=int, required=True,
                     help="Target FULL token budget (including [CLS]/[SEP]) -- e.g. 128 or 240")
    ap.add_argument("--overlap", type=int, required=True,
                     help="Overlap in content tokens (not counted against the +2 special-token reserve)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers")

    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer
    max_seq_length = model.get_max_seq_length()

    if args.size > max_seq_length:
        raise SystemExit(f"--size {args.size} exceeds model max_seq_length ({max_seq_length}).")

    content_budget = args.size - SPECIAL_TOKEN_OVERHEAD
    if content_budget <= 0:
        raise SystemExit(f"--size {args.size} too small once {SPECIAL_TOKEN_OVERHEAD} special tokens are reserved.")

    print(f"Model: {args.model}  |  tokenizer: {type(tokenizer).__name__}  |  "
          f"max_seq_length: {max_seq_length}")
    print(f"Target FULL budget: {args.size} tokens (incl. specials)  |  "
          f"content budget: {content_budget}  |  overlap: {args.overlap} content tokens\n")

    clean_dir = Path(args.clean)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_pages = 0
    over_target = 0
    over_hard_limit = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for f in sorted(clean_dir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            units = build_atomic_units(rec["clean_text"], tokenizer, content_budget)
            pieces = pack_chunks(units, tokenizer, content_budget, args.overlap)
            total_pages += 1

            for i, piece in enumerate(pieces):
                # Ground-truth verification: the EXACT call SentenceTransformer
                # makes at inference time.
                full_ids = tokenizer.encode(piece, add_special_tokens=True)
                final_token_count = len(full_ids)

                if final_token_count > args.size:
                    over_target += 1
                if final_token_count > max_seq_length:
                    over_hard_limit += 1

                chunk_rec = {
                    "chunk_id": f"{f.stem}_{i:03d}",
                    "source_url": rec["url"],
                    "crawl_date": rec["fetched_at_utc"],
                    "chunk_index": i,
                    "chunk_size_setting": args.size,
                    "chunk_overlap_setting": args.overlap,
                    "tokenizer": f"sentence-transformers/{args.model} (BertTokenizer/WordPiece)",
                    "token_count": final_token_count,  # WITH special tokens -- matches inference-time count
                    "text": piece,
                }
                out_f.write(json.dumps(chunk_rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Done. {total_pages} pages -> {total_chunks} chunks "
          f"(size={args.size}, overlap={args.overlap}) -> {out_path}\n")
    print(f"Verification (ground truth: tokenizer.encode(text, add_special_tokens=True)):")
    print(f"  chunks > {args.size} (target size):           {over_target}")
    print(f"  chunks > {max_seq_length} (hard model limit): {over_hard_limit}")
    if over_target == 0 and over_hard_limit == 0:
        print(f"  PASS -- every chunk is within budget by construction.")
    else:
        print(f"  FAIL -- packing logic still has a gap. Do not proceed to embedding.")


if __name__ == "__main__":
    main()
