"""
Stage 5: Embedding + FAISS indexing
-------------------------------------
Turns a chunked .jsonl file into a searchable FAISS vector index.

Model: all-MiniLM-L6-v2 (sentence-transformers) -- the embedding model
justified in the dissertation's technology table: small, fast on local/
limited hardware, strong general-purpose semantic quality for its size.

Similarity: cosine similarity, implemented as inner product on L2-normalised
vectors (faiss.IndexFlatIP + normalised vectors is mathematically equivalent
to cosine similarity search). Normalization is verified empirically at
build time, not just assumed from the encode() flag.

Run once per chunk configuration -- each is a separate index:
    python embed_index.py --chunks data/chunks/chunks_128_25_minilm.jsonl --out data/index/idx_128_25_minilm
    python embed_index.py --chunks data/chunks/chunks_240_40_minilm.jsonl --out data/index/idx_240_40_minilm

Produces, in the --out directory:
    index.faiss     -- the FAISS vector index (vectors only, no metadata)
    metadata.jsonl  -- one line per vector, SAME ORDER as the index, so
                       FAISS result position i maps to metadata line i.
                       Holds chunk_id, source_url, page_title (extracted
                       from the raw HTML's <title> tag), crawl_date,
                       chunk_index, token_count, and the chunk text itself.
    manifest.json   -- full provenance for this index: embedding model,
                       dimension, vector count, source chunk file, chunk
                       config, normalization, FAISS index type, build time.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_url_title_map(raw_dir: Path) -> dict:
    """Extracts the real <title> tag from each page's raw HTML (already on
    disk from the crawl stage -- never re-fetches anything). This was never
    parsed out as a separate provenance field earlier in the pipeline, so
    it's derived here from the source of truth rather than approximated
    from the URL slug. Pages with no <title> tag simply have no entry."""
    if not raw_dir.exists():
        print(f"  [warn] raw dir not found ({raw_dir}) -- page_title will be omitted")
        return {}

    from bs4 import BeautifulSoup

    url_to_title = {}
    for f in raw_dir.glob("*.json"):
        if f.name == "_manifest.json":
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        html = rec.get("html")
        url = rec.get("url")
        if not html or not url:
            continue
        try:
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                url_to_title[url] = soup.title.string.strip()
        except Exception:
            continue
    return url_to_title


def main():
    ap = argparse.ArgumentParser(description="Embed chunks and build a FAISS index.")
    ap.add_argument("--chunks", required=True, help="Path to a chunks .jsonl file")
    ap.add_argument("--out", required=True, help="Output directory for index.faiss + metadata.jsonl")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--raw-dir", default="data/raw",
                     help="Directory of raw crawled pages, used to look up page titles for the metadata mapping")
    args = ap.parse_args()

    # Imports here (not top of file) so --help works even before these are
    # installed, and so the error message is clear if they're missing.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers is not installed. Run:\n"
            "    pip install sentence-transformers faiss-cpu"
        )
    try:
        import faiss
    except ImportError:
        raise SystemExit(
            "faiss-cpu is not installed. Run:\n"
            "    pip install sentence-transformers faiss-cpu"
        )

    chunks_path = Path(args.chunks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {chunks_path} ...")
    chunks = load_chunks(chunks_path)
    print(f"  {len(chunks)} chunks loaded")

    print(f"Loading embedding model: {args.model} ...")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dimension: {dim}")

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks (batch size {args.batch_size}) ...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise here -> inner product == cosine similarity
    ).astype("float32")
    print(f"  done in {time.time() - t0:.1f}s")

    # Verify normalization empirically rather than trusting the flag blindly --
    # cosine-via-inner-product is only mathematically correct if vectors are
    # genuinely unit-norm.
    sample_norms = np.linalg.norm(embeddings[: min(50, len(embeddings))], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-3):
        raise SystemExit(
            f"Embeddings are NOT unit-normalised (sample norms: min={sample_norms.min():.4f}, "
            f"max={sample_norms.max():.4f}). IndexFlatIP would not be equivalent to cosine "
            f"similarity in this state -- aborting rather than building an inconsistent index."
        )
    print(f"  Verified: embeddings are L2-normalised (sample norms in "
          f"[{sample_norms.min():.4f}, {sample_norms.max():.4f}], expected ~1.0)")
    print(f"  -> IndexFlatIP (inner product) on these vectors is mathematically "
          f"equivalent to cosine similarity search.")

    print("Building FAISS index (IndexFlatIP -- exact cosine similarity search) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  index size: {index.ntotal} vectors")

    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    print(f"Looking up page titles from raw HTML in {args.raw_dir} ...")
    url_to_title = build_url_title_map(Path(args.raw_dir))
    print(f"  {len(url_to_title)} page titles found")

    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            # Keep only what's useful at retrieval/citation time -- same
            # order as the index, so FAISS position i == metadata line i.
            f.write(json.dumps({
                "chunk_id": c["chunk_id"],
                "source_url": c["source_url"],
                "page_title": url_to_title.get(c["source_url"]),
                "crawl_date": c["crawl_date"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "text": c["text"],
            }, ensure_ascii=False) + "\n")

    # Manifest: full provenance for this index, so later analysis/write-up
    # doesn't have to reconstruct how it was built.
    manifest = {
        "embedding_model": f"sentence-transformers/{args.model}",
        "embedding_dimension": dim,
        "n_vectors": index.ntotal,
        "source_chunk_file": str(chunks_path),
        "chunk_size_setting": chunks[0].get("chunk_size_setting") if chunks else None,
        "chunk_overlap_setting": chunks[0].get("chunk_overlap_setting") if chunks else None,
        "chunk_tokenizer": chunks[0].get("tokenizer") if chunks else None,
        "normalization": "L2 unit-norm (verified empirically at build time)",
        "faiss_index_type": "IndexFlatIP (exact inner-product search; mathematically "
                             "equivalent to cosine similarity given L2-normalised vectors)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Index:    {index_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nSearch it with:")
    print(f"  python search_index.py --index {out_dir} --query \"your question here\"")


if __name__ == "__main__":
    main()
