"""
Generation stage -- three system modes, built on the FROZEN 128/25
retriever and the pre-declared primary threshold (tau=0.70).

v3: the --final-test-run guard has been removed following explicit
approval of the smoke test and the frozen scoring rubric (see
generation_rubric.md). This version targets the 30-QUESTION FINAL-TEST
SPLIT ONLY when --final-test-run is passed -- calibration questions are
excluded, since they were already used for threshold selection and
reusing them here would leak into the generation evaluation.

v2 changes (retained): untimed warm-up call, keep_alive on every Ollama
call, seeded deterministic per-question A/B/C execution order
randomization, conflict-case exclusion flagging (see EXCLUSION_FLAGS).

v3 additions: INCREMENTAL, RESUMABLE logging. Each (question, mode) row is
written and flushed to disk immediately, not held in memory until the end
-- a CPU-bound 30-question x 3-mode run can plausibly take over an hour,
and losing that to an interruption (closed terminal, Ollama hiccup, sleep)
would be a real cost. Re-running the same command after an interruption
detects which questions are already fully logged (all 3 modes present)
and skips them, continuing from where it left off. The seeded RNG state is
advanced to match a from-scratch run's position, so the deterministic
order property still holds after a resume.

Usage:
    python run_generation_modes.py                    # smoke test (5 fixed questions)
    python run_generation_modes.py --final-test-run     # the 30-question primary run
"""

import argparse
import csv
import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

# --- Fixed inference parameters (identical across all modes/questions/runs) ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_TAG = "mistral"  # resolved to mistral:latest, id 6577803aa9a0 at time of smoke test --
                        # record this exact ID in methodology notes; the floating tag itself
                        # is not a strict pin.
TEMPERATURE = 0.0
MAX_TOKENS = 300
KEEP_ALIVE = "30m"  # keep the model resident in memory for the duration of a run
PRIMARY_THRESHOLD = 0.70
CONTEXT_K = 5

# Fixed seed for the per-question A/B/C execution order randomization.
# Re-running with this same seed reproduces the identical sequence of
# mode orders -- this is deterministic, not re-randomized per run.
ORDER_SEED = 20260818

GROUNDED_SYSTEM_PROMPT = """You are a factual assistant answering questions about GITAM University using ONLY the context provided below. Follow these rules strictly:
1. Answer using ONLY the information in the provided context. Do not use any outside knowledge, even if you believe you know the answer.
2. If the context does not contain enough information to answer the question, respond with EXACTLY this text and nothing else: INSUFFICIENT_EVIDENCE
3. If you do answer, cite the source URL(s) you used at the end of your answer, in the format: Source(s): <url>

Context:
{context}

Question: {question}

Remember: if the context above does not contain enough evidence to answer, respond with exactly INSUFFICIENT_EVIDENCE and nothing else.

Answer:"""

SMOKE_TEST_QUESTION_IDS = ["ADM-01", "CPX-02", "LDR-03", "OOD-01", "OOD-09"]

# Known post-freeze conflict cases -- run normally, but excluded from
# primary generation-correctness metrics and reported separately.
# See ambiguity_log.md entry #12 for LDR-03.
EXCLUSION_FLAGS = {
    "LDR-03": "conflict_case",
}


def load_metadata(index_dir: Path) -> list[dict]:
    records = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def call_ollama(prompt: str) -> tuple[str, float]:
    """Returns (generated_text, latency_ms). Raises on connection failure."""
    payload = {
        "model": MODEL_TAG,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL} -- is it running? "
            f"Original error: {e}"
        )
    latency_ms = (time.time() - t0) * 1000
    return result.get("response", "").strip(), latency_ms


def warm_up_ollama():
    """One untimed call to load the model into memory before any timed run."""
    print("Warming up Mistral (untimed, not logged)...")
    _, latency = call_ollama("Hello.")
    print(f"  Warm-up complete ({latency:.0f}ms, discarded).\n")


def is_refusal(text: str) -> bool:
    return "INSUFFICIENT_EVIDENCE" in text.upper()


def retrieve(model, index, metadata, question: str, k: int) -> tuple[list[dict], float]:
    t0 = time.time()
    query_vec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    latency_ms = (time.time() - t0) * 1000
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        rec = metadata[idx]
        results.append({
            "chunk_id": rec["chunk_id"], "source_url": rec["source_url"],
            "score": float(score), "text": rec["text"],
        })
    return results, latency_ms


def build_context_block(retrieved: list[dict]) -> str:
    return "\n\n".join(f"[Source: {r['source_url']}]\n{r['text']}" for r in retrieved)


def run_mode_a(question: str) -> dict:
    answer, gen_latency = call_ollama(question)
    return {
        "mode": "A_no_rag", "top_k_chunk_ids": "", "source_urls": "", "retrieval_scores": "",
        "threshold_decision": "n/a", "prompt_sent": question, "generated_answer": answer,
        "refused": False, "generation_latency_ms": round(gen_latency, 1), "retrieval_latency_ms": 0.0,
    }


def run_mode_b(question: str, retrieved: list[dict], retrieval_latency: float) -> dict:
    context = build_context_block(retrieved)
    prompt = GROUNDED_SYSTEM_PROMPT.format(context=context, question=question)
    answer, gen_latency = call_ollama(prompt)
    return {
        "mode": "B_standard_rag",
        "top_k_chunk_ids": " ; ".join(r["chunk_id"] for r in retrieved),
        "source_urls": " ; ".join(r["source_url"] for r in retrieved),
        "retrieval_scores": " ; ".join(f"{r['score']:.3f}" for r in retrieved),
        "threshold_decision": "n/a (no gate in Mode B)", "prompt_sent": prompt,
        "generated_answer": answer, "refused": is_refusal(answer),
        "generation_latency_ms": round(gen_latency, 1), "retrieval_latency_ms": round(retrieval_latency, 1),
    }


def run_mode_c(question: str, retrieved: list[dict], retrieval_latency: float) -> dict:
    top1_score = retrieved[0]["score"] if retrieved else 0.0
    gate_passed = top1_score >= PRIMARY_THRESHOLD

    if not gate_passed:
        return {
            "mode": "C_thresholded_rag",
            "top_k_chunk_ids": " ; ".join(r["chunk_id"] for r in retrieved),
            "source_urls": " ; ".join(r["source_url"] for r in retrieved),
            "retrieval_scores": " ; ".join(f"{r['score']:.3f}" for r in retrieved),
            "threshold_decision": f"GATE FAILED (top1={top1_score:.3f} < {PRIMARY_THRESHOLD}) -- Mistral NOT called",
            "prompt_sent": "(gate failed -- no prompt sent to Mistral)",
            "generated_answer": "INSUFFICIENT_EVIDENCE", "refused": True,
            "generation_latency_ms": 0.0, "retrieval_latency_ms": round(retrieval_latency, 1),
        }

    context = build_context_block(retrieved)
    prompt = GROUNDED_SYSTEM_PROMPT.format(context=context, question=question)
    answer, gen_latency = call_ollama(prompt)
    return {
        "mode": "C_thresholded_rag",
        "top_k_chunk_ids": " ; ".join(r["chunk_id"] for r in retrieved),
        "source_urls": " ; ".join(r["source_url"] for r in retrieved),
        "retrieval_scores": " ; ".join(f"{r['score']:.3f}" for r in retrieved),
        "threshold_decision": f"GATE PASSED (top1={top1_score:.3f} >= {PRIMARY_THRESHOLD})",
        "prompt_sent": prompt, "generated_answer": answer, "refused": is_refusal(answer),
        "generation_latency_ms": round(gen_latency, 1), "retrieval_latency_ms": round(retrieval_latency, 1),
    }


MODE_RUNNERS = {"A": run_mode_a, "B": run_mode_b, "C": run_mode_c}
MODE_LABELS = {"A": "Mode A (No-RAG)", "B": "Mode B (Standard RAG)", "C": "Mode C (Thresholded RAG)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="evaluation_questions_split.csv")
    ap.add_argument("--index", default="data/index/idx_128_25_minilm")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--final-test-run", action="store_true",
                     help="Run the primary 30-question final-test evaluation.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or ("generation_final_test_log.csv" if args.final_test_run
                             else "generation_smoke_test_log.csv")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is not installed. Run: pip install sentence-transformers faiss-cpu")
    try:
        import faiss
    except ImportError:
        raise SystemExit("faiss-cpu is not installed. Run: pip install sentence-transformers faiss-cpu")

    with open(args.questions, encoding="utf-8") as f:
        all_questions_rows = list(csv.DictReader(f))
    all_questions = {r["question_id"]: r for r in all_questions_rows}

    if args.final_test_run:
        target_ids = sorted(r["question_id"] for r in all_questions_rows if r["split"] == "final_test")
        print(f"FINAL-TEST RUN: {len(target_ids)} questions (split=='final_test' only -- "
              f"calibration questions are NOT included, since they were already used for "
              f"threshold selection).")
    else:
        target_ids = SMOKE_TEST_QUESTION_IDS

    missing = [qid for qid in target_ids if qid not in all_questions]
    if missing:
        raise SystemExit(f"Question ID(s) not found in {args.questions}: {missing}")

    # --- Resume support: skip questions already fully logged (all 3 modes present) ---
    fieldnames = ["question_id", "question", "category", "split", "question_answerable",
                  "negative_type", "exclusion_flag", "mode", "top_k_chunk_ids", "source_urls",
                  "retrieval_scores", "threshold_decision", "prompt_sent", "generated_answer",
                  "refused", "generation_latency_ms", "retrieval_latency_ms", "total_latency_ms",
                  "execution_order_position"]

    # --- Resume support: skip questions already fully logged (all 3 modes present).
    # A question with SOME rows but <3 means a mid-question crash -- those stale
    # partial rows must be REMOVED (not left alongside fresh ones, which would
    # duplicate that question), so the file is rewritten keeping only genuinely
    # complete questions before appending resumes. Assumes normal sequential
    # interruption (a clean prefix of sorted target_ids done) -- true for any
    # real crash/close/interrupt of this script, not guaranteed under manual
    # file editing.
    already_done = set()
    out_file = Path(out_path)
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        counts = {}
        for r in existing:
            counts[r["question_id"]] = counts.get(r["question_id"], 0) + 1
        already_done = {qid for qid, n in counts.items() if n >= 3}
        partial = {qid for qid, n in counts.items() if 0 < n < 3}

        if partial:
            print(f"Found {len(partial)} partially-logged question(s) from an interrupted "
                  f"run (mid-question crash): {sorted(partial)} -- discarding their stale "
                  f"rows and will redo these cleanly.")
            clean_rows = [r for r in existing if r["question_id"] in already_done]
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(clean_rows)

        if already_done:
            print(f"Resuming: {len(already_done)} question(s) already complete in {out_path}, will skip.")
        csv_file = open(out_file, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    else:
        csv_file = open(out_file, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

    remaining_ids = [qid for qid in target_ids if qid not in already_done]

    print(f"Model tag in use: {MODEL_TAG}  (record the resolved ID from `ollama list` in your notes)")
    print(f"Temperature: {TEMPERATURE}  |  Max tokens: {MAX_TOKENS}  |  Threshold (Mode C): {PRIMARY_THRESHOLD}")
    print(f"keep_alive: {KEEP_ALIVE}  |  Execution-order seed: {ORDER_SEED}")
    print(f"Questions remaining: {len(remaining_ids)} of {len(target_ids)}\n")

    if not remaining_ids:
        print("All questions already complete. Nothing to do.")
        csv_file.close()
        return

    index_dir = Path(args.index)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    metadata = load_metadata(index_dir)
    model = SentenceTransformer(args.model)

    warm_up_ollama()

    rng = random.Random(ORDER_SEED)
    # Advance the RNG state past already-completed questions so the seeded
    # sequence for remaining questions matches what a from-scratch run would
    # have produced at this position (keeps the deterministic order property
    # meaningful even after a resume).
    for _ in range(len(already_done)):
        order = ["A", "B", "C"]
        rng.shuffle(order)

    run_start = time.time()
    display_rows = []

    for qi, qid in enumerate(remaining_ids, start=1):
        row = all_questions[qid]
        question = row["question"]
        exclusion_flag = EXCLUSION_FLAGS.get(qid, "")

        elapsed_min = (time.time() - run_start) / 60
        print(f"\n{'='*70}\n[{qi}/{len(remaining_ids)}]  {qid}: {question}"
              f"  (elapsed: {elapsed_min:.1f} min)\n{'='*70}")
        if exclusion_flag:
            print(f"  [NOTE] exclusion_flag='{exclusion_flag}' -- runs normally, "
                  f"excluded from primary correctness metrics downstream.")

        retrieved, retrieval_latency = retrieve(model, index, metadata, question, CONTEXT_K)

        mode_order = ["A", "B", "C"]
        rng.shuffle(mode_order)
        print(f"  Execution order: {mode_order}")

        results_by_mode = {}
        for position, mode_key in enumerate(mode_order, start=1):
            print(f"  [{position}/3] {MODE_LABELS[mode_key]}...")
            t0 = time.time()
            if mode_key == "A":
                res = run_mode_a(question)
            elif mode_key == "B":
                res = run_mode_b(question, retrieved, retrieval_latency)
            else:
                res = run_mode_c(question, retrieved, retrieval_latency)
            res["total_latency_ms"] = round((time.time() - t0) * 1000, 1)
            res["execution_order_position"] = position
            results_by_mode[mode_key] = res

            log_row = {
                "question_id": qid, "question": question, "category": row["category"],
                "split": row["split"], "question_answerable": row["answerable"],
                "negative_type": row.get("negative_type", ""),
                "exclusion_flag": exclusion_flag,
                **res,
            }
            writer.writerow(log_row)
            csv_file.flush()  # write to disk immediately -- survives interruption

        display_rows.append((qid, question, exclusion_flag, results_by_mode))

    csv_file.close()
    total_min = (time.time() - run_start) / 60
    print(f"\n\nRun complete in {total_min:.1f} minutes. Log saved -> {out_path}")

    print(f"\n{'='*100}\nSIDE-BY-SIDE RESULTS (this session)\n{'='*100}")
    for qid, question, exclusion_flag, results in display_rows:
        tag = f"  [exclusion_flag={exclusion_flag}]" if exclusion_flag else ""
        print(f"\n{'-'*100}\n{qid}: {question}{tag}\n{'-'*100}")
        for mode_key in ["A", "B", "C"]:
            r = results[mode_key]
            print(f"\n  {MODE_LABELS[mode_key]}  (executed {r['execution_order_position']}/3)")
            print(f"    Threshold decision: {r['threshold_decision']}")
            print(f"    Refused: {r['refused']}")
            print(f"    Answer: {r['generated_answer'][:300]}")
            print(f"    Latency: gen={r['generation_latency_ms']}ms  "
                  f"retrieval={r['retrieval_latency_ms']}ms  total={r['total_latency_ms']}ms")

    print(f"\n\nDone. No threshold/retrieval/prompt changes were made based on these outputs.")
    print(f"Run prepare_scoring_template.py next to generate the blank rubric-scoring file.")


if __name__ == "__main__":
    main()
