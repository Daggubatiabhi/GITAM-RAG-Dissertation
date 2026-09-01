# GITAM RAG Dissertation

## Project

Design and Evaluation of a Lightweight Hallucination-Aware Retrieval-Augmented Generation Framework.

This repository contains the processing, retrieval, calibration, generation, scoring, and evaluation artefacts used for the final dissertation experiments over a frozen GITAM University website corpus.

## Experimental Modes

- Mode A: No-RAG baseline using Mistral directly
- Mode B: Standard RAG using MiniLM embeddings, FAISS retrieval, and Mistral
- Mode C: Thresholded RAG with a cosine-similarity evidence gate before generation

## Retrieval Configuration

- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Index: FAISS IndexFlatIP
- Similarity: cosine similarity using L2-normalised vectors
- Final chunking configuration: 128 tokens with 25-token overlap
- Comparison configuration: 240 tokens with 40-token overlap
- Retrieval depth: top-k = 5
- Primary threshold: tau = 0.70
- Sensitivity threshold: tau = 0.65

## Repository Structure

scripts/
- corpus cleaning
- tokenizer-aligned chunking
- embedding and FAISS indexing
- retrieval evaluation
- threshold calibration
- generation modes
- manual scoring utilities
- aggregate analysis

evaluation/
- frozen evaluation dataset
- dataset SHA-256 hash
- calibration/final-test split
- generation scoring rubric
- manual scores

results/
- retrieval metrics
- threshold calibration results
- held-out threshold results
- generation logs
- aggregate results
- latency and qualitative-analysis tables

docs/
- ambiguity and manual-review notes

## Evaluation

The study evaluates:

- retrieval effectiveness
- factual correctness
- unsupported factual claims
- groundedness
- citation correctness
- refusal behaviour
- answer coverage
- latency

The evaluation dataset contains 45 questions divided into a calibration split and a held-out final-test split.

## Reproducibility

The frozen evaluation dataset is accompanied by a SHA-256 hash so that the exact evaluation version used in the experiments can be verified.

Large local model files, virtual environments, downloaded installers, generated FAISS indexes, and temporary cache files are intentionally excluded from version control.
