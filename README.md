
# AI-Backed Python Coding (Hugging Face + ChromaDB)

This repo contains two small RAG-style demos built in Python:

- **ENBD PDF Financial Analyzer (Flask web app)**: Upload a PDF, extract key financial metrics/ratios, and ask questions grounded in the PDF using **Hugging Face** LLMs + **ChromaDB** retrieval.
- **E-commerce RAG Assistant (CLI app)**: A simple shopping/returns/support assistant that uses **ChromaDB + embeddings** (and optional reranking) plus **Hugging Face** generation.

## What’s inside

- [enbd_extraction.py](enbd_extraction.py): Flask app for PDF upload + extraction + chat (RAG).
- [ecommerce_hf_assistant.py](ecommerce_hf_assistant.py): CLI assistant demonstrating vector DB retrieval + optional reranking.

## Tech stack

- **Generation**: Hugging Face `transformers` pipeline (default model `google/flan-t5-base`)
- **Vector DB**: ChromaDB (local persistent collections)
- **Embeddings**: SentenceTransformers (default `sentence-transformers/all-MiniLM-L6-v2`)
- **Optional reranking**: CrossEncoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`)

## Quickstart

Install deps:

```powershell
pip install -r requirements.txt
```

### 1) ENBD PDF Financial Analyzer (web)

Run:

```powershell
python .\enbd_extraction.py
```

Open:

- http://127.0.0.1:5089

### 2) E-commerce assistant (CLI)

Run:

```powershell
python .\ecommerce_hf_assistant.py
```

## Configuration (env vars)

Common:

- `HF_MODEL_ID` (default: `google/flan-t5-base`)
- `HF_MAX_NEW_TOKENS` (default: `300`)

Chroma / embeddings:

- `USE_VECTOR_DB` (default: `1`)
- `CHROMA_DIR` (ENBD default: `./.chroma_enbd`, e-commerce default: `./.chroma_ecommerce`)
- `EMBEDDING_MODEL_ID` (default: `sentence-transformers/all-MiniLM-L6-v2`)

Reranking (optional):

- `USE_RERANKER` (default: `0` in ENBD; e-commerce may enable it)
- `RERANK_MODEL_ID` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RAG_CANDIDATES` (default: `10`)
- `RAG_TOP_K` (e-commerce default: `4`)

## Notes

- If you have multiple Python installs (e.g., Anaconda + system Python), prefer running via the workspace venv interpreter to avoid `transformers`/`sentence-transformers` version conflicts.

