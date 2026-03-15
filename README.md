
# AI-Backed Python Coding (Hugging Face + ChromaDB)

This repo contains a few small RAG-style demos built in Python:

- **ENBD PDF Financial Analyzer (Flask web app)**: Upload a PDF, extract key financial metrics/ratios, and ask questions grounded in the PDF using **Hugging Face** LLMs + **ChromaDB** retrieval.
- **HDFC PDF Financial Analyzer (Flask web app)**: Same idea as ENBD, adapted for HDFC-style financial statements. Includes optional OCR (Tesseract) when PDFs are scanned.
- **E-commerce RAG Assistant (CLI app)**: A simple shopping/returns/support assistant that uses **ChromaDB + embeddings** (and optional reranking) plus **Hugging Face** generation.
- **Mini Agentic Flask Demo (web app)**: A tiny, deterministic example that shows an agent loop (plan → tool calls → verification) with an execution trace UI.

## What’s inside

- [enbd_extraction.py](enbd_extraction.py): Flask app for PDF upload + extraction + chat (RAG).
- [hdfc_extraction.py](hdfc_extraction.py): Flask app for HDFC-style PDF upload + extraction + chat (RAG).
- [ecommerce_hf_assistant.py](ecommerce_hf_assistant.py): CLI assistant demonstrating vector DB retrieval + optional reranking.
- [agentic_flask_ui_example.py](agentic_flask_ui_example.py): Minimal Flask UI that demonstrates agent-style tool execution (no external LLM).

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

### 2) HDFC PDF Financial Analyzer (web)

Run:

```powershell
python .\hdfc_extraction.py
```

Open:

- http://127.0.0.1:5077

### 3) E-commerce assistant (CLI)

Run:

```powershell
python .\ecommerce_hf_assistant.py
```

### 4) Mini agentic Flask demo (web)

Run:

```powershell
python .\agentic_flask_ui_example.py
```

Open:

- http://127.0.0.1:5091

## Configuration (env vars)

Common:

- `HF_MODEL_ID` (default: `google/flan-t5-base`)
- `HF_MAX_NEW_TOKENS` (default: `300`)

Chroma / embeddings:

- `USE_VECTOR_DB` (default: `1`)
- `CHROMA_DIR` (defaults: ENBD `./.chroma_enbd`, HDFC `./.chroma_hdfc`, e-commerce `./.chroma_ecommerce`)
- `CHROMA_COLLECTION_PREFIX` (HDFC default: `hdfc_pdf`)
- `EMBEDDING_MODEL_ID` (default: `sentence-transformers/all-MiniLM-L6-v2`)

Reranking (optional):

- `USE_RERANKER` (default: `0` in ENBD; e-commerce may enable it)
- `RERANK_MODEL_ID` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RAG_CANDIDATES` (default: `10`)
- `RAG_TOP_K` (e-commerce default: `4`)

## Notes

- If you have multiple Python installs (e.g., Anaconda + system Python), prefer running via the workspace venv interpreter to avoid `transformers`/`sentence-transformers` version conflicts.
- HDFC analyzer OCR: for scanned PDFs, install Tesseract and either add it to `PATH` or set `TESSERACT_CMD` (e.g. `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`). If the PDF has selectable text, OCR is typically not needed.

