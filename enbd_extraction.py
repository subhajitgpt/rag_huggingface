from flask import Flask, request, render_template_string, session, redirect, url_for, jsonify
import fitz, tempfile, re, os, sys, time
import difflib
from dotenv import load_dotenv

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reduce noisy/broken TF/Flax imports in mixed environments
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from transformers import pipeline

# --- API + Flask setup ---
load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "300"))

# --- Optional ChromaDB + Embeddings retrieval ---
USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "1").strip().lower() in {"1", "true", "yes"}
CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_enbd")
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "enbd_pdf")

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "0").strip().lower() in {"1", "true", "yes"}
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RAG_CANDIDATES = int(os.getenv("RAG_CANDIDATES", "10"))

# ===================== In-memory doc store (avoid cookie bloat) =====================
_DOC_TTL_SECONDS = 60 * 60  # 1 hour
_DOC_STORE_LOCK = threading.Lock()


@dataclass
class StoredDoc:
  created_at: float
  chunks: List[str]
  vectorizer: Any
  tfidf_matrix: Any
  chroma_collection: Optional[str] = None


_DOC_STORE: Dict[str, StoredDoc] = {}


_CHROMA_LOCK = threading.Lock()
_CHROMA_CLIENT: Any = None

_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER: Any = None

_RERANKER_LOCK = threading.Lock()
_RERANKER: Any = None


def _warn_if_not_using_venv() -> None:
  exe = (sys.executable or "").replace("\\", "/").lower()
  if "/.venv/" not in exe and "\\.venv\\" not in (sys.executable or ""):
    print(
      "[warn] Not running workspace venv; sentence-transformers/transformers may be mismatched. "
      "Prefer: C:/ai_backed_python_coding/.venv/Scripts/python.exe c:/ai_backed_python_coding/enbd_extraction.py"
    )


def _get_chroma_client():
  global _CHROMA_CLIENT
  with _CHROMA_LOCK:
    if _CHROMA_CLIENT is not None:
      return _CHROMA_CLIENT
    try:
      import chromadb  # type: ignore

      _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
      return _CHROMA_CLIENT
    except Exception as e:
      raise RuntimeError(f"ChromaDB unavailable: {e}")


def _safe_delete_chroma_collection(name: Optional[str]) -> None:
  if not name:
    return
  try:
    client = _get_chroma_client()
    client.delete_collection(name=name)
  except Exception:
    # Best-effort cleanup only
    return


def _get_embedder():
  global _EMBEDDER
  with _EMBEDDER_LOCK:
    if _EMBEDDER is not None:
      return _EMBEDDER
    try:
      from sentence_transformers import SentenceTransformer as _ST  # type: ignore

      _EMBEDDER = _ST(EMBEDDING_MODEL_ID)
      return _EMBEDDER
    except Exception as e:
      raise RuntimeError(
        "Failed to import sentence-transformers. This usually happens when you're not running the project venv "
        "or when transformers/sentence-transformers are mismatched. "
        "Run with: C:/ai_backed_python_coding/.venv/Scripts/python.exe c:/ai_backed_python_coding/enbd_extraction.py"
      ) from e


def _get_reranker():
  global _RERANKER
  if not USE_RERANKER:
    return None
  with _RERANKER_LOCK:
    if _RERANKER is not None:
      return _RERANKER
    try:
      from sentence_transformers import CrossEncoder as _CE  # type: ignore

      _RERANKER = _CE(RERANK_MODEL_ID)
      return _RERANKER
    except Exception as e:
      raise RuntimeError(f"Failed to initialize reranker: {e}")


def _doc_store_cleanup_locked() -> None:
  now = time.time()
  expired = [k for k, v in _DOC_STORE.items() if now - v.created_at > _DOC_TTL_SECONDS]
  for k in expired:
    doc = _DOC_STORE.pop(k, None)
    if doc and doc.chroma_collection:
      _safe_delete_chroma_collection(doc.chroma_collection)


def _doc_store_put(doc: StoredDoc) -> str:
  doc_id = os.urandom(12).hex()
  with _DOC_STORE_LOCK:
    _DOC_STORE[doc_id] = doc
    _doc_store_cleanup_locked()
  return doc_id


def _doc_store_get(doc_id: Optional[str]) -> Optional[StoredDoc]:
  if not doc_id:
    return None
  with _DOC_STORE_LOCK:
    _doc_store_cleanup_locked()
    return _DOC_STORE.get(doc_id)


def _doc_store_delete(doc_id: Optional[str]) -> None:
  if not doc_id:
    return
  with _DOC_STORE_LOCK:
    doc = _DOC_STORE.pop(doc_id, None)
  if doc and doc.chroma_collection:
    _safe_delete_chroma_collection(doc.chroma_collection)


def _index_chunks_in_chroma(doc_id: str, chunks: List[str]) -> Optional[str]:
  if not USE_VECTOR_DB or not chunks:
    return None

  try:
    embedder = _get_embedder()
    client = _get_chroma_client()

    # One collection per uploaded PDF (tied to our internal doc_id).
    collection_name = f"{CHROMA_COLLECTION_PREFIX}_{doc_id}"
    try:
      client.delete_collection(name=collection_name)
    except Exception:
      pass
    collection = client.get_or_create_collection(name=collection_name)

    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return collection_name
  except Exception as e:
    print(f"[warn] Vector DB disabled for this document: {e}")
    return None


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 160) -> List[str]:
  text = (text or "").strip()
  if not text:
    return []
  text = re.sub(r"\s+", " ", text)

  chunks: List[str] = []
  start = 0
  n = len(text)
  while start < n:
    end = min(n, start + chunk_size)
    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)
    if end >= n:
      break
    start = max(0, end - overlap)
  return chunks


def _build_retriever(chunks: List[str]) -> Tuple[Optional[TfidfVectorizer], Any]:
  if not chunks:
    return None, None

  # Word retrieval for semantics + char retrieval for typos/OCR noise.
  word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
  word_matrix = word_vectorizer.fit_transform(chunks)

  char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
  char_matrix = char_vectorizer.fit_transform(chunks)

  bundle = {"word": word_vectorizer, "char": char_vectorizer}
  matrices = {"word": word_matrix, "char": char_matrix}
  return bundle, matrices


def _retrieve_top_chunks(doc: StoredDoc, question: str, top_k: int = 4) -> List[str]:
  q = (question or "").strip()
  if not q or not doc or not doc.chunks or doc.vectorizer is None or doc.tfidf_matrix is None:
    # If TF-IDF isn't present (shouldn't happen) still allow Chroma if available.
    if not q or not doc or not doc.chunks:
      return []

  # Prefer Chroma retrieval when present.
  if USE_VECTOR_DB and doc.chroma_collection:
    try:
      embedder = _get_embedder()
      client = _get_chroma_client()
      collection = client.get_collection(name=doc.chroma_collection)
      q_emb = embedder.encode([q], normalize_embeddings=True).tolist()
      n_results = max(1, min(len(doc.chunks), max(top_k, RAG_CANDIDATES)))
      res = collection.query(query_embeddings=q_emb, n_results=n_results, include=["documents", "distances", "metadatas"])
      docs = (res.get("documents") or [[]])[0]
      dists = (res.get("distances") or [[]])[0]

      retrieved = [(str(d), float(dist) if dist is not None else 1e9) for d, dist in zip(docs, dists)]

      # Optional reranking over candidates.
      if USE_RERANKER and retrieved:
        try:
          reranker = _get_reranker()
          if reranker is not None:
            pairs = [(q, d) for (d, _dist) in retrieved]
            scores = reranker.predict(pairs)
            scored = list(zip([d for (d, _dist) in retrieved], [float(s) for s in scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for (d, _s) in scored[: max(1, top_k)]]
        except Exception as e:
          print(f"[warn] Reranker disabled due to error: {e}")

      # Otherwise return in vector order (dist ascending).
      return [d for (d, _dist) in retrieved[: max(1, top_k)] if d]
    except Exception as e:
      print(f"[warn] Chroma retrieval failed; falling back to TF-IDF: {e}")

  try:
    vectorizer = doc.vectorizer
    matrices = doc.tfidf_matrix

    if isinstance(vectorizer, dict) and isinstance(matrices, dict) and vectorizer.get("word") is not None and vectorizer.get("char") is not None:
      qv_word = vectorizer["word"].transform([q])
      qv_char = vectorizer["char"].transform([q])
      sims_word = cosine_similarity(qv_word, matrices["word"]).ravel()
      sims_char = cosine_similarity(qv_char, matrices["char"]).ravel()
      # Prefer whichever similarity is stronger for each chunk.
      sims = np.maximum(sims_word, sims_char)
    else:
      qv = vectorizer.transform([q])
      sims = cosine_similarity(qv, matrices).ravel()

    if sims.size == 0:
      return []
    top_idx = sims.argsort()[::-1][: max(1, top_k)]
    out: List[str] = []
    for i in top_idx:
      if sims[int(i)] <= 0:
        continue
      out.append(doc.chunks[int(i)])
    return out
  except Exception:
    return []


_HF_LOCK = threading.Lock()
_HF_GENERATORS: Dict[str, Any] = {}


def _get_hf_generator(model_id: str):
  with _HF_LOCK:
    if model_id in _HF_GENERATORS:
      return _HF_GENERATORS[model_id]
    gen = pipeline("text2text-generation", model=model_id)
    _HF_GENERATORS[model_id] = gen
    return gen


def _hf_answer(question: str, computed_context: str, excerpts: List[str], chat_history: List[Dict[str, str]]) -> str:
  gen = _get_hf_generator(HF_MODEL_ID)

  system = (
    "You are a financial analyst specializing in UAE banking. "
    "Be concise and numeric. Use ONLY the provided context. "
    "If a requested value is not present, say you cannot find it in the provided context."
  )

  parts: List[str] = [system]
  if computed_context:
    parts.append("COMPUTED METRICS & RATIOS:\n" + computed_context)
  if excerpts:
    parts.append("PDF EXCERPTS:\n" + "\n\n".join(f"[Excerpt {i+1}]\n{ex}" for i, ex in enumerate(excerpts)))
  if chat_history:
    recent = chat_history[-5:]
    parts.append("RECENT CHAT:\n" + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent))

  parts.append("QUESTION:\n" + (question or ""))
  prompt = "\n\n".join(parts)

  try:
    out = gen(prompt, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
    if isinstance(out, list) and out:
      text = out[0].get("generated_text") or out[0].get("text")
      if text:
        return str(text).strip()
    return "I couldn't generate an answer right now."
  except Exception as e:
    return f"AI service error: {e}"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

# ---------- Helpers ----------
def to_float(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None

def safe_div(a, b):
    return round(a / b, 4) if a is not None and b not in (None, 0) else None

def fmt_pct(x):
    return f"{x*100:.2f}%" if x is not None else "N/A"

def filter_ratios(ratio_list):
    """Remove ratios with None values (which would show as N/A)."""
    return [(name, val) for (name, val) in ratio_list if val is not None]

def filter_dual_metrics(dual_dict):
    """Keep only lines where at least one of current/prior is not None."""
    out = {}
    for k, v in (dual_dict or {}).items():
        cur = v.get("current")
        pri = v.get("prior")
        if cur is not None or pri is not None:
            out[k] = v
    return out

def filter_single_metrics(single_dict):
    """Keep only lines where value is not None."""
    return {k: v for k, v in (single_dict or {}).items() if v is not None}

def normalize_unit_label(label_raw):
    if not label_raw:
        return None
    s = str(label_raw).lower()
    if "billion" in s:  return "billions"
    if "million" in s:  return "millions"
    if "thousand" in s: return "thousands"
    return None

def detect_units(text):
    """
    Best-effort scan for currency + magnitude like 'AED millions' / 'Amounts in thousands of UAE dirhams'.
    Returns dict: {"currency": "AED", "units_label": "millions"|"thousands"|"billions"|None}
    """
    currency = None
    units_label = None

    # Common patterns across front matter / notes
    pats = [
        r"(?:all amounts|figures)[^.\n]{0,80}(?:in|expressed in)[^.\n]{0,80}(AED|UAE\s*Dirhams|UAE\s*Dhirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(AED|UAE\s*Dirhams|UAE\s*Dhirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(Amounts? in)[^.\n]{0,40}(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
        r"(AED)[^\n]{0,10}\((?:in\s+)?(thousand|thousands|million|millions|billion|billions)\)",
    ]
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            # Heuristic: last group is the units word
            groups = [g for g in m.groups() if g]
            # Find currency and unit among captured groups
            for g in groups:
                if re.search(r"AED|Dirhams", g, re.I):
                    currency = "AED"
                if re.search(r"thousand|million|billion", g, re.I):
                    units_label = normalize_unit_label(g)
            break

    return {"currency": currency or "AED", "units_label": units_label}


def _normalize_query(text: str) -> str:
  s = (text or "").lower()
  s = re.sub(r"[^a-z0-9]+", " ", s)
  s = re.sub(r"\s+", " ", s).strip()
  return s


def _best_fuzzy_choice(query: str, choices: List[str]) -> Tuple[Optional[str], float]:
  q = _normalize_query(query)
  if not q or not choices:
    return None, 0.0

  best = None
  best_score = 0.0
  q_tokens = set(q.split())

  for ch in choices:
    c = _normalize_query(ch)
    if not c:
      continue
    c_tokens = set(c.split())
    token_jaccard = (len(q_tokens & c_tokens) / len(q_tokens | c_tokens)) if (q_tokens or c_tokens) else 0.0
    seq = difflib.SequenceMatcher(a=q, b=c).ratio()
    score = max(seq, token_jaccard)
    if score > best_score:
      best_score = score
      best = ch

  return best, best_score


def _metric_synonyms() -> Dict[str, List[str]]:
  return {
    "Net Interest + Islamic Income": [
      "net interest income and net income from islamic financing and investment products",
      "net interest and islamic income",
      "islamic income",
      "income from islamic financing",
      "islamic financing income",
    ],
    "Customer and Islamic Deposits": [
      "customer deposits",
      "islamic deposits",
      "customer and islamic deposits",
    ],
  }


def _try_metric_fast_path(prompt: str, dual: Dict[str, Any], single: Dict[str, Any]) -> Optional[str]:
  if not prompt:
    return None

  key_to_phrases: Dict[str, List[str]] = {}
  for k in list((dual or {}).keys()) + list((single or {}).keys()):
    key_to_phrases.setdefault(k, []).append(k)

  # Add synonyms only for keys that actually exist in the extracted set.
  for k, syns in _metric_synonyms().items():
    if k in key_to_phrases:
      key_to_phrases[k].extend(syns)

  all_phrases: List[str] = []
  phrase_to_key: Dict[str, str] = {}
  for k, phrases in key_to_phrases.items():
    for p in phrases:
      all_phrases.append(p)
      phrase_to_key[p] = k

  best_phrase, score = _best_fuzzy_choice(prompt, all_phrases)
  if not best_phrase:
    return None

  # Handles short + misspelled queries like "islamic inome".
  if score < 0.58:
    return None

  key = phrase_to_key.get(best_phrase)
  if not key:
    return None

  if key in (dual or {}):
    v = dual[key]
    cur = v.get("current")
    pri = v.get("prior")
    if cur is None and pri is None:
      return None
    return f"{key}: current={cur}, prior={pri}"

  if key in (single or {}):
    v = single[key]
    if v is None:
      return None
    return f"{key}: {v}"

  return None


def _ui_model_badge(has_context: bool, doc: Optional[StoredDoc]) -> str:
  parts: List[str] = [f"HF: {HF_MODEL_ID}"]

  if not has_context:
    parts.append("RAG: off")
    return " • ".join(parts)

  retriever = "TF-IDF"
  if USE_VECTOR_DB and doc and doc.chroma_collection:
    retriever = "Chroma"
  parts.append(f"RAG: {retriever}")
  if USE_RERANKER:
    parts.append("Rerank: ON")
  return " • ".join(parts)

# Regexes for ENBD-style statements (tweak as needed)

# Dual-line patterns (Current vs Prior)
patterns_dual = {
    "Total Operating Income": r"Total operating income\s+([\d,]+)\s+([\d,]+)",
    "General and Administrative Expenses": r"General and administrative expenses\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Operating Profit Before Impairment": r"Operating profit before impairment\s+([\d,]+)\s+([\d,]+)",
    "Profit Before Tax": r"Profit for the period before taxation\s+([\d,]+)\s+([\d,]+)",
    "Taxation Charge": r"Taxation charge\s+\(([\d,]+)\)\s+\(([\d,]+)\)",
    "Profit for the Period": r"Profit for the period\s+([\d,]+)\s+([\d,]+)",
    "Earnings Per Share (AED)": r"Earnings per share\s*\(AED\)\s+([\d\.]+)\s+([\d\.]+)",

    # Balance sheet / interest lines (Current vs Prior when present)
    "Customer and Islamic Deposits": r"Customer(?:\s+and)?\s+Islamic deposits\s+([\d,]+)\s+([\d,]+)",
    "Gross Loans and Receivables (dual)": r"Gross loans and receivables\s+([\d,]+)\s+([\d,]+)",
    "Net Interest + Islamic Income": r"Net interest income and net income from Islamic financing and\s*investment products\s+([\d,]+)\s+([\d,]+)",
}

# Single-line patterns (Current only)
patterns_single = {
    "Gross Loans": r"Gross loans and receivables\s+([\d,]+)\s+[\d,]+",
    "ECL": r"Less:\s*Expected credit losses\s+\(([\d,]+)\)\s+\([\d,]+\)",
    "NPLs": r"Total of credit impaired loans and receivables\s+([\d,]+)\s+[\d,]+",
    "Total Assets": r"Segment Assets[\s\S]*?(\d{1,3}(?:,\d{3})+)\s*\n\s*Segment Liabilities",

    # Liquidity line items (Current only from notes)
    "Cash & CB": r"Cash and deposits with Central Banks\s+([\d,]+)\s+[\d,]+",
    "Due from Banks": r"Due from banks\s+([\d,]+)\s+[\d,]+",
    "Investment Securities": r"Net Investment securities\s+([\d,]+)",
    # Best-effort equity capture from Statement of Changes in Equity
    "Equity (Group Total)": r"Balance as at .*?Group\s+Total\s+(\d{1,3}(?:,\d{3})+)",
}

def extract_dual(text):
    out = {}
    for k, p in patterns_dual.items():
        m = re.search(p, text, re.I)
        out[k] = {
            "current": to_float(m.group(1)) if m else None,
            "prior": to_float(m.group(2)) if (m and m.lastindex and m.lastindex >= 2) else None,
        }
    return out

def extract_single(text):
    out = {}
    for k, p in patterns_single.items():
        m = re.search(p, text, re.I)
        out[k] = to_float(m.group(1)) if m else None
    return out

def parse_pdf(path):
    print(f"[parse_pdf] path={path}")
    with fitz.open(path) as doc:
        txt = "\n".join(pg.get_text() for pg in doc)
    print(f"[parse_pdf] extracted text len={len(txt)}")
    if txt:
        print("[parse_pdf] text sample:", repr(txt[:300]))
    units = detect_units(txt)
    dual_raw = extract_dual(txt)
    single_raw = extract_single(txt)
    
    # Filter for display context
    dual = filter_dual_metrics(dual_raw)
    single = filter_single_metrics(single_raw)
    
    return txt, dual_raw, single_raw, dual, single, units

def compute_ratios(dual, single):
    # Income statement lines
    toi = dual.get("Total Operating Income", {}).get("current")
    ga  = dual.get("General and Administrative Expenses", {}).get("current")
    opb = dual.get("Operating Profit Before Impairment", {}).get("current")
    pat = dual.get("Profit for the Period", {}).get("current")
    pbt = dual.get("Profit Before Tax", {}).get("current")
    tax = dual.get("Taxation Charge", {}).get("current")
    eps_c = dual.get("Earnings Per Share (AED)", {}).get("current")
    eps_p = dual.get("Earnings Per Share (AED)", {}).get("prior")
    nii_islamic = dual.get("Net Interest + Islamic Income", {}).get("current")

    # Balances (current)
    gross  = single.get("Gross Loans")
    ecl    = single.get("ECL")
    npl    = single.get("NPLs")
    assets = single.get("Total Assets")
    cash_cb = single.get("Cash & CB")
    due_banks = single.get("Due from Banks")
    inv_sec = single.get("Investment Securities")
    equity = single.get("Equity (Group Total)")

    # Balances (prior where available in dual)
    gross_prior = dual.get("Gross Loans and Receivables (dual)", {}).get("prior")
    depo_cur    = dual.get("Customer and Islamic Deposits", {}).get("current")
    depo_prior  = dual.get("Customer and Islamic Deposits", {}).get("prior")

    # --- Core set ---
    ratios = [
        ("Cost-to-Income",        safe_div(ga, toi)),
        ("Net Profit Margin",     safe_div(pat, toi)),
        ("Pre-impairment Margin", safe_div(opb, toi)),
        ("NPL Ratio",             safe_div(npl, gross)),
        ("Coverage Ratio",        safe_div(ecl, npl)),
        ("ECL/Gross Loans",       safe_div(ecl, gross)),
        ("Tax Rate",              safe_div(tax, pbt)),
        ("ROA",                   safe_div(pat, assets)),
        ("EPS YoY",               safe_div((eps_c - eps_p) if (eps_c is not None and eps_p is not None) else None, eps_p)),
    ]

    # --- Liquidity ---
    ratios += [
        ("Loan-to-Deposit (LDR)", safe_div(gross, depo_cur)),
        ("Liquid Assets % (Cash+Due+Inv)/Assets", safe_div(
            (cash_cb or 0) + (due_banks or 0) + (inv_sec or 0) if any(x is not None for x in [cash_cb, due_banks, inv_sec]) else None,
            assets
        )),
        # Placeholders (filtered out if None)
        ("NSFR (Basel III)", None),
        ("LCR (Basel III)", None),
    ]

    # --- Profitability / Efficiency additions ---
    earning_assets_proxy = None
    if any(v is not None for v in [gross, inv_sec, due_banks, cash_cb]):
        earning_assets_proxy = (gross or 0) + (inv_sec or 0) + (due_banks or 0) + (cash_cb or 0)
    ratios += [
        ("Operating Expense / Operating Income", safe_div(ga, toi)),
        ("Net Interest Margin (approx.)", safe_div(nii_islamic, earning_assets_proxy)),
        ("ROE", safe_div(pat, equity)),
        ("Interest Spread (placeholder)", None),
    ]

    # --- Asset quality enhancements ---
    net_npl_ratio = None
    if npl is not None and ecl is not None and gross is not None:
        numerator = max(npl - ecl, 0)
        denominator = gross - ecl if (gross - ecl) > 0 else None
        net_npl_ratio = safe_div(numerator, denominator)
    ratios += [
        ("Net NPL Ratio", net_npl_ratio),
        ("Stage 1 ECL / Gross Loans", None),
        ("Stage 2 ECL / Gross Loans", None),
        ("Stage 3 Coverage (ECL/Stage3)", None),
        ("Credit Cost Ratio", safe_div(None, gross)),
    ]

    # --- Growth (if prior values present) ---
    ratios += [
        ("Loan Growth YoY", safe_div((gross - gross_prior) if (gross is not None and gross_prior is not None) else None, gross_prior)),
        ("Deposit Growth YoY", safe_div((depo_cur - depo_prior) if (depo_cur is not None and depo_prior is not None) else None, depo_prior)),
    ]

    return filter_ratios(ratios)

def recommendations(ratios):
    recs = []
    d = dict(ratios)
    # Commented examples from earlier:
    # if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
    #     recs.append("High cost-to-income; review operating expenses.")
    # if d.get("NPL Ratio") is not None and d["NPL Ratio"] > 0.06:
    #     recs.append("NPL ratio elevated; examine credit concentrations.")
    if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
        recs.append("LDR is high; consider deposit growth or term funding.")
    if d.get("Liquid Assets % (Cash+Due+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Due+Inv)/Assets"] < 0.25:
        recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")
    return recs

def metrics_to_context(dual, single, ratios, units):
    lines = ["Key metrics & ratios (ENBD):"]
    if units and units.get("units_label"):
        lines.append(f"Units detected: {units.get('currency','AED')} {units['units_label']}")
    # Only include already-filtered metrics
    if dual:
        lines.append("\nP&L / Dual-column lines:")
        for k, v in dual.items():
            lines.append(f"  {k}: current={v['current']}, prior={v['prior']}")
    if single:
        lines.append("\nBalance / Single-column lines:")
        for k, v in single.items():
            lines.append(f"  {k}: {v}")
    if ratios:
        lines.append("\nRatios:")
        for name, val in ratios:
            lines.append(f"  {name}: {fmt_pct(val)}")
    return "\n".join(lines)

# --- Jinja filters ---
@app.template_filter("pct")
def pct(v): return fmt_pct(v)

@app.template_filter("fmt_num")
def jinja_fmt_num(v):
    if v is None: return "N/A"
    try: return f"{float(v):,.2f}"
    except Exception: return str(v)

# ---------- Enhanced ChatGPT-Style Template ----------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ENBD Financial Analyzer - AI Assistant</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
      :root {
        --primary-color: #10a37f;
        --secondary-color: #0066cc;
        --bg-light: #f7f7f8;
        --border-color: #e5e5e5;
        --text-muted: #6b7280;
        --shadow: 0 2px 4px rgba(0,0,0,0.1);
      }

      body {
        background-color: var(--bg-light);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }

      .card { 
        margin-bottom: 24px; 
        border: none;
        box-shadow: var(--shadow);
        border-radius: 12px;
      }
      
      .badge { font-size: 12px; }
      .monospace { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
      
      /* ChatGPT-style chat interface */
      .chat-container {
        height: 600px;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        background: white;
        box-shadow: var(--shadow);
        overflow: hidden;
      }
      
      .chat-header {
        padding: 16px 20px;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      
      .chat-header h5 {
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .model-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
      }
      
      .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        background: #fafafa;
        scroll-behavior: smooth;
      }
      
      .message {
        margin-bottom: 24px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
      }
      
      .message.user {
        flex-direction: row-reverse;
      }
      
      .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      }
      
      .message.user .message-avatar {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
      }
      
      .message.assistant .message-avatar {
        background: linear-gradient(135deg, var(--primary-color), #0d9488);
        color: white;
      }
      
      .message-content {
        padding: 16px 20px;
        border-radius: 20px;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        position: relative;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
      }
      
      .message.user .message-content {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
      }
      
      .message.assistant .message-content {
        background: white;
        color: #374151;
        border: 1px solid var(--border-color);
      }
      
      .chat-input-container {
        padding: 20px;
        border-top: 1px solid var(--border-color);
        background: white;
      }
      
      .chat-input-form {
        display: flex;
        gap: 12px;
        align-items: flex-end;
      }
      
      .chat-input-wrapper {
        flex: 1;
      }
      
      .chat-input {
        width: 100%;
        resize: vertical;
        min-height: 48px;
        max-height: 120px;
        border: 2px solid var(--border-color);
        border-radius: 24px;
        padding: 14px 20px;
        font-size: 15px;
        line-height: 1.4;
        transition: all 0.2s ease;
        background: #f8f9fa;
      }
      
      .chat-input:focus {
        outline: none;
        border-color: var(--primary-color);
        background: white;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
      }
      
      .chat-controls {
        display: flex;
        gap: 8px;
        flex-direction: column;
      }
      
      .btn-chat {
        border-radius: 24px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        min-width: 100px;
      }
      
      .btn-chat:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      
      .btn-primary-chat {
        background: var(--primary-color);
        color: white;
      }
      
      .btn-reset {
        background: #f59e0b;
        color: white;
      }
      
      .no-messages {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
      }
      
      .no-messages i {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
      }
      
      /* Scrollbar styling */
      .chat-messages::-webkit-scrollbar {
        width: 6px;
      }
      
      .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
      }
      
      .chat-messages::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
      }
      
      .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
      }
    </style>
</head>
<body>

<div class="container my-4">
  <div class="card">
    <div class="card-body">
      <h4 class="card-title">
        <i class="fas fa-file-upload me-2"></i>Upload ENBD Financial Statement PDF
        {% if has_context %}
          <span class="badge text-bg-success ms-2"><i class="fas fa-check"></i> Context: Active</span>
        {% else %}
          <span class="badge text-bg-secondary ms-2"><i class="fas fa-times"></i> Context: None</span>
        {% endif %}
        {% if units_label %}
          <span class="badge text-bg-info ms-2"><i class="fas fa-coins"></i> {{ currency }} {{ units_label }}</span>
        {% endif %}
      </h4>
      <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
        <div class="row g-3 align-items-center">
          <div class="col-auto">
            <input class="form-control" type="file" name="pdf_file" accept=".pdf" required>
          </div>
          <div class="col-auto">
            <button class="btn btn-primary" type="submit">
              <i class="fas fa-chart-line me-2"></i>Analyze
            </button>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-secondary" href="{{ url_for('clear') }}">
              <i class="fas fa-eraser me-2"></i>Clear Context
            </a>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-danger" href="{{ url_for('reset_all') }}">
              <i class="fas fa-trash-alt me-2"></i>Reset All
            </a>
          </div>
          <div class="col-auto">
            <a class="btn btn-outline-dark" href="{{ url_for('debug') }}">
              <i class="fas fa-bug me-2"></i>Debug
            </a>
          </div>
        </div>
      </form>
      {% if upload_error %}<div class="alert alert-danger mt-3"><i class="fas fa-exclamation-triangle me-2"></i>{{ upload_error }}</div>{% endif %}
    </div>
  </div>

  {% if dual or single %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title">
        <i class="fas fa-analytics me-2"></i>Extracted Financial Metrics 
        {% if units_label %}<small class="text-muted">(Values in {{ currency }} {{ units_label }})</small>{% endif %}
      </h3>
      <div class="row">
        {% if dual %}
        <div class="col-md-7">
          <h5><i class="fas fa-chart-bar me-2"></i>Income Statement (Current vs Prior)</h5>
          <div class="table-responsive">
            <table class="table table-sm table-striped align-middle">
              <thead class="table-dark"><tr><th>Line Item</th><th class="text-end">Current</th><th class="text-end">Prior</th></tr></thead>
              <tbody>
                {% for k,v in dual.items() %}
                  <tr>
                    <td><strong>{{ k }}</strong></td>
                    <td class="text-end">{{ v.current|fmt_num }}</td>
                    <td class="text-end">{{ v.prior|fmt_num }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
        {% endif %}
        {% if single %}
        <div class="col-md-5">
          <h5><i class="fas fa-balance-scale me-2"></i>Balance Sheet Items</h5>
          <div class="table-responsive">
            <table class="table table-sm table-striped">
              <thead class="table-dark"><tr><th>Item</th><th class="text-end">Value</th></tr></thead>
              <tbody>
                {% for k,v in single.items() %}
                  <tr><td><strong>{{ k }}</strong></td><td class="text-end">{{ v|fmt_num }}</td></tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
  {% endif %}

  {% if ratios %}
  <div class="card">
    <div class="card-body">
      <h3 class="card-title"><i class="fas fa-calculator me-2"></i>Financial Ratios</h3>
      <div class="row">
        {% for name,val in ratios %}
          <div class="col-md-6 col-lg-4 mb-3">
            <div class="p-3 bg-light rounded">
              <div class="fw-bold text-primary">{{ name }}</div>
              <div class="fs-4 fw-bold">{{ val|pct }}</div>
            </div>
          </div>
        {% endfor %}
      </div>
      {% if recs %}
        <hr>
        <h5><i class="fas fa-lightbulb me-2"></i>Recommendations</h5>
        <div class="alert alert-info">
          <ul class="mb-0">{% for r in recs %}<li>{{ r }}</li>{% endfor %}</ul>
        </div>
      {% endif %}
    </div>
  </div>
  {% endif %}

  <div class="card">
    <div class="chat-container">
      <div class="chat-header">
        <h5><i class="fas fa-robot me-2"></i>AI Financial Assistant</h5>
        <div class="model-badge">{{ model_badge }}</div>
      </div>
      
      <div class="chat-messages" id="chatMessages">
        {% if chat_history %}
          {% for msg in chat_history %}
            <div class="message {{ msg.role }}">
              <div class="message-avatar">
                {% if msg.role == 'user' %}<i class="fas fa-user"></i>{% else %}<i class="fas fa-robot"></i>{% endif %}
              </div>
              <div class="message-content">{{ msg.content }}</div>
            </div>
          {% endfor %}
          <div id="chat-bottom-anchor"></div>
        {% else %}
          <div class="no-messages">
            <i class="fas fa-comments"></i>
            <h5>Ready to analyze your ENBD financial data!</h5>
            <p>Upload a PDF above, then ask me about profitability, efficiency, risk metrics, or any financial insights.</p>
          </div>
        {% endif %}
      </div>
      
      <div class="chat-input-container">
        {% if not has_context %}
          <div class="alert alert-info mb-3">
            <i class="fas fa-info-circle me-2"></i>
            Upload a financial statement first for contextual analysis, or ask general questions.
          </div>
        {% endif %}
        
        <form method="post" action="{{ url_for('ask') }}" class="chat-input-form">
          <div class="chat-input-wrapper">
            <textarea 
              class="chat-input" 
              name="prompt" 
              rows="2" 
              placeholder="Ask about profitability, cost efficiency, credit quality, Islamic banking..."
              required
            ></textarea>
          </div>
          <div class="chat-controls">
            <button type="submit" class="btn btn-chat btn-primary-chat">
              <i class="fas fa-paper-plane me-2"></i>Send
            </button>
            <a href="{{ url_for('reset_chat') }}" class="btn btn-chat btn-reset">
              <i class="fas fa-refresh me-2"></i>Reset
            </a>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>

<script>
// Improved auto-scroll to bottom of chat messages
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const chatContainer = document.querySelector('.chat-container');
    
    // Function to scroll to bottom
    function scrollToBottom() {
        const anchor = document.getElementById('chat-bottom-anchor');
        if (anchor) {
            anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Also scroll the page to keep chat visible
        setTimeout(() => {
            if (chatContainer) {
                chatContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        }, 300);
    }
    
    // Scroll on page load if there are messages
    if (!chatMessages.querySelector('.no-messages')) {
        scrollToBottom();
    }
    
    // Aggressively clear all form inputs to prevent browser prefill
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.value = '';
        if (fileInput.files) {
            fileInput.files = null;
        }
    }
    
    // Clear textarea completely
    const textarea = document.querySelector('textarea[name="prompt"]');
    if (textarea) {
        textarea.value = '';
        textarea.innerHTML = '';
        textarea.textContent = '';
    }
    
    // Clear any other form elements
    const allInputs = document.querySelectorAll('input, textarea, select');
    allInputs.forEach(input => {
        if (input.type !== 'submit' && input.type !== 'button') {
            if (input.type === 'file') {
                input.value = '';
            } else if (input.tagName.toLowerCase() === 'textarea') {
                input.value = '';
            }
        }
    });
});

// Auto-resize textarea with error protection
const textarea = document.querySelector('textarea[name="prompt"]');
if (textarea) {
    // Clear on focus to ensure no prefill
    textarea.addEventListener('focus', function() {
        if (!this.value.trim()) {
            this.value = '';
        }
    });
    
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}
</script>

</body>
</html>
"""

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    # Clear ALL session data on fresh home load to prevent any prefill
    _doc_store_delete(session.get("financial_doc_id"))
    session.clear()
    
    units = {"currency": "AED", "units_label": None}  # Default values

    model_badge = _ui_model_badge(False, None)
    
    return render_template_string(
        TEMPLATE,
      model_badge=model_badge,
        has_context=False,  # Always false on fresh load
        dual={},
        single={},
        ratios=[],
        recs=[],
        currency=units.get("currency"),
        units_label=units.get("units_label"),
        chat_history=[],  # Always empty on fresh load
        prompt=None,  # Always None on home load
        upload_error=None
    )

@app.route("/upload", methods=["POST"])
def upload():
    # Clear all previous data on new upload
    _doc_store_delete(session.get("financial_doc_id"))
    for k in ["financial_context", "financial_dual", "financial_single", "financial_ratios", "financial_units", "chat_history", "financial_doc_id"]:
        session.pop(k, None)

    f = request.files.get("pdf_file")
    if not f or f.filename == "":
      model_badge = _ui_model_badge(False, None)
      return render_template_string(
        TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
        currency="AED", units_label=None, chat_history=[],
        upload_error="Please select an ENBD PDF file.", prompt=None,
        model_badge=model_badge,
      )
    if not f.filename.lower().endswith(".pdf"):
      model_badge = _ui_model_badge(False, None)
      return render_template_string(
        TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
        currency="AED", units_label=None, chat_history=[],
        upload_error="Please upload a PDF file only.", prompt=None,
        model_badge=model_badge,
      )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        pdf_text, dual_raw, single_raw, dual, single, units = parse_pdf(tmp.name)
    except Exception as e:
        print("[upload] error:", e)
        model_badge = _ui_model_badge(False, None)
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="AED", units_label=None, chat_history=[],
            upload_error=f"Error processing PDF: {e}", prompt=None,
            model_badge=model_badge,
        )
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    ratios = compute_ratios(dual_raw, single_raw)
    recs   = recommendations(ratios)

    # Build retrieval index over PDF text and store outside cookie session
    chunks = _chunk_text(pdf_text or "", chunk_size=1200, overlap=160) if pdf_text else []
    vectorizer, matrix = _build_retriever(chunks)

    # Create doc_id early so we can tie a Chroma collection to it.
    doc_id = os.urandom(12).hex()
    chroma_collection = _index_chunks_in_chroma(doc_id=doc_id, chunks=chunks)

    with _DOC_STORE_LOCK:
        _DOC_STORE[doc_id] = StoredDoc(
            created_at=time.time(),
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=matrix,
            chroma_collection=chroma_collection,
        )
        _doc_store_cleanup_locked()

    # Persist new data with fresh chat
    session["financial_units"]  = units
    session["financial_dual"]   = dual
    session["financial_single"] = single
    session["financial_ratios"] = ratios
    session["financial_context"] = metrics_to_context(dual, single, ratios, units)
    session["chat_history"] = []
    session["financial_doc_id"] = doc_id

    model_badge = _ui_model_badge(True, _doc_store_get(doc_id))

    return render_template_string(
        TEMPLATE, has_context=True,
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
        chat_history=[], upload_error=None, prompt=None,
        model_badge=model_badge,
    )

@app.route("/ask", methods=["POST"])
def ask():
    prompt = (request.form.get("prompt") or "").strip()
    context = session.get("financial_context")
    dual    = session.get("financial_dual") or {}
    single  = session.get("financial_single") or {}
    ratios  = session.get("financial_ratios") or []
    units   = session.get("financial_units") or {"currency": "AED", "units_label": None}

    doc = _doc_store_get(session.get("financial_doc_id"))

    # Get or initialize chat history
    chat_history = session.get("chat_history", [])
    
    if prompt:
        # Add user message to history
        chat_history.append({"role": "user", "content": prompt})

    answer = None
    error_msg = None

    if not prompt:
        error_msg = "Please enter a question."
    else:
        # Metric fast-path: if user asks a known ratio, return the computed value
        qn = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
        ratio_map = {str(k).lower(): v for (k, v) in (ratios or [])}
        metric_aliases = {
            "roa": "roa",
            "return on assets": "roa",
            "roe": "roe",
            "return on equity": "roe",
            "npl": "npl ratio",
            "npl ratio": "npl ratio",
            "cost to income": "cost-to-income",
            "cost-to-income": "cost-to-income",
            "loan to deposit": "loan-to-deposit (ldr)",
            "ldr": "loan-to-deposit (ldr)",
        }

        asked_key = None
        for alias, key in metric_aliases.items():
            if (
                qn == alias
                or qn.endswith(" " + alias)
                or qn.startswith(alias + " ")
                or (" " + alias + " ") in (" " + qn + " ")
            ):
                asked_key = key
                break

        if asked_key and asked_key in ratio_map and ratio_map[asked_key] is not None:
            answer = f"{asked_key.upper()}: {fmt_pct(ratio_map[asked_key])}"
            chat_history.append({"role": "assistant", "content": answer})
        else:
            try:
                metric_answer = _try_metric_fast_path(prompt, dual=dual, single=single)
                if metric_answer:
                    answer = metric_answer
                    chat_history.append({"role": "assistant", "content": answer})
                else:
                    retrieval_q = _normalize_query(prompt)
                    excerpts = _retrieve_top_chunks(doc, retrieval_q or prompt, top_k=4) if doc else []
                    computed = context or ""
                    answer = _hf_answer(prompt, computed_context=computed, excerpts=excerpts, chat_history=chat_history)
                    chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"AI service error: {str(e)}"
                print(f"[ask] HF error: {e}")

    # Keep only last 20 messages
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # Save updated chat history
    session["chat_history"] = chat_history

    recs = recommendations(ratios) if ratios else []

    model_badge = _ui_model_badge(bool(context), doc)

    return render_template_string(
        TEMPLATE, has_context=bool(context),
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
      chat_history=chat_history, upload_error=None, prompt=None,
      model_badge=model_badge,
    )

@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("home"))

@app.route("/reset_all")
def reset_all():
    """Reset everything - clear all session data for fresh start"""
    _doc_store_delete(session.get("financial_doc_id"))
    session.clear()
    return redirect(url_for("home"))

@app.route("/clear")
def clear():
  _doc_store_delete(session.get("financial_doc_id"))
  for k in ["financial_context", "financial_ratios", "financial_dual", "financial_single", "financial_units", "chat_history", "financial_doc_id"]:
    session.pop(k, None)
  return redirect(url_for("home"))

@app.route("/debug")
def debug():
    u = session.get("financial_units") or {}
    return jsonify({
        "has_context": bool(session.get("financial_context")),
        "has_ratios": bool(session.get("financial_ratios")),
        "dual_keys": list((session.get("financial_dual") or {}).keys()),
        "single_keys": list((session.get("financial_single") or {}).keys()),
        "units": u,
        "llm_backend": "huggingface",
        "hf_model_id": HF_MODEL_ID,
        "context_length": len(session.get("financial_context", "")),
        "session_keys": list(session.keys()),
        "chat_history_length": len(session.get("chat_history", [])),
        "has_doc_store": bool(_doc_store_get(session.get("financial_doc_id"))),
    })

# Optional CLI mode
def cli_chat():
    context = ""
    doc: Optional[StoredDoc] = None
    try:
        pdf_path = input("PDF path for context (Enter to skip): ").strip()
        if pdf_path:
            pdf_text, dual_raw, single_raw, dual, single, units = parse_pdf(pdf_path)
            ratios_local = compute_ratios(dual_raw, single_raw)
            context = metrics_to_context(dual, single, ratios_local, units)
            chunks = _chunk_text(pdf_text or "", chunk_size=1200, overlap=160) if pdf_text else []
            vectorizer, matrix = _build_retriever(chunks)
            doc = StoredDoc(created_at=time.time(), chunks=chunks, vectorizer=vectorizer, tfidf_matrix=matrix)
            print("Parsed PDF. Context prepared.")
        else:
            print("No PDF context. Chatting without financial context.")
    except Exception as e:
        print(f"[PDF parse skipped] {e}")

    print("\n💬 Chat mode (type 'q' to quit)")
    while True:
        q = input("\nYour prompt> ").strip()
        if q.lower() == "q":
            print("Bye!")
            break
        if not q:
            continue
        try:
          excerpts = _retrieve_top_chunks(doc, q, top_k=4) if doc else []
          ans = _hf_answer(q, computed_context=context, excerpts=excerpts, chat_history=[])
          print("\nAssistant:", ans.strip())
        except Exception as e:
          print(f"[HF error] {e}")

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    cli_chat()
  else:
    _warn_if_not_using_venv()
    print("🏦 ENBD Financial Analyzer with ChatGPT-style AI Assistant")
    print("🌐 Running on http://127.0.0.1:5089")
    print(f"🤖 Powered by Hugging Face: {HF_MODEL_ID}")
    app.run(host="127.0.0.1", port=5089, debug=True, use_reloader=False)
