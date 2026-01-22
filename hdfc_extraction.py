from flask import Flask, request, render_template_string, session, redirect, url_for, jsonify
import fitz, tempfile, re, os, io, sys, time
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

# ---- OCR deps
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pytesseract import TesseractNotFoundError

# ---- Extras for robust Tesseract detection + logging
import platform, shutil, logging

load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "300"))

# --- Optional ChromaDB + Embeddings retrieval ---
USE_VECTOR_DB = os.getenv("USE_VECTOR_DB", "1").strip().lower() in {"1", "true", "yes"}
CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_hdfc")
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "hdfc_pdf")

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
        "Failed to import sentence-transformers. If you're on Windows, ensure you're running the workspace venv."
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

  word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
  word_matrix = word_vectorizer.fit_transform(chunks)

  char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
  char_matrix = char_vectorizer.fit_transform(chunks)

  bundle = {"word": word_vectorizer, "char": char_vectorizer}
  matrices = {"word": word_matrix, "char": char_matrix}
  return bundle, matrices


def _retrieve_top_chunks(doc: StoredDoc, question: str, top_k: int = 4) -> List[str]:
  q = (question or "").strip()
  if not q or not doc or not doc.chunks:
    return []

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


def _token_len(gen: Any, text: str) -> int:
  try:
    tok = getattr(gen, "tokenizer", None)
    if tok is None:
      return len((text or "").split())

    enc = tok(
      text or "",
      add_special_tokens=True,
      truncation=False,
      return_attention_mask=False,
      return_token_type_ids=False,
    )
    ids = enc.get("input_ids")
    if ids is None:
      return len((text or "").split())
    # Some tokenizers return List[List[int]]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
      return int(len(ids[0]))
    return int(len(ids))
  except Exception:
    return len((text or "").split())


def _truncate_prompt_to_model_limit(gen: Any, prompt: str) -> str:
  """Hard truncate prompt to the model's max input tokens to avoid indexing errors."""
  tok = getattr(gen, "tokenizer", None)
  if tok is None:
    return prompt

  max_in = _model_max_input_tokens(gen)
  try:
    enc = tok(
      prompt or "",
      add_special_tokens=True,
      truncation=True,
      max_length=max_in,
      return_attention_mask=False,
      return_token_type_ids=False,
    )
    ids = enc.get("input_ids")
    if ids is None:
      return prompt
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
      ids = ids[0]
    return tok.decode(ids, skip_special_tokens=True)
  except Exception:
    return prompt


def _model_max_input_tokens(gen: Any) -> int:
  tok = getattr(gen, "tokenizer", None)
  if tok is None:
    return 512
  mx = getattr(tok, "model_max_length", None)
  try:
    mx_i = int(mx)
    if mx_i <= 0 or mx_i > 4096:
      return 512
    return mx_i
  except Exception:
    return 512


def _build_budgeted_prompt(
  gen: Any,
  question: str,
  computed_context: str,
  excerpts: List[str],
  chat_history: List[Dict[str, str]],
) -> str:
  system = (
    "You are a financial analyst specializing in Indian banking. "
    "Be concise and numeric. Use ONLY the provided context. "
    "If a requested value is not present, say you cannot find it in the provided context."
  )

  q = (question or "").strip()
  computed = (computed_context or "").strip()

  recent = chat_history[-3:] if chat_history else []
  recent_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

  keep_excerpts = list(excerpts or [])
  max_excerpt_chars = 900
  keep_excerpts = [str(ex)[:max_excerpt_chars] for ex in keep_excerpts]

  max_computed_chars = 2400
  if len(computed) > max_computed_chars:
    computed = computed[:max_computed_chars] + "\n...[truncated]"

  max_tokens = _model_max_input_tokens(gen)

  # Put QUESTION early so any truncation keeps it.
  def assemble(exs: List[str], comp: str, chat: str) -> str:
    parts: List[str] = [system]
    parts.append("QUESTION:\n" + q)
    if chat:
      parts.append("RECENT CHAT:\n" + chat)
    if comp:
      parts.append("COMPUTED METRICS & RATIOS:\n" + comp)
    if exs:
      parts.append("PDF EXCERPTS:\n" + "\n\n".join(f"[Excerpt {i+1}]\n{ex}" for i, ex in enumerate(exs)))
    return "\n\n".join(parts)

  prompt = assemble(keep_excerpts, computed, recent_text)
  target = max(128, max_tokens - 64)

  while _token_len(gen, prompt) > target:
    if keep_excerpts:
      keep_excerpts = keep_excerpts[: max(0, len(keep_excerpts) - 1)]
      prompt = assemble(keep_excerpts, computed, recent_text)
      continue
    if recent_text:
      recent_text = ""
      prompt = assemble(keep_excerpts, computed, recent_text)
      continue
    if len(computed) > 600:
      computed = computed[: max(200, int(len(computed) * 0.7))] + "\n...[truncated]"
      prompt = assemble(keep_excerpts, computed, recent_text)
      continue
    break

  return prompt


def _normalize_query(text: str) -> str:
  s = (text or "").lower()
  s = re.sub(r"[^a-z0-9%]+", " ", s)
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

    # Heuristics for metric lookups:
    # - If the choice phrase appears in the query (e.g. "roa" in "what is roa"), treat as strong match.
    # - If all choice tokens are present in the query, also treat as strong match.
    if c in q:
      score = max(score, 0.99)
    if c_tokens and c_tokens.issubset(q_tokens):
      score = max(score, 0.92)
    # Acronyms like ROA/LDR/GNPA are short; exact token presence should win.
    if len(c) <= 5 and c in q_tokens:
      score = max(score, 0.98)

    if score > best_score:
      best_score = score
      best = ch

  return best, best_score


def _metric_synonyms_hdfc() -> Dict[str, List[str]]:
  return {
    "ROA (reported)": ["roa", "return on assets", "return on asset", "roa reported"],
    "Cost-to-Income": ["cost to income", "c/i", "cost income", "cost-to-income"],
    "Loan-to-Deposit (LDR)": ["ldr", "loan to deposit", "loan-to-deposit", "loan deposit ratio"],
    "Gross NPA / Advances": ["gross npa", "gross npa ratio", "gnpa"],
    "Net NPA / Advances": ["net npa", "net npa ratio", "nnpa"],
    "Net Profit for the period": ["net profit", "profit after tax", "pat", "net profit for the period"],
  }


def _try_metric_fast_path(prompt: str, dual: Dict[str, Any], single: Dict[str, Any], ratios: List[Tuple[str, Any]]) -> Optional[Dict[str, Any]]:
  """Match a user query to an extracted metric/ratio.

  Returns a dict describing the match so we can answer deterministically.
  """
  if not prompt:
    return None

  ratio_map = dict(ratios or [])

  key_to_phrases: Dict[str, List[str]] = {}
  for k in list((dual or {}).keys()) + list((single or {}).keys()) + list((ratio_map or {}).keys()):
    key_to_phrases.setdefault(k, []).append(k)

  # Add synonyms only for keys that exist.
  for k, syns in _metric_synonyms_hdfc().items():
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

  if score < 0.58:
    return None

  key = phrase_to_key.get(best_phrase)
  if not key:
    return None

  if key in ratio_map:
    val = ratio_map.get(key)
    if val is None:
      return None
    return {"kind": "ratio", "key": key, "value": val}

  if key in (dual or {}):
    v = dual[key]
    cur = v.get("current")
    pri = v.get("prior")
    if cur is None and pri is None:
      return None
    return {"kind": "dual", "key": key, "current": cur, "prior": pri}

  if key in (single or {}):
    v = single[key]
    if v is None:
      return None
    return {"kind": "single", "key": key, "value": v}

  return None


def _format_fast_answer(match: Dict[str, Any], units: Dict[str, Any]) -> str:
  currency = (units or {}).get("currency") or "INR"
  units_label = (units or {}).get("units_label") or ""
  units_part = f" ({currency} {units_label})".strip() if (currency or units_label) else ""

  kind = match.get("kind")
  key = match.get("key")
  if not key:
    return "I couldn't find that metric in the extracted data."

  if kind == "ratio":
    v = match.get("value")
    if v is None:
      return "I couldn't find that ratio in the extracted data."
    pct = fmt_pct(v) if isinstance(v, (int, float)) else str(v)
    return f"{key}: {pct} (raw={v})"

  if kind == "dual":
    cur = match.get("current")
    pri = match.get("prior")
    return f"{key}{units_part}: current={cur}, prior={pri}"

  if kind == "single":
    v = match.get("value")
    return f"{key}{units_part}: {v}"

  return "I couldn't find that metric in the extracted data."


def _hf_answer(question: str, computed_context: str, excerpts: List[str], chat_history: List[Dict[str, str]]) -> str:
  gen = _get_hf_generator(HF_MODEL_ID)

  prompt = _build_budgeted_prompt(
    gen,
    question=question,
    computed_context=computed_context,
    excerpts=excerpts,
    chat_history=chat_history,
  )

  # Final safety clamp: even if budgeting misestimates, never exceed model max input tokens.
  prompt = _truncate_prompt_to_model_limit(gen, prompt)

  try:
    out = gen(
      prompt,
      max_new_tokens=HF_MAX_NEW_TOKENS,
      do_sample=False,
      truncation=True,
    )
    if isinstance(out, list) and out:
      text = out[0].get("generated_text") or out[0].get("text")
      if text:
        return str(text).strip()
    return "I couldn't generate an answer right now."
  except Exception as e:
    return f"AI service error: {e}"

# If Tesseract is not on PATH (Windows), set it here (or via env var TESSERACT_CMD)
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")


# ===================== Tesseract Locator =====================
def ensure_tesseract_available():
    """
    Locate the Tesseract binary and wire it to pytesseract.
    Uses TESSERACT_CMD env, PATH, or common install locations.
    Returns (ok: bool, msg: str, path: str|None)
    """
    # 1) Respect env var if set
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        try:
            _ = pytesseract.get_tesseract_version()
            return True, f"Tesseract via TESSERACT_CMD: {env_path}", env_path
        except Exception as e:
            return False, f"Tesseract at TESSERACT_CMD failed: {e}", env_path

    # 2) PATH
    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        try:
            _ = pytesseract.get_tesseract_version()
            return True, f"Tesseract on PATH: {which}", which
        except Exception as e:
            return False, f"Tesseract on PATH failed: {e}", which

    # 3) Common install paths
    candidates = []
    sysname = platform.system().lower()
    if "windows" in sysname:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif "darwin" in sysname:  # macOS
        candidates = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    else:  # Linux
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]

    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                _ = pytesseract.get_tesseract_version()
                return True, f"Tesseract found at: {path}", path
            except Exception as e:
                return False, f"Tesseract found but failed to run: {e}", path

    return False, (
        "Tesseract not found. Install it and/or set TESSERACT_CMD to the full path, "
        "e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ), None


def check_ocr_dependencies():
    """Check if OCR dependencies are available and log status."""
    try:
        logging.basicConfig(level=logging.INFO)
    except Exception:
        pass
    ok, msg, path = ensure_tesseract_available()
    logging.info(msg)
    return ok, ("OCR available" if ok else f"OCR not available: {msg}")


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB


# ===================== Utilities =====================
def to_float(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def safe_div(a, b):
    return round(a / b, 4) if (a is not None and b not in (None, 0)) else None


def fmt_pct(x):
    return f"{x*100:.2f}%" if x is not None else "N/A"


def normalize_unit_label(label_raw):
    if not label_raw:
        return None
    s = str(label_raw).lower()
    if "billion" in s:  return "billions"
    if "million" in s:  return "millions"
    if "thousand" in s: return "thousands"
    if "crore" in s or "crores" in s: return "crores"
    if "lakh" in s or "lakhs" in s:   return "lakhs"
    return None


def detect_units(text):
    """
    Detect currency + magnitude. Handles '₹ in crore', 'Rs. in crores', etc.
    Falls back to AED phrasing if present (legacy).
    """
    currency = None
    units_label = None

    pats_inr = [
        r"(₹|INR|Rs\.?|Rupees)[^\n]{0,30}(in\s+)?(lakh|lakhs|crore|crores|million|millions|billion|billions)",
        r"(₹)\s*in\s*(crore|crores|lakh|lakhs|million|millions|billion|billions)",
        r"₹\s*in\s*crore"
    ]
    for p in pats_inr:
        m = re.search(p, text, re.I)
        if m:
            currency = "INR"
            groups = [g for g in m.groups() if g]
            for g in groups:
                if re.search(r"lakh|crore|million|billion", g, re.I):
                    units_label = normalize_unit_label(g)
            break

    if not currency:
        pats_aed = [
            r"(?:all amounts|figures)[^.\n]{0,80}(?:in|expressed in)[^.\n]{0,80}(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
            r"(AED|UAE\s*Dirhams)[^.\n]{0,40}(thousand|thousands|million|millions|billion|billions)",
            r"(AED)[^\n]{0,10}\((?:in\s+)?(thousand|thousands|million|millions|billion|billions)\)",
        ]
        for p in pats_aed:
            m = re.search(p, text, re.I)
            if m:
                currency = "AED"
                groups = [g for g in m.groups() if g]
                for g in groups:
                    if re.search(r"thousand|million|billion", g, re.I):
                        units_label = normalize_unit_label(g)
                break

    return {"currency": currency or "INR", "units_label": units_label or "crores"}


# ===================== OCR PIPELINE =====================
def preprocess_for_ocr(pix):
    """
    Convert a PyMuPDF pixmap to a denoised, high-contrast PIL image for OCR.
    """
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    img = img.point(lambda x: 255 if x > 180 else 0, mode='1')
    return img


def ocr_pdf_to_text(path, dpi=300, lang="eng"):
    ocr_available, _ = check_ocr_dependencies()
    if not ocr_available:
        return ""
    out = []
    with fitz.open(path) as doc:
        for pg in doc:
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            img = preprocess_for_ocr(pix)
            txt = pytesseract.image_to_string(
                img, lang=lang,
                config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
            )
            out.append(txt)
    return "\n".join(out)


def extract_text_embedded(path):
    """Pull embedded text using PyMuPDF."""
    with fitz.open(path) as doc:
        return "\n".join(pg.get_text("text") for pg in doc)


def extract_text_with_ocr_fallback(path):
    """
    Try native text extraction; if weak (scanned), OCR it; else merge if helpful.
    """
    text = extract_text_embedded(path)
    if len(text.strip()) >= 200:
        # Still try OCR; if OCR adds more digits/length, merge
        ocr = ocr_pdf_to_text(path) or ""
        d_native = sum(ch.isdigit() for ch in text)
        d_ocr = sum(ch.isdigit() for ch in ocr)
        if d_ocr > d_native * 1.10 or len(ocr) > len(text) * 1.10:
            base = ocr.splitlines()
            extra = [ln for ln in text.splitlines() if ln.strip() and ln not in base]
            return "\n".join(base + extra)
        return text
    # Likely scanned → OCR
    ocr = ocr_pdf_to_text(path) or ""
    if not ocr:
        # Final fallback: block/span crawl
        try:
            with fitz.open(path) as doc:
                text_blocks = []
                for page in doc:
                    blocks = page.get_text("dict")
                    for block in blocks.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    text_blocks.append(span.get("text", ""))
                return " ".join(text_blocks)
        except Exception:
            pass
    return ocr


# ===================== HDFC-specific extraction =====================
NUM = r"([\d,]+(?:\.\d+)?)"

# Two-column (Current vs Prior)
PATTERNS_DUAL = {
    # P&L
    "Interest earned": rf"Interest\s+earned.*?{NUM}\s+{NUM}",
    "Other income": rf"Other\s+Income.*?{NUM}\s+{NUM}",
    "Total income": rf"Total\s+Income.*?{NUM}\s+{NUM}",
    "Interest expended": rf"Interest\s+expended.*?{NUM}\s+{NUM}",
    "Operating expenses": rf"Operating\s+expenses.*?{NUM}\s+{NUM}",
    "Total expenditure": rf"Total\s+Expenditure.*?{NUM}\s+{NUM}",
    "Operating Profit before provisions and contingencies": rf"Operating\s+Profit\s+before\s+provisions\s+and\s+contingencies.*?{NUM}\s+{NUM}",
    "Provisions (other than tax) and Contingencies": rf"Provisions\s*\(other\s+than\s+tax\)\s*and\s*Contingencies.*?{NUM}\s+{NUM}",
    "Profit before tax": rf"Profit\s+from\s+ordinary\s+activities\s+before\s+tax.*?{NUM}\s+{NUM}",
    "Tax Expense": rf"Tax\s+Expense.*?{NUM}\s+{NUM}",
    "Net Profit for the period": rf"Net\s+Profit\s+(?:from\s+ordinary\s+activities\s+)?after\s+tax.*?{NUM}\s+{NUM}",
    # Balance (first two numeric cols if present twice)
    "Deposits (dual)": rf"Deposits\s+{NUM}\s+{NUM}",
    "Borrowings (dual)": rf"Borrowings\s+{NUM}\s+{NUM}",
    "Investments (dual)": rf"Investments\s+{NUM}\s+{NUM}",
    "Advances (dual)": rf"Advances\s+{NUM}\s+{NUM}",
}

# Single-column (Analytical ratios / BS)
PATTERNS_SINGLE = {
    "Gross NPAs": rf"Gross\s+NPAs\s+{NUM}",
    "Net NPAs": rf"Net\s+NPAs\s+{NUM}",
    "% of Gross NPAs to Gross Advances": r"%\s*of\s*Gross\s*NPAs\s*to\s*Gross\s*Advances\s+(1?\d+(?:\.\d+)?%)",
    "% of Net NPAs to Net Advances": r"%\s*of\s*Net\s*NPAs\s*to\s*Net\s*Advances\s+(0?\d+(?:\.\d+)?%)",
    "Return on assets (reported)": r"Return\s+on\s+assets\s*\(average\).*?(\d+(?:\.\d+)?%)",
    "Net worth": rf"Net\s+worth\s+{NUM}",
    # Statement of Assets & Liabilities (current)
    "Deposits": rf"Deposits\s+{NUM}",
    "Borrowings": rf"Borrowings\s+{NUM}",
    "Investments": rf"Investments\s+{NUM}",
    "Advances": rf"Advances\s+{NUM}",
    "Cash and balances with RBI": rf"Cash\s+and\s+balances\s+with\s+Reserve\s+Bank\s+of\s+India\s+{NUM}",
    "Balances with banks": rf"Balances\s+with\s+banks.*?{NUM}",
    "Other assets": rf"Other\s+assets\s+{NUM}",
    "Total assets (BS)": rf"Total\s+{NUM}\s+\d",
}


def _to_pct_str_or_float(maybe_str):
    if maybe_str is None:
        return None
    if isinstance(maybe_str, str) and maybe_str.strip().endswith("%"):
        try:
            return float(maybe_str.strip().strip("%")) / 100.0
        except Exception:
            return None
    return to_float(maybe_str)


def extract_dual(text):
    out = {}
    for k, p in PATTERNS_DUAL.items():
        m = re.search(p, text, re.I | re.S)
        out[k] = {
            "current": to_float(m.group(1)) if m else None,
            "prior": to_float(m.group(2)) if (m and m.lastindex and m.lastindex >= 2) else None,
        }
    return out


def extract_single(text):
    out = {}
    for k, p in PATTERNS_SINGLE.items():
        m = re.search(p, text, re.I | re.S)
        if m:
            g1 = m.group(1)
            if isinstance(g1, str) and g1.strip().endswith("%"):
                out[k] = _to_pct_str_or_float(g1.strip())
            else:
                out[k] = to_float(g1)
        else:
            out[k] = None
    return out


def parse_pdf(path):
    """Parse HDFC PDF with OCR fallback and unit detection."""
    print(f"[parse_pdf] path={path}")
    text = extract_text_with_ocr_fallback(path)
    print(f"[parse_pdf] extracted text len={len(text)}")
    if text:
        print("[parse_pdf] text sample:", repr(text[:300]))
    units = detect_units(text)
    dual_raw = extract_dual(text)
    single_raw = extract_single(text)

    # Filter for display context
    dual = {k: v for k, v in dual_raw.items() if (v.get("current") is not None or v.get("prior") is not None)}
    single = {k: v for k, v in single_raw.items() if v is not None}

    return text, dual_raw, single_raw, dual, single, units


# ===================== Ratios & Recs =====================
def compute_ratios(dual_raw, single_raw):
    interest_earned = (dual_raw.get("Interest earned") or {}).get("current")
    interest_exp    = (dual_raw.get("Interest expended") or {}).get("current")
    other_inc       = (dual_raw.get("Other income") or {}).get("current")
    oper_exp        = (dual_raw.get("Operating expenses") or {}).get("current")
    op_profit       = (dual_raw.get("Operating Profit before provisions and contingencies") or {}).get("current")
    pbt             = (dual_raw.get("Profit before tax") or {}).get("current")
    tax             = (dual_raw.get("Tax Expense") or {}).get("current")
    pat             = (dual_raw.get("Net Profit for the period") or {}).get("current")

    deposits        = single_raw.get("Deposits") or (dual_raw.get("Deposits (dual)") or {}).get("current")
    advances        = single_raw.get("Advances") or (dual_raw.get("Advances (dual)") or {}).get("current")
    investments     = single_raw.get("Investments") or (dual_raw.get("Investments (dual)") or {}).get("current")
    cash_rbi        = single_raw.get("Cash and balances with RBI")
    bal_banks       = single_raw.get("Balances with banks")
    total_assets    = single_raw.get("Total assets (BS)")

    gross_npa       = single_raw.get("Gross NPAs")
    net_npa         = single_raw.get("Net NPAs")
    roa_reported    = single_raw.get("Return on assets (reported)")

    nii = None
    if interest_earned is not None and interest_exp is not None:
        nii = interest_earned - interest_exp

    net_revenue = None
    if nii is not None:
        net_revenue = nii + (other_inc or 0)

    ratios = [
        ("Cost-to-Income",        safe_div(oper_exp, net_revenue)),
        ("Pre-provision Operating Margin", safe_div(op_profit, net_revenue)),
        ("Tax Rate",              safe_div(tax, pbt)),
        ("Net Profit Margin (on net revenue)", safe_div(pat, net_revenue)),
        ("Loan-to-Deposit (LDR)", safe_div(advances, deposits)),
        ("Liquid Assets % (Cash+Banks+Inv)/Assets", safe_div(
            (cash_rbi or 0) + (bal_banks or 0) + (investments or 0)
            if any(x is not None for x in [cash_rbi, bal_banks, investments]) else None,
            total_assets
        )),
        ("ROA (reported)", roa_reported),  # already converted to fraction if % found
        ("Gross NPA / Advances",  safe_div(gross_npa, advances)),
        ("Net NPA / Advances",    safe_div(net_npa, advances)),
    ]
    return [(n, v) for n, v in ratios if v is not None]


def recommendations(ratios):
    recs = []
    d = dict(ratios)
    if d.get("Cost-to-Income") is not None and d["Cost-to-Income"] > 0.50:
        recs.append("High cost-to-income; review operating expenses.")
    if d.get("Gross NPA / Advances") is not None and d["Gross NPA / Advances"] > 0.035:
        recs.append("Gross NPA ratio looks elevated; strengthen recoveries and provisioning.")
    if d.get("Loan-to-Deposit (LDR)") is not None and d["Loan-to-Deposit (LDR)"] > 0.95:
        recs.append("LDR is high; consider deposit growth or term funding.")
    if d.get("Liquid Assets % (Cash+Banks+Inv)/Assets") is not None and d["Liquid Assets % (Cash+Banks+Inv)/Assets"] < 0.25:
        recs.append("Low liquid-asset buffer; monitor LCR/NSFR and short-term gaps.")
    return recs


def metrics_to_context(dual, single, ratios, units):
    lines = ["Key metrics & ratios (HDFC):"]
    if units and units.get("units_label"):
        lines.append(f"Units detected: {units.get('currency','INR')} {units['units_label']}")
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


# ---------- Enhanced ChatGPT-Style Template with Simple Form ----------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDFC Financial Analyzer - AI Assistant</title>
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
        <i class="fas fa-file-upload me-2"></i>Upload Financial Statement PDF
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
          <div class="col-auto">
            <a class="btn btn-outline-info" href="{{ url_for('test_pdf') }}">
              <i class="fas fa-vial me-2"></i>Test PDF
            </a>
          </div>
        </div>
      </form>
      {% if upload_error %}<div class="alert alert-danger mt-3"><i class="fas fa-exclamation-triangle me-2"></i>{{ upload_error }}</div>{% endif %}
      {% if not ocr_available %}
        <div class="alert alert-warning mt-3">
          <i class="fas fa-exclamation-triangle me-2"></i>
          OCR is disabled (Tesseract not found). Using embedded PDF text only; scanned documents may not parse optimally.
        </div>
      {% endif %}
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
        <div class="model-badge">HF: {{ hf_model_id }}</div>
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
            <h5>Ready to analyze your financial data!</h5>
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
              placeholder="Ask about profitability, efficiency, risk analysis, trends..."
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
    try:
        _doc_store_delete(session.get("hdfc_doc_id"))
    except Exception:
        pass
    session.clear()

    ocr_available, ocr_status = check_ocr_dependencies()
    units = {"currency": "INR", "units_label": "crores"}  # Default values

    return render_template_string(
        TEMPLATE,
        has_context=False,  # Always false on fresh load
        dual={},
        single={},
        ratios=[],
        recs=[],
        currency=units.get("currency"),
        units_label=units.get("units_label"),
        ocr_available=ocr_available,
        hf_model_id=HF_MODEL_ID,
        chat_history=[],  # Always empty on fresh load
        prompt=None,  # Always None on home load
        upload_error=None
    )


@app.route("/upload", methods=["POST"])
def upload():
    # Clear all previous data on new upload
    try:
        _doc_store_delete(session.get("hdfc_doc_id"))
    except Exception:
        pass

    for k in ["hdfc_context", "hdfc_dual", "hdfc_single", "hdfc_ratios", "hdfc_recs", "chat_history", "hdfc_doc_id"]:
        session.pop(k, None)
        
    f = request.files.get("pdf_file")
    if not f or f.filename == "":
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[], 
            upload_error="Please select an HDFC PDF file.", ocr_available=check_ocr_dependencies()[0],
          hf_model_id=HF_MODEL_ID,
            prompt=None
        )
    if not f.filename.lower().endswith(".pdf"):
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[],
            upload_error="Please upload a PDF file only.", ocr_available=check_ocr_dependencies()[0],
          hf_model_id=HF_MODEL_ID,
            prompt=None
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        f.save(tmp.name)
        extracted_text, dual_raw, single_raw, dual, single, units = parse_pdf(tmp.name)
    except Exception as e:
        print("[upload] error:", e)
        return render_template_string(
            TEMPLATE, has_context=False, dual={}, single={}, ratios=[], recs=[],
            currency="INR", units_label="crores", chat_history=[],
            upload_error=f"Error processing PDF: {e}", ocr_available=check_ocr_dependencies()[0],
          hf_model_id=HF_MODEL_ID,
            prompt=None
        )
    finally:
        try:
            tmp.close(); os.unlink(tmp.name)
        except Exception:
            pass

    ratios = compute_ratios(dual_raw, single_raw)
    recs   = recommendations(ratios)

    # Build retriever + optional Chroma index for PDF excerpts
    chunks = _chunk_text(extracted_text)
    vectorizer, tfidf_matrix = _build_retriever(chunks)
    stored_doc = StoredDoc(created_at=time.time(), chunks=chunks, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)
    doc_id = _doc_store_put(stored_doc)
    chroma_collection = _index_chunks_in_chroma(doc_id=doc_id, chunks=chunks)
    if chroma_collection:
        doc_ref = _doc_store_get(doc_id)
        if doc_ref is not None:
            doc_ref.chroma_collection = chroma_collection

    # Persist new data with fresh chat
    session["hdfc_units"]  = units
    session["hdfc_dual"]   = dual
    session["hdfc_single"] = single
    session["hdfc_ratios"] = ratios
    session["hdfc_recs"]   = recs
    session["hdfc_context"] = metrics_to_context(dual, single, ratios, units)
    session["chat_history"] = []
    session["hdfc_doc_id"] = doc_id

    return render_template_string(
        TEMPLATE, has_context=True,
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
      chat_history=[], upload_error=None, ocr_available=check_ocr_dependencies()[0],
      hf_model_id=HF_MODEL_ID,
        prompt=None
    )


@app.route("/ask", methods=["POST"])
def ask():
    prompt = (request.form.get("prompt") or "").strip()
    context = session.get("hdfc_context")
    dual    = session.get("hdfc_dual") or {}
    single  = session.get("hdfc_single") or {}
    ratios  = session.get("hdfc_ratios") or []
    recs    = session.get("hdfc_recs") or []
    units   = session.get("hdfc_units") or {"currency": "INR", "units_label": "crores"}
    doc_id = session.get("hdfc_doc_id")
    doc = _doc_store_get(doc_id)

    # Get or initialize chat history
    chat_history = session.get("chat_history", [])
    
    if prompt:
        # Add user message to history
        chat_history.append({"role": "user", "content": prompt})

    answer = None
    error_msg = None
    
    if prompt:
      try:
        # Deterministic fast-path for extracted metrics/ratios (most accurate for "Net Profit", "ROA", etc.).
        fast_match = _try_metric_fast_path(prompt, dual=dual, single=single, ratios=ratios)
        if fast_match:
          answer = _format_fast_answer(fast_match, units=units)
        else:
          # Otherwise use RAG (Chroma if available, else TF-IDF) + Hugging Face generation.
          excerpts = _retrieve_top_chunks(doc, prompt, top_k=4) if doc else []
          answer = _hf_answer(prompt, computed_context=(context or ""), excerpts=excerpts, chat_history=chat_history)

        if answer:
          chat_history.append({"role": "assistant", "content": answer})
      except Exception as e:
        error_msg = f"AI service error: {str(e)}"
    else:
        error_msg = "Please enter a question."

    # Keep only last 20 messages
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # Save updated chat history
    session["chat_history"] = chat_history

    return render_template_string(
        TEMPLATE, has_context=bool(context),
        dual=dual, single=single, ratios=ratios, recs=recs,
        currency=units.get("currency"), units_label=units.get("units_label"),
      chat_history=chat_history, upload_error=error_msg, ocr_available=check_ocr_dependencies()[0],
      hf_model_id=HF_MODEL_ID,
        prompt=None  # Don't prefill the form after submission
    )


@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("home"))


@app.route("/reset_all")
def reset_all():
    """Reset everything - clear all session data for fresh start"""
    try:
        _doc_store_delete(session.get("hdfc_doc_id"))
    except Exception:
        pass
    session.clear()
    return redirect(url_for("home"))


@app.route("/clear")
def clear():
    try:
        _doc_store_delete(session.get("hdfc_doc_id"))
    except Exception:
        pass

    for k in ["hdfc_context", "hdfc_units", "hdfc_dual", "hdfc_single", "hdfc_ratios", "hdfc_recs", "chat_history", "hdfc_doc_id"]:
        session.pop(k, None)
    return redirect(url_for("home"))


@app.route("/debug")
def debug():
    ocr_available, ocr_status = check_ocr_dependencies()
    units = session.get("hdfc_units") or {}
    doc = _doc_store_get(session.get("hdfc_doc_id"))
    return jsonify({
        "has_context": bool(session.get("hdfc_context")),
        "dual_keys": list((session.get("hdfc_dual") or {}).keys()),
        "single_keys": list((session.get("hdfc_single") or {}).keys()),
        "ratios": session.get("hdfc_ratios"),
        "recs": session.get("hdfc_recs"),
        "ocr_available": ocr_available,
        "ocr_status": ocr_status,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        "hf_model_id": HF_MODEL_ID,
        "use_vector_db": USE_VECTOR_DB,
        "doc_id": session.get("hdfc_doc_id"),
        "chroma_collection": doc.chroma_collection if doc else None,
        "units": units,
        "context_length": len(session.get("hdfc_context", "")),
        "session_keys": list(session.keys()),
        "chat_history_length": len(session.get("chat_history", []))
    })


@app.route("/test-pdf")
def test_pdf():
    """
    Quick test route for debugging a fixed local PDF path (update path below).
    """
    pdf_path = os.environ.get("HDFC_TEST_PDF", r"C:\Financial\sample_hdfc.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"Test PDF not found at: {pdf_path} (set HDFC_TEST_PDF env var)"}), 404

    try:
        # Native text
        with fitz.open(pdf_path) as doc:
            native_text = "\n".join(pg.get_text() for pg in doc)

        # OCR text (if available)
        ocr_available, ocr_status = check_ocr_dependencies()
        ocr_text = ""
        if ocr_available:
            try:
                ocr_text = ocr_pdf_to_text(pdf_path)
            except Exception as e:
                ocr_text = f"OCR Error: {e}"

        # Final chosen
        final_text = extract_text_with_ocr_fallback(pdf_path)
        extracted_text, dual_raw, single_raw, dual, single, units = parse_pdf(pdf_path)
        ratios = compute_ratios(dual_raw, single_raw)
        recs   = recommendations(ratios)

        return jsonify({
            "pdf_file": pdf_path,
            "native_text_length": len(native_text),
            "native_sample": native_text[:500],
            "ocr_available": ocr_available,
            "ocr_status": ocr_status,
            "ocr_text_length": len(ocr_text),
            "ocr_sample": ocr_text[:500] if ocr_text else "No OCR text",
            "final_text_length": len(final_text),
            "final_sample": final_text[:500],
            "parse_text_length": len(extracted_text),
            "units": units,
            "dual_extracted": list(dual.keys()),
            "single_extracted": list(single.keys()),
            "computed_ratios": ratios,
            "recommendations": recs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Main ----------
if __name__ == "__main__":
    ok, msg = check_ocr_dependencies()
    print(("✅ " if ok else "⚠️ ") + msg)
    print("🏦 HDFC Financial Analyzer with AI Assistant")
    print("🌐 Running on http://127.0.0.1:5077")
    print(f"🤖 Powered by Hugging Face: {HF_MODEL_ID}")
    app.run(host="127.0.0.1", port=5077, debug=True, use_reloader=False)
