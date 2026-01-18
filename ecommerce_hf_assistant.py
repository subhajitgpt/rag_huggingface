"""E-commerce assistant demo (Hugging Face + vector DB RAG).

What this gives you
- Product catalog + store policies as an in-memory knowledge base
- Dense embeddings + Chroma vector DB retrieval to fetch the best context
- Hugging Face text2text generation to answer using ONLY retrieved context

Run
  C:/ai_backed_python_coding/.venv/Scripts/python.exe c:/ai_backed_python_coding/ecommerce_hf_assistant.py

Optional env vars
  HF_MODEL_ID=google/flan-t5-base
  HF_MAX_NEW_TOKENS=256
    EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
    CHROMA_DIR=./.chroma_ecommerce
    RERANK_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
    USE_RERANKER=true|false
    RAG_CANDIDATES=12
    RAG_TOP_K=4

Notes
- This is a self-contained example that mirrors the pattern used in enbd_extraction.py
  (retrieval + generation), but for an e-commerce use case.
"""

from __future__ import annotations

import os
import sys
import re
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Prevent transformers from trying to import TF/Flax backends in noisy/broken environments.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

SentenceTransformer = None  # type: ignore
CrossEncoder = None  # type: ignore

from transformers import pipeline


HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma_ecommerce")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ecommerce")

RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "true").strip().lower() in {"1", "true", "yes"}
RAG_CANDIDATES = int(os.getenv("RAG_CANDIDATES", "12"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))


def _warn_if_not_using_venv() -> None:
    exe = (sys.executable or "").replace("\\", "/").lower()
    if "/.venv/" not in exe and "\\.venv\\" not in (sys.executable or ""):
        print(
            "[warn] You are not running the project venv. Use: "
            "C:/ai_backed_python_coding/.venv/Scripts/python.exe c:/ai_backed_python_coding/ecommerce_hf_assistant.py"
        )


# -----------------------------
# Sample e-commerce knowledge
# -----------------------------

PRODUCTS = [
    {
        "sku": "TSHIRT-001",
        "name": "Nimbus Cotton T‑Shirt",
        "category": "apparel",
        "price": 79.0,
        "currency": "AED",
        "colors": ["black", "white", "navy"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "highlights": ["100% cotton", "pre-shrunk", "breathable"],
        "shipping": "Ships in 1–2 business days",
        "returns": "30-day returns (unworn, tags attached)",
    },
    {
        "sku": "SNEAK-042",
        "name": "Orion Running Sneakers",
        "category": "footwear",
        "price": 299.0,
        "currency": "AED",
        "colors": ["grey", "black"],
        "sizes": ["40", "41", "42", "43", "44"],
        "highlights": ["lightweight", "arch support", "road running"],
        "shipping": "Ships in 2–3 business days",
        "returns": "14-day exchanges for size issues",
    },
    {
        "sku": "CABLE-USB-C-2M",
        "name": "2m USB‑C Fast Charge Cable",
        "category": "electronics",
        "price": 39.0,
        "currency": "AED",
        "colors": ["white"],
        "sizes": ["2m"],
        "highlights": ["60W PD", "braided", "USB‑IF certified"],
        "shipping": "Ships same day for orders before 2pm",
        "returns": "30-day returns",
    },
]

POLICIES = {
    "shipping": (
        "Shipping: UAE standard delivery is 1–3 business days. Same-day dispatch for orders placed before 2pm (Mon–Fri). "
        "Free shipping for orders above AED 200. Express delivery available in Dubai for AED 25."
    ),
    "returns": (
        "Returns: Most items have 30-day returns. Items must be unused, with original packaging and tags. "
        "Final-sale items are not eligible. Refunds are processed to the original payment method within 5–10 business days."
    ),
    "warranty": (
        "Warranty: Electronics include a 12-month warranty covering manufacturing defects. Physical damage and water damage are excluded."
    ),
    "payments": (
        "Payments: We accept Visa/Mastercard, Apple Pay, and cash on delivery (COD) for orders under AED 1,000."
    ),
}


def _product_to_doc(p: Dict[str, Any]) -> str:
    return (
        f"PRODUCT\n"
        f"SKU: {p['sku']}\n"
        f"Name: {p['name']}\n"
        f"Category: {p['category']}\n"
        f"Price: {p['currency']} {p['price']}\n"
        f"Colors: {', '.join(p.get('colors') or [])}\n"
        f"Sizes: {', '.join(p.get('sizes') or [])}\n"
        f"Highlights: {', '.join(p.get('highlights') or [])}\n"
        f"Shipping: {p.get('shipping','')}\n"
        f"Returns: {p.get('returns','')}\n"
    )


def _policies_to_docs(policies: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for k, v in policies.items():
        out.append(f"POLICY\nTopic: {k}\n{v}\n")
    return out


# -----------------------------
# Vector DB retrieval (Chroma)
# -----------------------------


@dataclass
class RetrievedDoc:
    text: str
    score: float
    metadata: Dict[str, Any]


class RetrievalBackend:
    """Small interface so we can swap vector DB retrieval vs fallback retrieval."""

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        raise NotImplementedError()


def _normalize_query(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_EMBEDDER: Any = None
_RERANKER: Any = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST  # type: ignore
                SentenceTransformer = _ST
            except Exception as e:
                raise RuntimeError(
                    "Failed to import sentence-transformers. This usually happens when you're not running the project venv or when transformers/sentence-transformers are mismatched. "
                    "Run with: C:/ai_backed_python_coding/.venv/Scripts/python.exe ecommerce_hf_assistant.py"
                ) from e
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_ID)
    return _EMBEDDER


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        global CrossEncoder
        if CrossEncoder is None:
            try:
                from sentence_transformers import CrossEncoder as _CE  # type: ignore
                CrossEncoder = _CE
            except Exception as e:
                raise RuntimeError(
                    "Failed to import CrossEncoder for reranking. Set USE_RERANKER=false or run with the project venv: C:/ai_backed_python_coding/.venv/Scripts/python.exe ecommerce_hf_assistant.py"
                ) from e
        # CrossEncoder outputs a relevance score for (query, doc) pairs.
        _RERANKER = CrossEncoder(RERANK_MODEL_ID)
    return _RERANKER


class ChromaBackend(RetrievalBackend):
    def __init__(self, collection: Any):
        self.collection = collection

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        return retrieve_top_docs(self.collection, query, top_k=top_k)


class TfidfBackend(RetrievalBackend):
    """Fallback retrieval when sentence-transformers cannot be imported.

    Uses scikit-learn TF-IDF entirely in-memory (no vector DB).
    """

    def __init__(self, docs: List[str], metas: List[Dict[str, Any]]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        self._docs = docs
        self._metas = metas
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self._matrix = self._vectorizer.fit_transform(docs)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        q = _normalize_query(query)
        if not q:
            return []
        qv = self._vectorizer.transform([q])
        sims = self._cosine_similarity(qv, self._matrix).ravel()
        if sims.size == 0:
            return []
        idx = sims.argsort()[::-1][: max(1, top_k)]
        out: List[RetrievedDoc] = []
        for i in idx:
            score = float(sims[int(i)])
            if score <= 0:
                continue
            out.append(RetrievedDoc(text=self._docs[int(i)], score=score, metadata=dict(self._metas[int(i)] or {})))
        return out


def build_backend() -> RetrievalBackend:
    """Prefer Chroma + SentenceTransformer; fall back to TF-IDF if embedder can't load."""

    # If sentence-transformers can't import in this interpreter, we still want the demo to run.
    try:
        _get_embedder()
    except Exception as e:
        print(f"[warn] Falling back to TF-IDF retrieval (no vector DB). Reason: {e}")
        docs, metas, _ids = build_docs()
        return TfidfBackend(docs=docs, metas=metas)

    rebuild = os.getenv("REBUILD_VECTOR_DB", "").strip().lower() in {"1", "true", "yes"}
    collection = build_vector_db(rebuild=rebuild)
    return ChromaBackend(collection)


def _get_chroma_collection(recreate: bool = False):
    """Create/load the Chroma collection.

    We keep this import inside the function so the script can still explain what
    to install if chromadb isn't available.
    """
    try:
        import chromadb
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: chromadb. Install with: pip install chromadb"
        ) from e

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if recreate:
        try:
            client.delete_collection(name=CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def _upsert_docs(collection, docs: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
    embedder = _get_embedder()
    # Normalize embeddings to make cosine similarity meaningful.
    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()

    # Chroma's add() fails on duplicate IDs; for this demo we delete+rebuild when asked.
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)


def build_vector_db(rebuild: bool = False) -> Any:
    """Build or load the vector DB (Chroma)."""
    collection = _get_chroma_collection(recreate=rebuild)

    # If not rebuilding and collection already has content, keep it.
    if not rebuild:
        try:
            existing = collection.count()
            if isinstance(existing, int) and existing > 0:
                return collection
        except Exception:
            pass

    docs, metas, ids = build_docs()
    _upsert_docs(collection, docs=docs, metadatas=metas, ids=ids)
    return collection


def retrieve_top_docs(collection: Any, query: str, top_k: int = 4) -> List[RetrievedDoc]:
    q = _normalize_query(query)
    if not q:
        return []

    embedder = _get_embedder()
    q_embedding = embedder.encode([q], normalize_embeddings=True)[0].tolist()

    res = collection.query(
        query_embeddings=[q_embedding],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[RetrievedDoc] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # For cosine distance: similarity ≈ 1 - distance
        try:
            score = float(1.0 - float(dist))
        except Exception:
            score = 0.0
        out.append(RetrievedDoc(text=str(doc), score=score, metadata=dict(meta or {})))
    return out


def rerank_docs(query: str, docs: List[RetrievedDoc], top_k: int = 4) -> List[RetrievedDoc]:
    """Rerank retrieved docs using a cross-encoder.

    This typically improves top-1 relevance vs pure embedding similarity.
    """
    if not docs:
        return []

    reranker = _get_reranker()
    pairs = [(query, d.text) for d in docs]
    scores = reranker.predict(pairs)

    # Ensure list[float]
    scores_list = [float(s) for s in scores]

    reranked: List[RetrievedDoc] = []
    for d, s in zip(docs, scores_list):
        meta = dict(d.metadata or {})
        meta["rerank_score"] = s
        reranked.append(RetrievedDoc(text=d.text, score=s, metadata=meta))

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[: max(1, top_k)]


# -----------------------------
# Metric-ish fast paths
# -----------------------------

def _best_fuzzy_choice(query: str, choices: List[str]) -> Tuple[Optional[str], float]:
    q = _normalize_query(query)
    if not q:
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


def try_fast_product_lookup(query: str) -> Optional[str]:
    """Direct lookups for common intents (price, sizes, colors) with typo tolerance."""

    product_names = [p["name"] for p in PRODUCTS]
    best_name, score = _best_fuzzy_choice(query, product_names)
    if not best_name or score < 0.6:
        return None

    product = next((p for p in PRODUCTS if p["name"] == best_name), None)
    if not product:
        return None

    q = _normalize_query(query)

    # Basic intent detection
    if any(w in q for w in ["price", "cost", "how much"]):
        return f"{product['name']} costs {product['currency']} {product['price']}."
    if any(w in q for w in ["size", "sizes", "fit"]):
        return f"Available sizes for {product['name']}: {', '.join(product.get('sizes') or [])}."
    if any(w in q for w in ["color", "colours", "colors"]):
        return f"Available colors for {product['name']}: {', '.join(product.get('colors') or [])}."

    return None


# -----------------------------
# Hugging Face answer
# -----------------------------

_HF_GEN = None


def get_hf_generator(model_id: str):
    global _HF_GEN
    if _HF_GEN is None:
        _HF_GEN = pipeline("text2text-generation", model=model_id)
    return _HF_GEN


def hf_answer(question: str, retrieved: List[RetrievedDoc]) -> str:
    gen = get_hf_generator(HF_MODEL_ID)

    system = (
        "You are a helpful e-commerce assistant. "
        "Use ONLY the provided CONTEXT to answer. "
        "If the answer isn't in CONTEXT, say you don't have enough info and suggest what to ask next. "
        "Keep answers short and specific."
    )

    context_blocks = "\n\n".join(
        f"[Doc {i+1} | score={d.score:.3f} | type={d.metadata.get('type')} | key={d.metadata.get('key')}]\n{d.text}"
        for i, d in enumerate(retrieved)
    )

    prompt = (
        f"{system}\n\n"
        f"CONTEXT:\n{context_blocks}\n\n"
        f"QUESTION:\n{question}\n"
    )

    out = gen(prompt, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
    if isinstance(out, list) and out:
        text = out[0].get("generated_text") or out[0].get("text")
        if text:
            return str(text).strip()
    return "I couldn't generate an answer right now."


# -----------------------------
# Demo CLI
# -----------------------------


def build_docs() -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for p in PRODUCTS:
        docs.append(_product_to_doc(p))
        metas.append({"type": "product", "key": p.get("sku"), "name": p.get("name"), "category": p.get("category")})
        ids.append(f"product::{p.get('sku')}")

    for topic, text in POLICIES.items():
        docs.append(f"POLICY\nTopic: {topic}\n{text}\n")
        metas.append({"type": "policy", "key": topic})
        ids.append(f"policy::{topic}")

    return docs, metas, ids


def chat_loop() -> None:
    print("🛒 E-commerce HF Assistant")
    print(f"🤖 Model: {HF_MODEL_ID}")
    print(f"🧠 Embeddings: {EMBEDDING_MODEL_ID}")
    print(f"🗄️  Vector DB: Chroma (dir={CHROMA_DIR}, collection={CHROMA_COLLECTION})")
    print(f"🧾 Reranker: {'ON' if USE_RERANKER else 'OFF'} ({RERANK_MODEL_ID})")
    print("Type 'q' to quit. Try typos like 'runnig snakers price' or 'return policcy'.")

    _warn_if_not_using_venv()
    backend = build_backend()

    while True:
        q = input("\nYou> ").strip()
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break

        # Fast-path for direct product attribute lookups
        fast = try_fast_product_lookup(q)
        if fast:
            print("Assistant>", fast)
            continue

        # Retrieve more candidates, then rerank to the final top_k.
        candidates = backend.retrieve(q, top_k=max(RAG_CANDIDATES, RAG_TOP_K))
        if USE_RERANKER:
            try:
                retrieved = rerank_docs(q, candidates, top_k=RAG_TOP_K)
            except Exception as e:
                print(f"[warn] Reranker disabled due to error: {e}")
                retrieved = candidates[: max(1, RAG_TOP_K)]
        else:
            retrieved = candidates[: max(1, RAG_TOP_K)]
        if not retrieved:
            print("Assistant> I couldn't find anything relevant in the catalog/policies. Try asking about a product name or shipping/returns.")
            continue

        ans = hf_answer(q, retrieved)
        print("Assistant>", ans)


if __name__ == "__main__":
    chat_loop()
