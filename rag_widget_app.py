from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

from flask import Flask, render_template, request


def _safe_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
        ) from e


def _safe_import_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: transformers. Install with: pip install transformers torch"
        ) from e


def _chunk_text(text: str, *, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Simple chunker for demo purposes.

    - Splits on whitespace to keep it robust across input types.
    - Creates overlapping chunks to avoid losing context at boundaries.

    For production, you’d likely chunk by sentence/token count and preserve metadata.
    """

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split(" ") if cleaned else []


def _cosine_sim_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for w in a.values():
        norm_a += w * w
    for w in b.values():
        norm_b += w * w

    # iterate smaller dict for dot
    if len(a) <= len(b):
        small, large = a, b
    else:
        small, large = b, a

    for tok, w in small.items():
        dot += w * large.get(tok, 0.0)

    denom = (math.sqrt(norm_a) * math.sqrt(norm_b)) + 1e-12
    return dot / denom


def _retrieve_lite(chunks: list[str], question: str, *, top_k: int) -> list[RetrievedChunk]:
    """Dependency-free retrieval (TF-IDF lite).

    This is used when HF dependencies are missing OR when the user disables HF embeddings.
    """

    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    docs_tokens = [_tokenize(c) for c in chunks]
    n_docs = max(1, len(docs_tokens))

    df: Counter[str] = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    def tfidf(tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = float(len(tokens))
        vec: dict[str, float] = {}
        for tok, count in tf.items():
            # smooth idf
            idf = math.log((n_docs + 1.0) / (df.get(tok, 0) + 1.0)) + 1.0
            vec[tok] = (count / total) * idf
        return vec

    q_vec = tfidf(q_tokens)
    scored: list[RetrievedChunk] = []
    for chunk, toks in zip(chunks, docs_tokens, strict=False):
        score = _cosine_sim_sparse(q_vec, tfidf(toks))
        scored.append(RetrievedChunk(score=float(score), chunk=chunk))

    scored.sort(key=lambda r: r.score, reverse=True)
    top_k = max(1, min(int(top_k), len(scored)))
    return scored[:top_k]


def _parse_colon_sections(docs_text: str) -> list[tuple[str, str]]:
    """Parse simple 'Title:' sections from raw text.

    Example:
        Refund policy:
        ...

        Shipping policy:
        ...

    Returns a list of (title, body) pairs in order.
    """

    lines = (docs_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    def flush():
        nonlocal current_title, current_body
        if current_title is None:
            return
        body = " ".join([re.sub(r"\s+", " ", x).strip() for x in current_body if x.strip()]).strip()
        sections.append((current_title, body))
        current_title = None
        current_body = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Heuristic: a short line ending with ':' is a heading
        if line.endswith(":") and 1 <= len(line) <= 80:
            flush()
            current_title = line[:-1].strip()
            continue

        if current_title is None:
            # Ignore preamble text if no heading found yet
            continue
        current_body.append(line)

    flush()
    return [(t, b) for (t, b) in sections if t and b]


def _pick_best_section(sections: list[tuple[str, str]], question: str) -> tuple[str, str] | None:
    if not sections:
        return None

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return None

    best: tuple[str, str] | None = None
    best_score = 0

    for title, body in sections:
        title_tokens = set(_tokenize(title))
        body_tokens = set(_tokenize(body))

        # Bias toward title matches (refund/shipping/security)
        score = 3 * len(q_tokens & title_tokens) + len(q_tokens & body_tokens)

        if score > best_score:
            best_score = score
            best = (title, body)

    # Require at least one overlapping token to avoid random picks
    return best if best_score > 0 else None


def _cosine_sim_matrix(query_vec, doc_matrix):
    """Compute cosine similarities.

    Uses numpy only (no sklearn dependency required here).
    """

    import numpy as np

    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)
    return d @ q


@dataclass
class RetrievedChunk:
    score: float
    chunk: str


class RagEngine:
    def __init__(
        self,
        *,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model_id: str = "google/flan-t5-base",
    ) -> None:
        self.embed_model_id = embed_model_id
        self.gen_model_id = gen_model_id

        self._embedder = None
        self._generator = None

        self._chunks: list[str] = []
        self._chunk_vectors = None

    def set_corpus(self, docs_text: str) -> None:
        chunks = _chunk_text(docs_text)
        if not chunks:
            raise ValueError("No documents provided. Paste some text in the Documents box.")

        self._chunks = chunks
        self._chunk_vectors = None  # rebuild lazily

    def _get_embedder(self):
        if self._embedder is None:
            SentenceTransformer = _safe_import_sentence_transformers()
            self._embedder = SentenceTransformer(self.embed_model_id)
        return self._embedder

    def _ensure_index(self):
        if self._chunk_vectors is not None:
            return

        embedder = self._get_embedder()
        vectors = embedder.encode(self._chunks, normalize_embeddings=False)

        import numpy as np

        self._chunk_vectors = np.asarray(vectors, dtype="float32")

    def retrieve(self, question: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        if not question.strip():
            return []

        self._ensure_index()
        embedder = self._get_embedder()

        q_vec = embedder.encode([question], normalize_embeddings=False)[0]
        sims = _cosine_sim_matrix(q_vec, self._chunk_vectors)

        import numpy as np

        top_k = max(1, min(int(top_k), len(self._chunks)))
        top_idx = np.argsort(-sims)[:top_k]

        return [
            RetrievedChunk(score=float(sims[i]), chunk=self._chunks[i])
            for i in top_idx
        ]

    def build_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            [f"Context {i+1} (score={r.score:.3f}):\n{r.chunk}" for i, r in enumerate(retrieved)]
        )

        return (
            "You are a helpful assistant. Answer the user's question using ONLY the context below.\n"
            "If the answer is not present, say: 'I don't know based on the provided context.'\n\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _get_generator(self):
        if self._generator is None:
            pipeline = _safe_import_transformers_pipeline()
            # text2text-generation is best for FLAN/T5 style instruction models
            self._generator = pipeline(
                task="text2text-generation",
                model=self.gen_model_id,
            )
        return self._generator

    def generate_answer(self, prompt: str) -> str:
        generator = self._get_generator()
        out = generator(prompt, max_new_tokens=200, do_sample=False)[0]
        # transformers pipeline output differs slightly by task; this key is consistent here
        return (out.get("generated_text") or "").strip()


SAMPLE_DOCS = """
Refund policy:
Refunds are available within 30 days of purchase with a valid receipt. Digital goods are non-refundable once downloaded.

Shipping policy:
Standard shipping takes 3-5 business days. Expedited shipping takes 1-2 business days. International shipping may take 10-15 days.

Account security:
Enable two-factor authentication (2FA) to secure your account. Passwords must be at least 12 characters. Do not reuse passwords.

Troubleshooting login:
If you cannot log in, reset your password. Check for caps lock. If 2FA fails, ensure your phone time is correct or use backup codes.
""".strip()


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def rag_widget() -> str:
    # UI-only Flask: no JSON APIs, no fetch().
    # The form posts back to '/', and we render the answer server-side.

    question = ""
    docs = SAMPLE_DOCS
    top_k = 4
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model = "google/flan-t5-base"
    use_hf_embeddings = False
    use_generator = True

    answer: str | None = None
    prompt: str | None = None
    retrieved: list[RetrievedChunk] = []
    elapsed_ms: int | None = None
    error: str | None = None

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        docs = (request.form.get("docs") or "").strip() or SAMPLE_DOCS
        embed_model = (request.form.get("embed_model") or embed_model).strip() or embed_model
        gen_model = (request.form.get("gen_model") or gen_model).strip() or gen_model
        use_hf_embeddings = bool(request.form.get("use_hf_embeddings"))
        # Default to LLM answers. The checkbox remains in the UI, but we keep LLM on by default.
        use_generator = True

        try:
            top_k = int(request.form.get("top_k") or top_k)
        except ValueError:
            top_k = 4

        if not question:
            error = "Please enter a question."
        else:
            try:
                t0 = time.perf_counter()

                # Prefer section-aware answers when docs have explicit headings.
                sections = _parse_colon_sections(docs)

                chunks = _chunk_text(docs)
                if not chunks:
                    raise ValueError("No documents provided. Paste some text in the Documents box.")

                # Retrieval: HF embeddings (optional) OR fallback TF-IDF lite
                if use_hf_embeddings:
                    engine = RagEngine(embed_model_id=embed_model, gen_model_id=gen_model)
                    engine.set_corpus(docs)
                    retrieved = engine.retrieve(question, top_k=top_k)
                    prompt = engine.build_prompt(question, retrieved)
                else:
                    retrieved = _retrieve_lite(chunks, question, top_k=top_k)
                    prompt = RagEngine(gen_model_id=gen_model).build_prompt(question, retrieved)

                if use_generator:
                    try:
                        engine_for_gen = RagEngine(embed_model_id=embed_model, gen_model_id=gen_model)
                        answer = engine_for_gen.generate_answer(prompt)
                    except Exception as gen_e:
                        # Never crash the request just because generation failed.
                        answer = "\n\n".join(
                            [
                                "Retrieval-only mode (LLM generation failed):",
                                "Here are the most relevant context chunks:",
                                "",
                                *[f"- (score={r.score:.3f}) {r.chunk}" for r in retrieved],
                            ]
                        )
                        error = f"Generation failed: {type(gen_e).__name__}: {gen_e}"
                else:
                    picked = _pick_best_section(sections, question)
                    if picked is not None:
                        title, body = picked
                        answer = "\n".join(
                            [
                                "Retrieval-only mode (no LLM):",
                                f"Selected: {title}",
                                "",
                                body,
                            ]
                        )
                    else:
                        answer = "\n\n".join(
                            [
                                "Retrieval-only mode (no LLM):",
                                "Here are the most relevant context chunks:",
                                "",
                                *[f"- (score={r.score:.3f}) {r.chunk}" for r in retrieved],
                            ]
                        )

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

    return render_template(
        "rag_widget.html",
        # inputs
        sample_docs=SAMPLE_DOCS,
        docs=docs,
        question=question,
        top_k=top_k,
        embed_model=embed_model,
        gen_model=gen_model,
        use_hf_embeddings=use_hf_embeddings,
        use_generator=use_generator,
        # outputs
        answer=answer,
        prompt=prompt,
        retrieved=retrieved,
        elapsed_ms=elapsed_ms,
        error=error,
    )


if __name__ == "__main__":
    # Run: python rag_widget_app.py
    # Then open: http://127.0.0.1:5055
    port = int(os.getenv("PORT", "5055"))
    app.run(host="127.0.0.1", port=port, debug=True)
