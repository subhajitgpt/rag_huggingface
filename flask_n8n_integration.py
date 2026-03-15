import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

# Keep your existing imports too:
# import fitz, re, sys
# from dotenv import load_dotenv
# from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

# n8n webhook URL
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://subhajit86.app.n8n.cloud/webhook/financial-statement-upload"
)

N8N_FILE_FIELD_NAME = os.getenv("N8N_FILE_FIELD_NAME", "pdf_file")

PDF_METRICS_MAX_PAGES = int(os.getenv("PDF_METRICS_MAX_PAGES", "20"))
PDF_PREVIEW_MAX_CHARS = int(os.getenv("PDF_PREVIEW_MAX_CHARS", "0"))

# Financial statement extraction (best-effort) from PDF text
FIN_METRICS_MAX_PAGES = int(os.getenv("FIN_METRICS_MAX_PAGES", "60"))
COMPANY_NAME_MAX_PAGES = int(os.getenv("COMPANY_NAME_MAX_PAGES", "2"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return float(s.replace(",", ""))
    except Exception:
        return None


def _to_ratio_from_percent_str(value: str | None) -> float | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None
    return _to_float(s)


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except Exception:
        return None


def _fmt_pct(x: float | None) -> str | None:
    if x is None:
        return None
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int) -> tuple[str, int] | tuple[None, int]:
    if not fitz:
        return None, 0
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None, 0

    try:
        page_count = int(getattr(doc, "page_count", 0) or 0)
        page_limit = page_count
        if max_pages > 0:
            page_limit = min(page_limit, max_pages)

        parts: list[str] = []
        for i in range(page_limit):
            try:
                page = doc.load_page(i)
                parts.append(page.get_text("text") or "")
            except Exception:
                continue
        return "\n".join(parts), page_limit
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _first_match_value(text: str, patterns: list[str], flags: int = re.IGNORECASE) -> str | None:
    for p in patterns:
        m = re.search(p, text, flags)
        if m and m.group(1):
            return m.group(1)
    return None


def extract_company_name(pdf_bytes: bytes) -> dict:
    """Best-effort extraction of company name from PDF metadata/first pages.

    Returns:
      {
        "company_name": str|None,
        "method": "metadata.title"|"metadata.author"|"text.heuristic"|"unavailable",
        "confidence": "high"|"medium"|"low"|"none",
        "candidates": [..]  # small list of top candidate lines (debug)
      }
    """

    result = {
        "company_name": None,
        "method": "unavailable",
        "confidence": "none",
        "candidates": [],
    }

    if not fitz:
        return result

    # 1) Metadata (fast + often correct when PDFs are well-authored)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return result

    try:
        md = getattr(doc, "metadata", None) or {}
        title = (md.get("title") or "").strip()
        author = (md.get("author") or "").strip()

        def _clean_meta(s: str) -> str:
            s = re.sub(r"\s+", " ", s).strip()
            # Avoid generic placeholders
            if not s or s.lower() in {"untitled", "unknown", "na", "n/a"}:
                return ""
            # Some generators put full report titles; keep first chunk.
            return s

        title_c = _clean_meta(title)
        if title_c and 4 <= len(title_c) <= 120:
            result.update(
                {
                    "company_name": title_c,
                    "method": "metadata.title",
                    "confidence": "medium",
                }
            )
            return result

        author_c = _clean_meta(author)
        if author_c and 4 <= len(author_c) <= 120:
            result.update(
                {
                    "company_name": author_c,
                    "method": "metadata.author",
                    "confidence": "low",
                }
            )
            # don't early-return; text may be better

        # 2) Heuristic from first pages text
        page_count = int(getattr(doc, "page_count", 0) or 0)
        page_limit = page_count
        if COMPANY_NAME_MAX_PAGES > 0:
            page_limit = min(page_limit, COMPANY_NAME_MAX_PAGES)

        raw_parts: list[str] = []
        for i in range(page_limit):
            try:
                page = doc.load_page(i)
                raw_parts.append(page.get_text("text") or "")
            except Exception:
                continue
        text = "\n".join(raw_parts)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if not text.strip():
        return result

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Common non-company header/footer noise
    noise_re = re.compile(
        r"\b(financial statements?|financial report|annual report|interim report|quarter|q[1-4]|\bfor the period\b|unaudited|audited|consolidated|statement of|notes to)\b",
        re.I,
    )

    def _score(line: str) -> int:
        if len(line) < 4 or len(line) > 80:
            return -10
        if noise_re.search(line):
            return -5
        if re.search(r"\d", line):
            return -3
        if line.count(" ") < 1:
            return -2
        letters = sum(1 for c in line if c.isalpha())
        uppers = sum(1 for c in line if c.isupper())
        # Prefer letter-heavy, title-ish lines near the top.
        return letters + (uppers // 2)

    scored = [(ln, _score(ln)) for ln in lines[:60]]
    scored.sort(key=lambda t: t[1], reverse=True)
    top = [ln for (ln, sc) in scored[:5] if sc > 0]
    result["candidates"] = top
    if not top:
        return result

    chosen = top[0]
    # If we had a metadata author guess, and it is contained in text, prefer author.
    if result.get("company_name") and result["company_name"].lower() in chosen.lower():
        result.update({"method": "metadata.author", "confidence": "low"})
        return result

    result.update(
        {
            "company_name": chosen,
            "method": "text.heuristic",
            "confidence": "medium" if len(chosen) >= 10 else "low",
        }
    )
    return result


def _first_match_dual(text: str, patterns: list[str], flags: int = re.IGNORECASE) -> dict:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return {
                "current": _to_float(m.group(1)) if m.lastindex and m.lastindex >= 1 else None,
                "prior": _to_float(m.group(2)) if m.lastindex and m.lastindex >= 2 else None,
                "pattern": p,
            }
    return {"current": None, "prior": None, "pattern": None}


def detect_units_light(text: str) -> dict:
    """Best-effort currency + magnitude detection (kept lightweight for webhook logging)."""
    if not text:
        return {"currency": None, "units_label": None}

    currency = None
    if re.search(r"\bAED\b|UAE\s*Dirhams?|Dhirhams?", text, re.I):
        currency = "AED"
    elif re.search(r"\bINR\b|\bRs\b|\bRupees\b|₹", text, re.I):
        currency = "INR"
    elif re.search(r"\bUSD\b|\bUS\s*\$\b|\$", text, re.I):
        currency = "USD"

    units_label = None
    if re.search(r"\bbillions?\b", text, re.I):
        units_label = "billions"
    elif re.search(r"\bmillions?\b", text, re.I):
        units_label = "millions"
    elif re.search(r"\bthousands?\b", text, re.I):
        units_label = "thousands"

    return {"currency": currency, "units_label": units_label}


def extract_financial_statement_details(pdf_bytes: bytes) -> dict:
    """Extract financial line-items + ratios (ROA/ROE etc.) from PDF text.

    This is a best-effort heuristic extraction meant for logging in n8n.
    It will work well for PDFs that follow common bank/financial-statement phrasing,
    and degrade gracefully (returns partial results) otherwise.
    """

    out: dict = {
        "extracted_at": _utc_now_iso(),
        "max_pages_scanned": FIN_METRICS_MAX_PAGES,
        "text_extraction_available": bool(fitz),
        "units": {"currency": None, "units_label": None},
        "income_statement": {},
        "balance_sheet": {},
        "reported_ratios": {},
        "computed_ratios": {},
        "extraction_quality": {},
    }

    text, sampled_pages = _extract_text_from_pdf_bytes(pdf_bytes, FIN_METRICS_MAX_PAGES)
    out["sampled_page_count"] = int(sampled_pages)
    if not text:
        out["text_extraction_available"] = False
        out["text_extraction_reason"] = "Unable to extract PDF text"
        return out

    # Normalize spacing for regex robustness
    norm = re.sub(r"\u00a0", " ", text)
    norm = re.sub(r"[ \t]+", " ", norm)

    out["units"] = detect_units_light(norm)

    NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"

    # ---- Income statement (current/prior when available) ----
    income_dual_defs: dict[str, list[str]] = {
        "total_operating_income": [
            rf"Total\s+operating\s+income\s+{NUM}\s+{NUM}",
            rf"Total\s+Income\s+{NUM}\s+{NUM}",
        ],
        "net_interest_income": [
            rf"Net\s+interest\s+income\s+and\s+net\s+income\s+from\s+Islamic[\s\S]*?{NUM}\s+{NUM}",
            rf"Net\s+interest\s+income\s+{NUM}\s+{NUM}",
        ],
        "operating_profit_before_impairment": [
            rf"Operating\s+profit\s+before\s+impairment\s+{NUM}\s+{NUM}",
        ],
        "profit_before_tax": [
            rf"Profit\s+for\s+the\s+period\s+before\s+taxation\s+{NUM}\s+{NUM}",
            rf"Profit\s+before\s+tax\s+{NUM}\s+{NUM}",
        ],
        "taxation": [
            rf"Taxation\s+charge\s+\(?{NUM}\)?\s+\(?{NUM}\)?",
            rf"Tax\s+Expense\s+\(?{NUM}\)?\s+\(?{NUM}\)?",
        ],
        "profit_for_period": [
            rf"Profit\s+for\s+the\s+period\s+{NUM}\s+{NUM}",
            rf"Net\s+Profit\s+.*?after\s+tax[\s\S]*?{NUM}\s+{NUM}",
        ],
        "eps": [
            rf"Earnings\s+per\s+share\s*\(.*?\)\s+{NUM}\s+{NUM}",
        ],
        "operating_expenses": [
            rf"General\s+and\s+administrative\s+expenses\s+\(?{NUM}\)?\s+\(?{NUM}\)?",
            rf"Operating\s+expenses\s+\(?{NUM}\)?\s+\(?{NUM}\)?",
        ],
    }

    income_statement: dict[str, dict] = {}
    for key, pats in income_dual_defs.items():
        income_statement[key] = _first_match_dual(norm, pats, flags=re.IGNORECASE | re.DOTALL)

    # ---- Balance sheet / key balances (single current best-effort) ----
    single_defs: dict[str, list[str]] = {
        "total_assets": [
            rf"Total\s+assets\s+{NUM}",
            rf"Total\s+{NUM}\s+\d",  # broad fallback seen in some PDFs
            rf"Segment\s+Assets[\s\S]*?({NUM})\s*\n\s*Segment\s+Liabilities",
        ],
        "gross_loans": [
            rf"Gross\s+loans\s+and\s+receivables\s+{NUM}",
            rf"Gross\s+Loans\s+{NUM}",
            rf"Advances\s+{NUM}",
        ],
        "deposits": [
            rf"Customer(?:\s+and)?\s+Islamic\s+deposits\s+{NUM}",
            rf"Deposits\s+{NUM}",
        ],
        "npls": [
            rf"Total\s+of\s+credit\s+impaired\s+loans\s+and\s+receivables\s+{NUM}",
            rf"Gross\s+NPAs\s+{NUM}",
        ],
        "ecl": [
            rf"Expected\s+credit\s+losses\s+\(?{NUM}\)?",
        ],
        "equity": [
            rf"Total\s+equity\s+{NUM}",
            rf"Net\s+worth\s+{NUM}",
            rf"Group\s+Total\s+({NUM})",
        ],
    }

    balance_sheet: dict[str, float | None] = {}
    for key, pats in single_defs.items():
        raw = _first_match_value(norm, pats, flags=re.IGNORECASE | re.DOTALL)
        balance_sheet[key] = _to_float(raw)

    # ---- Reported ratios (if present as % in the PDF) ----
    reported_roa = _first_match_value(
        norm,
        [r"Return\s+on\s+assets\s*\(.*?\).*?(\d+(?:\.\d+)?%)", r"\bROA\b\s*(\d+(?:\.\d+)?%)"],
        flags=re.IGNORECASE | re.DOTALL,
    )
    reported_roe = _first_match_value(
        norm,
        [r"Return\s+on\s+equity\s*\(.*?\).*?(\d+(?:\.\d+)?%)", r"\bROE\b\s*(\d+(?:\.\d+)?%)"],
        flags=re.IGNORECASE | re.DOTALL,
    )

    out["reported_ratios"] = {
        "roa": {"value": _to_ratio_from_percent_str(reported_roa), "as_percent": reported_roa},
        "roe": {"value": _to_ratio_from_percent_str(reported_roe), "as_percent": reported_roe},
    }

    # ---- Computed ratios (from extracted line items) ----
    toi = income_statement.get("total_operating_income", {}).get("current")
    pat = income_statement.get("profit_for_period", {}).get("current")
    ga = income_statement.get("operating_expenses", {}).get("current")
    opb = income_statement.get("operating_profit_before_impairment", {}).get("current")
    pbt = income_statement.get("profit_before_tax", {}).get("current")
    tax = income_statement.get("taxation", {}).get("current")
    assets = balance_sheet.get("total_assets")
    equity = balance_sheet.get("equity")
    gross_loans = balance_sheet.get("gross_loans")
    deposits = balance_sheet.get("deposits")
    npls = balance_sheet.get("npls")
    ecl = balance_sheet.get("ecl")

    computed = {
        "cost_to_income": _safe_div(ga, toi),
        "net_profit_margin": _safe_div(pat, toi),
        "pre_impairment_margin": _safe_div(opb, toi),
        "tax_rate": _safe_div(tax, pbt),
        "roa": _safe_div(pat, assets),
        "roe": _safe_div(pat, equity),
        "loan_to_deposit": _safe_div(gross_loans, deposits),
        "npl_ratio": _safe_div(npls, gross_loans),
        "coverage_ratio": _safe_div(ecl, npls),
        "ecl_to_gross_loans": _safe_div(ecl, gross_loans),
    }

    out["income_statement"] = income_statement
    out["balance_sheet"] = balance_sheet
    out["computed_ratios"] = {
        k: {"value": v, "as_percent": _fmt_pct(v)} for k, v in computed.items()
    }

    # Basic quality scoring
    found_line_items = sum(
        1
        for v in list(balance_sheet.values())
        if v is not None
    ) + sum(
        1
        for v in income_statement.values()
        if (v.get("current") is not None or v.get("prior") is not None)
    )
    out["extraction_quality"] = {
        "found_items_count": int(found_line_items),
        "has_income_statement": bool(toi is not None or pat is not None),
        "has_balance_sheet": bool(assets is not None),
    }

    return out


def extract_pdf_metrics(pdf_bytes: bytes) -> dict:
    """Extract lightweight, log-friendly metrics from a PDF.

    Notes:
    - Uses PyMuPDF (fitz) when available.
    - Text statistics are computed on at most `PDF_METRICS_MAX_PAGES` pages (0 = all pages).
    - Optional preview text (disabled by default) can be enabled via `PDF_PREVIEW_MAX_CHARS`.
    """

    metrics: dict = {
        "ingested_at": _utc_now_iso(),
        "file_size_bytes": len(pdf_bytes),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "pdf_signature_ok": bool(pdf_bytes.startswith(b"%PDF")),
        "metrics_sample_max_pages": PDF_METRICS_MAX_PAGES,
        "preview_max_chars": PDF_PREVIEW_MAX_CHARS,
    }

    if not fitz:
        metrics["text_extraction_available"] = False
        metrics["text_extraction_reason"] = "PyMuPDF (fitz) not available"
        return metrics

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        metrics["parse_error"] = str(e)
        return metrics

    try:
        metrics["page_count"] = int(getattr(doc, "page_count", 0) or 0)
        md = getattr(doc, "metadata", None) or {}
        metrics["metadata"] = {k: v for k, v in md.items() if v not in (None, "", "unknown")}

        page_limit = metrics["page_count"]
        if PDF_METRICS_MAX_PAGES > 0:
            page_limit = min(page_limit, PDF_METRICS_MAX_PAGES)
        metrics["sampled_page_count"] = int(page_limit)

        word_re = re.compile(r"\b\w+\b", re.UNICODE)
        number_re = re.compile(r"(?<!\w)[-+]?\d[\d,]*(?:\.\d+)?(?!\w)")
        email_re = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
        url_re = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)
        currency_re = re.compile(r"(?:\bAED\b|\bUSD\b|\bINR\b|₹|\$|€)")

        chars = 0
        words = 0
        nonempty_lines = 0
        numbers = 0
        emails = 0
        urls = 0
        currency_mentions = 0
        images = 0

        preview_text_parts: list[str] = []
        remaining_preview = max(PDF_PREVIEW_MAX_CHARS, 0)

        for i in range(page_limit):
            page = doc.load_page(i)
            try:
                images += len(page.get_images(full=True))
            except Exception:
                pass

            page_text = page.get_text("text") or ""

            chars += len(page_text)
            words += len(word_re.findall(page_text))
            nonempty_lines += sum(1 for ln in page_text.splitlines() if ln.strip())
            numbers += len(number_re.findall(page_text))
            emails += len(email_re.findall(page_text))
            urls += len(url_re.findall(page_text))
            currency_mentions += len(currency_re.findall(page_text))

            if remaining_preview > 0 and page_text:
                take = page_text[:remaining_preview]
                preview_text_parts.append(take)
                remaining_preview -= len(take)

        metrics["text_stats"] = {
            "chars": int(chars),
            "word_count": int(words),
            "nonempty_line_count": int(nonempty_lines),
            "number_token_count": int(numbers),
            "email_count": int(emails),
            "url_count": int(urls),
            "currency_mention_count": int(currency_mentions),
        }
        metrics["image_count_in_sampled_pages"] = int(images)

        if PDF_PREVIEW_MAX_CHARS > 0:
            metrics["preview_text"] = "".join(preview_text_parts).strip()

        return metrics

    finally:
        try:
            doc.close()
        except Exception:
            pass


INDEX_HTML = """\
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>PDF → n8n Webhook</title>
        <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
            .card { max-width: 760px; margin: 0 auto; padding: 18px 18px 10px; border: 1px solid #e5e7eb; border-radius: 12px; }
            label { display: block; font-weight: 600; margin-top: 12px; }
            input[type=text], input[type=file] { width: 100%; margin-top: 6px; padding: 10px; border: 1px solid #d1d5db; border-radius: 10px; }
            .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            button { margin-top: 14px; padding: 10px 14px; border: 0; border-radius: 10px; background: #111827; color: white; cursor: pointer; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            pre { white-space: pre-wrap; word-wrap: break-word; background: #0b1020; color: #e5e7eb; padding: 14px; border-radius: 12px; margin-top: 14px; }
            .muted { color: #6b7280; font-size: 0.95rem; }
            .ok { color: #065f46; }
            .err { color: #991b1b; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2 style="margin:0 0 6px">Upload a PDF and trigger n8n</h2>
            <div class="muted">On upload, this app calls your n8n webhook once and shows the response.</div>

            <form id="uploadForm" enctype="multipart/form-data">
                <div class="row">
                    <div>
                        <label for="company_name">Company</label>
                        <input id="company_name" name="company_name" type="text" value="" placeholder="(auto-detected from PDF if left blank)" />
                    </div>
                    <div>
                        <label for="quarter">Quarter</label>
                        <input id="quarter" name="quarter" type="text" value="Q1 2025" />
                    </div>
                </div>

                <label for="industry">Industry</label>
                <input id="industry" name="industry" type="text" value="Banking" />

                <label for="pdf_file">PDF</label>
                <input id="pdf_file" name="pdf_file" type="file" accept="application/pdf,.pdf" />

                <button id="submitBtn" type="submit">Upload</button>
            </form>

            <div id="status" class="muted" style="margin-top:10px"></div>
            <pre id="out" style="display:none"></pre>
        </div>

        <script>
            const form = document.getElementById('uploadForm');
            const fileInput = document.getElementById('pdf_file');
            const btn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            const out = document.getElementById('out');

            async function doUpload() {
                const file = fileInput.files && fileInput.files[0];
                if (!file) {
                    status.textContent = 'Please choose a PDF.';
                    status.className = 'err';
                    return;
                }
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    status.textContent = 'Only PDF files are allowed.';
                    status.className = 'err';
                    return;
                }

                btn.disabled = true;
                status.textContent = 'Uploading… (this will call n8n)';
                status.className = 'muted';
                out.style.display = 'none';

                const fd = new FormData(form);

                try {
                    const resp = await fetch('/upload', {
                        method: 'POST',
                        body: fd,
                        headers: { 'Accept': 'application/json' }
                    });
                    const data = await resp.json().catch(() => ({ ok: false, message: 'Non-JSON response from server' }));
                    if (!resp.ok) {
                        status.textContent = data && data.message ? data.message : ('Upload failed (HTTP ' + resp.status + ')');
                        status.className = 'err';
                    } else {
                        status.textContent = data && data.message ? data.message : 'Done.';
                        status.className = (data && data.ok) ? 'ok' : 'err';
                    }
                    out.textContent = JSON.stringify(data, null, 2);
                    out.style.display = 'block';
                } catch (e) {
                    status.textContent = 'Network error: ' + (e && e.message ? e.message : String(e));
                    status.className = 'err';
                } finally {
                    btn.disabled = false;
                }
            }

            form.addEventListener('submit', (e) => {
                e.preventDefault();
                doUpload();
            });
        </script>
    </body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

def send_pdf_to_n8n(file_storage, company_name: str | None = None, quarter: str | None = None, industry: str | None = None):
    """
    Sends the uploaded PDF from Flask to n8n using multipart/form-data.
    Accepts the original Werkzeug FileStorage object from request.files.
    """
    if not file_storage or not file_storage.filename:
        return {"ok": False, "message": "No file provided"}

    if not N8N_WEBHOOK_URL or "YOUR-" in N8N_WEBHOOK_URL:
        return {"ok": False, "message": "N8N_WEBHOOK_URL is not configured."}

    # Reset stream to start before reading
    file_storage.stream.seek(0)
    pdf_bytes = file_storage.read()
    filename = secure_filename(file_storage.filename)
    pdf_metrics = extract_pdf_metrics(pdf_bytes)
    company_guess = extract_company_name(pdf_bytes)
    final_company_name = (company_name or "").strip() or (company_guess.get("company_name") or "Unknown")
    return send_pdf_bytes_to_n8n(
        pdf_bytes=pdf_bytes,
        filename=filename,
        company_name=final_company_name,
        quarter=(quarter or "").strip() or "",
        industry=(industry or "").strip() or "",
        pdf_metrics=pdf_metrics,
        financial_details=None,
        company_name_extraction=company_guess,
    )


def send_pdf_bytes_to_n8n(
    *,
    pdf_bytes: bytes,
    filename: str,
    company_name: str,
    quarter: str,
    industry: str,
    pdf_metrics: dict | None = None,
    financial_details: dict | None = None,
    company_name_extraction: dict | None = None,
):
    """Send PDF bytes + optional metrics to n8n using multipart/form-data."""

    if not pdf_bytes:
        return {"ok": False, "message": "No PDF bytes provided"}

    if not N8N_WEBHOOK_URL or "YOUR-" in N8N_WEBHOOK_URL:
        return {"ok": False, "message": "N8N_WEBHOOK_URL is not configured."}

    # Basic PDF signature check (prevents accidental non-PDF uploads)
    if not pdf_bytes.startswith(b"%PDF"):
        return {"ok": False, "message": "Uploaded file does not look like a valid PDF."}

    files = {
        N8N_FILE_FIELD_NAME: (secure_filename(filename or "upload.pdf"), pdf_bytes, "application/pdf")
    }

    data = {
        "company_name": company_name,
        "quarter": quarter,
        "industry": industry,
    }

    if pdf_metrics is not None:
        data.update(
            {
                # Full metrics blob for logging in n8n
                "pdf_metrics_json": json.dumps(pdf_metrics, ensure_ascii=False),
                # A few flattened fields for easy filtering in n8n
                "pdf_sha256": pdf_metrics.get("sha256", ""),
                "pdf_size_bytes": str(pdf_metrics.get("file_size_bytes", "")),
                "pdf_page_count": str(pdf_metrics.get("page_count", "")),
                "pdf_word_count_sampled": str((pdf_metrics.get("text_stats") or {}).get("word_count", "")),
            }
        )

    if financial_details is not None:
        # Full details blob (JSON) plus a handful of flattened key numbers.
        best_income = financial_details.get("income_statement") or {}
        best_bs = financial_details.get("balance_sheet") or {}
        best_ratios = financial_details.get("computed_ratios") or {}
        data.update(
            {
                "financial_details_json": json.dumps(financial_details, ensure_ascii=False),
                "income_total_operating_income": str((best_income.get("total_operating_income") or {}).get("current") or ""),
                "income_profit_for_period": str((best_income.get("profit_for_period") or {}).get("current") or ""),
                "bs_total_assets": str(best_bs.get("total_assets") or ""),
                "ratio_roa": str(((best_ratios.get("roa") or {}).get("value")) or ""),
                "ratio_roe": str(((best_ratios.get("roe") or {}).get("value")) or ""),
                "ratio_cost_to_income": str(((best_ratios.get("cost_to_income") or {}).get("value")) or ""),
                "ratio_npl_ratio": str(((best_ratios.get("npl_ratio") or {}).get("value")) or ""),
                "ratio_coverage_ratio": str(((best_ratios.get("coverage_ratio") or {}).get("value")) or ""),
                "ratio_loan_to_deposit": str(((best_ratios.get("loan_to_deposit") or {}).get("value")) or ""),
            }
        )

    if company_name_extraction is not None:
        data.update(
            {
                "company_name_extraction_json": json.dumps(company_name_extraction, ensure_ascii=False),
                "company_name_extraction_method": str(company_name_extraction.get("method") or ""),
                "company_name_extraction_confidence": str(company_name_extraction.get("confidence") or ""),
            }
        )

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()

        # Try JSON first; fallback to text
        try:
            return response.json()
        except ValueError:
            return {
                "ok": True,
                "message": "n8n responded with non-JSON payload",
                "raw_response": response.text
            }

    except requests.exceptions.RequestException as e:
        return {
            "ok": False,
            "message": f"Failed to invoke n8n webhook: {str(e)}"
        }

# -----------------------------
# Example upload route patch
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf_file")
    if not f or f.filename == "":
        return jsonify({"ok": False, "message": "Please select a PDF file."}), 400

    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "message": "Please upload a PDF file only."}), 400

    # Read the PDF once, compute metrics once, then call n8n
    f.stream.seek(0)
    pdf_bytes = f.read()
    if not pdf_bytes.startswith(b"%PDF"):
        return jsonify({"ok": False, "message": "Uploaded file does not look like a valid PDF."}), 400

    pdf_metrics = extract_pdf_metrics(pdf_bytes)
    financial_details = extract_financial_statement_details(pdf_bytes)
    company_guess = extract_company_name(pdf_bytes)
    company_name_from_form = (request.form.get("company_name") or "").strip()
    company_name_final = company_name_from_form or (company_guess.get("company_name") or "Unknown")

    # 1) Send PDF+metrics to n8n once
    n8n_result = send_pdf_bytes_to_n8n(
        pdf_bytes=pdf_bytes,
        filename=f.filename,
        company_name=company_name_final,
        quarter=(request.form.get("quarter") or "").strip() or "",
        industry=(request.form.get("industry") or "").strip() or "",
        pdf_metrics=pdf_metrics,
        financial_details=financial_details,
        company_name_extraction=company_guess,
    )

    # 2) Continue your existing local parsing logic (optional)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        with open(tmp.name, "wb") as out_f:
            out_f.write(pdf_bytes)

        # Use your existing function:
        # dual_raw, single_raw, dual, single, units = parse_pdf(tmp.name)
        # ratios = compute_ratios(dual_raw, single_raw)
        # recs = recommendations(ratios)

        return jsonify({
            "ok": True,
            "message": "PDF uploaded and n8n webhook invoked.",
            "filename": secure_filename(f.filename),
            "n8n_result": n8n_result,
            "pdf_metrics": pdf_metrics,
            "financial_details": financial_details,
            "company_name": company_name_final,
            "company_name_extraction": company_guess,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Error processing PDF: {str(e)}",
            "n8n_result": n8n_result,
        }), 500

    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass

# -----------------------------
# Optional standalone caller
# -----------------------------
def invoke_n8n_from_path(pdf_path, company_name: str | None = None, quarter: str | None = None, industry: str | None = None):
    """
    Useful outside Flask if you want to test the webhook directly from Python.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_metrics = extract_pdf_metrics(pdf_bytes)
    financial_details = extract_financial_statement_details(pdf_bytes)
    company_guess = extract_company_name(pdf_bytes)
    company_name_final = (company_name or "").strip() or (company_guess.get("company_name") or "Unknown")
    return send_pdf_bytes_to_n8n(
        pdf_bytes=pdf_bytes,
        filename=os.path.basename(pdf_path),
        company_name=company_name_final,
        quarter=(quarter or "").strip() or "",
        industry=(industry or "").strip() or "",
        pdf_metrics=pdf_metrics,
        financial_details=financial_details,
        company_name_extraction=company_guess,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5093"))
    app.run(host="127.0.0.1", port=port, debug=True)
