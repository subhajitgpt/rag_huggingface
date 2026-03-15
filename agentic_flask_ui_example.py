from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, redirect, render_template_string, request, session, url_for

app = Flask(__name__)
app.secret_key = "dev-secret-change-in-production"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False


SAMPLE_DOC = (
    "Emirates NBD (sample excerpt) — For demonstration only.\n"
    "Total operating income: 10,000\n"
    "General and administrative expenses: (4,200)\n"
    "Customer deposits increased year-on-year due to retail growth.\n"
    "Liquidity remained strong with a diversified funding base.\n"
)


@dataclass
class Step:
    name: str
    tool: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    took_ms: int


# ---------------------- "Tools" (functions the agent can call) ----------------------

def tool_classify_intent(goal: str) -> Dict[str, Any]:
    g = (goal or "").strip().lower()

    # Very small heuristic classifier for demo purposes.
    if re.search(r"\b(cost\s*to\s*income|cost-to-income|cti)\b", g):
        return {"intent": "ratio", "ratio": "cost_to_income"}
    if re.search(r"\b(ldr|loan\s*to\s*deposit|loan-to-deposit)\b", g):
        return {"intent": "ratio", "ratio": "loan_to_deposit"}
    if re.search(r"\b(find|search|locate)\b", g):
        return {"intent": "search"}
    if re.search(r"\b(summarize|summary|tldr)\b", g):
        return {"intent": "summary"}

    return {"intent": "clarify"}


def tool_extract_numbers(goal: str) -> Dict[str, Any]:
    """Extract labeled numbers like income=10000, expenses=4200, loans=..., deposits=...."""
    g = goal or ""

    def find_after(words: List[str]) -> Optional[float]:
        for w in words:
            m = re.search(rf"\b{re.escape(w)}\b\s*[:=]?\s*\(?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*\)?", g, re.I)
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    income = find_after(["income", "operating income", "toi"])  # total operating income
    expenses = find_after(["expenses", "opex", "g&a", "general and administrative expenses", "ga"])  # op exp
    loans = find_after(["loans", "gross loans"])  # for LDR
    deposits = find_after(["deposits", "customer deposits"])  # for LDR

    return {
        "income": income,
        "expenses": expenses,
        "loans": loans,
        "deposits": deposits,
    }


def tool_retrieve_evidence(query: str, doc_text: str, window: int = 220) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"hit": False, "excerpt": None}

    m = re.search(re.escape(q), doc_text, re.I)
    if not m:
        return {"hit": False, "excerpt": None}

    start = max(0, m.start() - window)
    end = min(len(doc_text), m.end() + window)
    excerpt = doc_text[start:end].strip()
    return {"hit": True, "excerpt": excerpt}


def tool_compute_ratio(a: Optional[float], b: Optional[float]) -> Dict[str, Any]:
    if a is None or b is None or b == 0:
        return {"ok": False, "value": None, "reason": "Missing or invalid inputs"}
    return {"ok": True, "value": round(a / b, 4)}


def tool_verify_answer(answer: str, must_include: List[str]) -> Dict[str, Any]:
    a = (answer or "").lower()
    missing = [s for s in must_include if s.lower() not in a]
    return {"ok": not missing, "missing": missing}


# ---------------------- Agent loop ----------------------

def _run_step(name: str, tool: str, fn, tool_input: Dict[str, Any]) -> Tuple[Step, Dict[str, Any]]:
    t0 = time.perf_counter()
    out = fn(**tool_input)
    took_ms = int((time.perf_counter() - t0) * 1000)
    step = Step(name=name, tool=tool, input=tool_input, output=out, took_ms=took_ms)
    return step, out


def run_agent(goal: str) -> Tuple[List[Step], str]:
    steps: List[Step] = []

    s1, intent = _run_step(
        name="Classify user goal",
        tool="tool_classify_intent",
        fn=tool_classify_intent,
        tool_input={"goal": goal},
    )
    steps.append(s1)

    if intent.get("intent") == "ratio":
        s2, nums = _run_step(
            name="Extract numbers from prompt",
            tool="tool_extract_numbers",
            fn=tool_extract_numbers,
            tool_input={"goal": goal},
        )
        steps.append(s2)

        ratio_name = intent.get("ratio")
        if ratio_name == "cost_to_income":
            income = nums.get("income")
            expenses = nums.get("expenses")
            if income is None or expenses is None:
                return steps, (
                    "I can compute Cost-to-Income if you provide numbers like: "
                    "income=10000 and expenses=4200."
                )

            # CTI = expenses / income
            s3, res = _run_step(
                name="Compute ratio",
                tool="tool_compute_ratio",
                fn=tool_compute_ratio,
                tool_input={"a": float(expenses), "b": float(income)},
            )
            steps.append(s3)

            if not res.get("ok"):
                return steps, "I couldn’t compute the ratio due to missing/invalid inputs."

            pct = float(res["value"]) * 100
            answer = f"Cost-to-Income = {expenses:,.0f} / {income:,.0f} = {pct:.2f}% (CTI)."

            s4, ver = _run_step(
                name="Verify answer contains key elements",
                tool="tool_verify_answer",
                fn=tool_verify_answer,
                tool_input={"answer": answer, "must_include": ["Cost-to-Income", "%"]},
            )
            steps.append(s4)

            if not ver.get("ok"):
                answer += " (Verification warning: answer may be incomplete.)"
            return steps, answer

        if ratio_name == "loan_to_deposit":
            loans = nums.get("loans")
            deposits = nums.get("deposits")
            if loans is None or deposits is None:
                return steps, (
                    "I can compute Loan-to-Deposit if you provide numbers like: "
                    "loans=75000 and deposits=90000."
                )

            # LDR = loans / deposits
            s3, res = _run_step(
                name="Compute ratio",
                tool="tool_compute_ratio",
                fn=tool_compute_ratio,
                tool_input={"a": float(loans), "b": float(deposits)},
            )
            steps.append(s3)

            if not res.get("ok"):
                return steps, "I couldn’t compute the ratio due to missing/invalid inputs."

            pct = float(res["value"]) * 100
            return steps, f"Loan-to-Deposit (LDR) = {loans:,.0f} / {deposits:,.0f} = {pct:.2f}%."

        return steps, "I recognized a ratio request but don’t support that ratio yet."

    if intent.get("intent") == "search":
        # Try to infer query after the word 'find'/'search'/'locate'
        m = re.search(r"\b(?:find|search|locate)\b\s+(.*)$", goal or "", re.I)
        query = (m.group(1).strip() if m else "").strip("\"' ")
        if not query:
            return steps, "What keyword or phrase should I search for?"

        s2, ev = _run_step(
            name="Retrieve evidence from document",
            tool="tool_retrieve_evidence",
            fn=tool_retrieve_evidence,
            tool_input={"query": query, "doc_text": SAMPLE_DOC},
        )
        steps.append(s2)

        if not ev.get("hit"):
            return steps, f"I couldn’t find '{query}' in the sample document."
        excerpt = ev.get("excerpt") or ""
        return steps, f"Found '{query}'. Evidence excerpt:\n\n{excerpt}"

    if intent.get("intent") == "summary":
        # Simple non-LLM summary for demo: pick the 2-3 most "informative" lines.
        lines = [ln.strip() for ln in SAMPLE_DOC.splitlines() if ln.strip()]
        keep = lines[:1] + lines[-2:] if len(lines) >= 3 else lines
        answer = "Summary (demo):\n- " + "\n- ".join(keep)
        return steps, answer

    return steps, (
        "Try one of these:\n"
        "- 'Compute cost-to-income with income=10000 expenses=4200'\n"
        "- 'Compute LDR with loans=75000 deposits=90000'\n"
        "- 'Find liquidity'\n"
        "- 'Summarize the document'"
    )


# ---------------------- UI ----------------------

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini Agentic Flask Demo</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body { background: #f7f7f8; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
    .card { border: none; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    .badge-soft { background: #eef2ff; color: #3730a3; }
  </style>
</head>
<body>
<div class="container my-4" style="max-width: 980px;">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <div>
      <h3 class="mb-0">Mini Agentic Flask Demo</h3>
      <div class="text-muted">Shows: plan → tool calls → verification → answer</div>
    </div>
    <div>
      <a class="btn btn-outline-secondary" href="{{ url_for('reset') }}">Reset</a>
    </div>
  </div>

  <div class="card mb-3">
    <div class="card-body">
      <form method="post" action="{{ url_for('run') }}">
        <label class="form-label fw-semibold">Your goal</label>
        <textarea class="form-control" rows="3" name="goal" placeholder="e.g. Compute cost-to-income with income=10000 expenses=4200" required>{{ last_goal or '' }}</textarea>
        <div class="d-flex gap-2 mt-3">
          <button class="btn btn-primary" type="submit">Run agent</button>
          <a class="btn btn-outline-primary" href="#examples">Examples</a>
        </div>
      </form>
    </div>
  </div>

  {% if answer %}
  <div class="card mb-3">
    <div class="card-body">
      <div class="d-flex align-items-center justify-content-between">
        <h5 class="mb-2">Answer</h5>
        <span class="badge badge-soft">Agent loop demo</span>
      </div>
      <div class="mono">{{ answer }}</div>
    </div>
  </div>
  {% endif %}

  {% if steps %}
  <div class="card mb-3">
    <div class="card-body">
      <h5 class="mb-3">Execution trace</h5>
      <div class="accordion" id="trace">
        {% for s in steps %}
          <div class="accordion-item">
            <h2 class="accordion-header" id="h{{ loop.index }}">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#c{{ loop.index }}">
                Step {{ loop.index }} — {{ s.name }}
                <span class="ms-2 badge text-bg-light">{{ s.tool }}</span>
                <span class="ms-2 text-muted">{{ s.took_ms }}ms</span>
              </button>
            </h2>
            <div id="c{{ loop.index }}" class="accordion-collapse collapse" data-bs-parent="#trace">
              <div class="accordion-body">
                <div class="row g-3">
                  <div class="col-md-6">
                    <div class="fw-semibold">Input</div>
                    <div class="mono">{{ s.input | tojson(indent=2) }}</div>
                  </div>
                  <div class="col-md-6">
                    <div class="fw-semibold">Output</div>
                    <div class="mono">{{ s.output | tojson(indent=2) }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
  {% endif %}

  <div class="card" id="examples">
    <div class="card-body">
      <h5 class="mb-2">Try these examples</h5>
      <ul class="mb-0">
        <li><span class="mono">Compute cost-to-income with income=10000 expenses=4200</span></li>
        <li><span class="mono">Compute LDR with loans=75000 deposits=90000</span></li>
        <li><span class="mono">Find liquidity</span></li>
        <li><span class="mono">Summarize the document</span></li>
      </ul>
    </div>
  </div>

  <div class="text-muted small mt-3">
    Note: This is a deterministic demo (no external LLM). It’s designed to show how an agent can select tools,
    run multi-step actions, and verify outputs.
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(
        TEMPLATE,
        last_goal=session.get("last_goal"),
        steps=session.get("last_steps"),
        answer=session.get("last_answer"),
    )


@app.post("/run")
def run():
    goal = (request.form.get("goal") or "").strip()
    steps, answer = run_agent(goal)

    session["last_goal"] = goal
    session["last_steps"] = [
        {
            "name": s.name,
            "tool": s.tool,
            "input": s.input,
            "output": s.output,
            "took_ms": s.took_ms,
        }
        for s in steps
    ]
    session["last_answer"] = answer
    return redirect(url_for("home"))


@app.get("/reset")
def reset():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    print("Mini agentic demo running on http://127.0.0.1:5091")
    app.run(host="127.0.0.1", port=5091, debug=True, use_reloader=False)
