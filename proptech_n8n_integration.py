import json
import os
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

PROPTECH_WEBHOOK_URL = os.getenv(
    "PROPTECH_WEBHOOK_URL",
    "https://subhajit86.app.n8n.cloud/webhook/atlas-property-backend",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "JSON root must be an object"

    action = payload.get("action")
    requests_list = payload.get("requests")

    if not isinstance(action, str) or not action.strip():
        return False, "Missing/invalid 'action' (must be a non-empty string)"

    if not isinstance(requests_list, list) or not requests_list:
        return False, "Missing/invalid 'requests' (must be a non-empty array)"

    # Best-effort validation: ensure each request is an object with request_id.
    for i, req in enumerate(requests_list[:200]):
        if not isinstance(req, dict):
            return False, f"requests[{i}] must be an object"
        if not isinstance(req.get("request_id"), str) or not req["request_id"].strip():
            return False, f"requests[{i}].request_id is required"

    return True, "ok"


INDEX_HTML = """\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PropTech JSON → n8n Webhook</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
      .card { max-width: 980px; margin: 0 auto; padding: 18px 18px 10px; border: 1px solid #e5e7eb; border-radius: 12px; }
      label { display: block; font-weight: 600; margin-top: 12px; }
      input[type=file] { width: 100%; margin-top: 6px; padding: 10px; border: 1px solid #d1d5db; border-radius: 10px; }
      button { margin-top: 14px; padding: 10px 14px; border: 0; border-radius: 10px; background: #111827; color: white; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      pre { white-space: pre-wrap; word-wrap: break-word; background: #0b1020; color: #e5e7eb; padding: 14px; border-radius: 12px; margin-top: 14px; }
      .muted { color: #6b7280; font-size: 0.95rem; }
      .ok { color: #065f46; }
      .err { color: #991b1b; }
      .hint { background: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; border-radius: 10px; margin-top: 10px; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.95rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2 style="margin:0 0 6px">Upload PropTech JSON and trigger n8n</h2>
      <div class="muted">This uploads a JSON file, then POSTs its contents to your n8n webhook and displays the response.</div>

      <div class="hint">
        <div style="font-weight:600; margin-bottom:6px">Expected JSON format</div>
        <pre style="margin:0; background:#111827; color:#e5e7eb">{
  "action": "bulk_property_analysis",
  "requests": [
    {
      "request_id": "REQ-1",
      "city": "Dubai",
      "location": "Dubai Marina",
      "property_type": "2 BHK",
      "size_sqft": 1250,
      "median_sale_psf": 2100,
      "median_rent_psf_monthly": 95,
      "year_built": 2019,
      "condition_premium_pct": 6,
      "vacancy_pct": 5,
      "maintenance_pct": 1.2,
      "counts": {
        "supermarket": 8,
        "hospital": 2,
        "school": 4,
        "metro_station": 1,
        "restaurant": 25
      },
      "ratings": {
        "supermarket": 4.4,
        "hospital": 4.3,
        "school": 4.2,
        "metro_station": 4.5,
        "restaurant": 4.6
      }
    }
  ]
}</pre>
      </div>

      <form id="uploadForm" enctype="multipart/form-data">
        <label for="json_file">JSON file</label>
        <input id="json_file" name="json_file" type="file" accept="application/json,.json" />
        <button id="submitBtn" type="submit">Upload & Invoke</button>
      </form>

      <div id="status" class="muted" style="margin-top:10px"></div>
      <pre id="out" style="display:none"></pre>
    </div>

    <script>
      const form = document.getElementById('uploadForm');
      const fileInput = document.getElementById('json_file');
      const btn = document.getElementById('submitBtn');
      const status = document.getElementById('status');
      const out = document.getElementById('out');

      async function doUpload() {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
          status.textContent = 'Please choose a JSON file.';
          status.className = 'err';
          return;
        }
        if (!file.name.toLowerCase().endsWith('.json')) {
          status.textContent = 'Only .json files are allowed.';
          status.className = 'err';
          return;
        }

        btn.disabled = true;
        status.textContent = 'Uploading… (this will call n8n)';
        status.className = 'muted';
        out.style.display = 'none';

        const fd = new FormData(form);

        try {
          const resp = await fetch('/invoke', {
            method: 'POST',
            body: fd,
            headers: { 'Accept': 'application/json' }
          });
          const data = await resp.json().catch(() => ({ ok: false, message: 'Non-JSON response from server' }));
          if (!resp.ok) {
            status.textContent = data && data.message ? data.message : ('Invoke failed (HTTP ' + resp.status + ')');
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


@app.route("/invoke", methods=["POST"])
def invoke():
    f = request.files.get("json_file")
    if not f or not f.filename:
        return jsonify({"ok": False, "message": "Please select a JSON file."}), 400

    if not f.filename.lower().endswith(".json"):
        return jsonify({"ok": False, "message": "Please upload a .json file only."}), 400

    try:
        raw = f.read()
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return jsonify({"ok": False, "message": "JSON file must be UTF-8 encoded."}), 400
    except json.JSONDecodeError as e:
        return jsonify({"ok": False, "message": f"Invalid JSON: {e.msg} at line {e.lineno} col {e.colno}"}), 400

    ok, msg = _validate_payload(payload)
    if not ok:
        return jsonify({"ok": False, "message": msg}), 400

    if not PROPTECH_WEBHOOK_URL:
        return jsonify({"ok": False, "message": "PROPTECH_WEBHOOK_URL is not configured."}), 500

    try:
        resp = requests.post(
            PROPTECH_WEBHOOK_URL,
            json=payload,
            timeout=120,
            headers={"Accept": "application/json"},
        )
        # n8n often responds with JSON, but be safe.
        try:
            body = resp.json()
        except ValueError:
            body = {"raw_response": resp.text}

        return jsonify(
            {
                "ok": resp.ok,
                "message": "Webhook invoked." if resp.ok else f"Webhook returned HTTP {resp.status_code}",
                "invoked_at": _utc_now_iso(),
                "request_action": payload.get("action"),
                "request_count": len(payload.get("requests") or []),
                "webhook_url": PROPTECH_WEBHOOK_URL,
                "status_code": resp.status_code,
                "n8n_result": body,
            }
        ), (200 if resp.ok else 502)

    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "message": f"Failed to invoke webhook: {e}"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5094"))
    app.run(host="127.0.0.1", port=port, debug=True)
