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
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
      .kv { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; }
      .kv h3 { margin: 0 0 8px; font-size: 1rem; }
      .kvrow { display: grid; grid-template-columns: 220px 1fr; gap: 10px; padding: 6px 0; border-top: 1px solid #f3f4f6; }
      .kvrow:first-of-type { border-top: 0; }
      .k { color: #374151; font-weight: 600; }
      .v { color: #111827; overflow-wrap: anywhere; }
      .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid #e5e7eb; }
      .badge.ok { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
      .badge.err { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
      details { margin-top: 12px; }
      summary { cursor: pointer; color: #111827; font-weight: 600; }
      table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 10px; }
      th, td { text-align: left; padding: 10px 12px; border-top: 1px solid #f3f4f6; vertical-align: top; }
      th { font-size: 0.9rem; letter-spacing: 0.01em; color: #374151; background: #f9fafb; position: sticky; top: 0; z-index: 1; border-top: 0; }
      td { color: #111827; }
      tbody tr:nth-child(even) td { background: #fcfcfd; }
      tbody tr:hover td { background: #f9fafb; }
      .tableWrap { max-height: 420px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; }
      .tableWrap::-webkit-scrollbar { height: 12px; width: 12px; }
      .tableWrap::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 999px; }
      .num { text-align: right; font-variant-numeric: tabular-nums; }
      .cell { max-width: 360px; white-space: normal; overflow-wrap: break-word; word-break: normal; hyphens: auto; line-height: 1.25rem; }
      .cell.small { max-width: 240px; }
      .cell.wide { max-width: 560px; }
      .nowrap { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .clamp { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 6; overflow: hidden; }
      .pill { display: inline-block; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid #e5e7eb; background: #f9fafb; color: #111827; }

      /* Sticky first column improves scanability */
      th.stickyCol, td.stickyCol { position: sticky; left: 0; z-index: 2; }
      th.stickyCol { z-index: 3; }
      th.stickyCol, td.stickyCol { border-right: 1px solid #e5e7eb; }

      /* Column sizing (applied via classes) */
      .col-request_id { min-width: 96px; }
      .col-city { min-width: 90px; }
      .col-location { min-width: 160px; }
      .col-property_type { min-width: 110px; }
      .col-action { min-width: 190px; }
      .col-best_option { min-width: 260px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2 style="margin:0 0 6px">Upload PropTech JSON and trigger n8n</h2>
      <div class="muted">This uploads a JSON file, then POSTs its contents to your n8n webhook and displays the response.</div>

      <form id="uploadForm" enctype="multipart/form-data">
        <label for="json_file">JSON file</label>
        <input id="json_file" name="json_file" type="file" accept="application/json,.json" />
        <button id="submitBtn" type="submit">Upload & Invoke</button>
      </form>

      <div id="status" class="muted" style="margin-top:10px"></div>

      <div id="summary" class="grid" style="display:none"></div>
      <div id="result" class="kv" style="display:none; margin-top:12px"></div>
      <details id="rawBox" style="display:none">
        <summary>Raw n8n response JSON</summary>
        <pre id="out" style="margin-top:10px"></pre>
      </details>
    </div>

    <script>
      const form = document.getElementById('uploadForm');
      const fileInput = document.getElementById('json_file');
      const btn = document.getElementById('submitBtn');
      const status = document.getElementById('status');
      const out = document.getElementById('out');
      const summary = document.getElementById('summary');
      const result = document.getElementById('result');
      const rawBox = document.getElementById('rawBox');

      function escapeHtml(s) {
        return String(s)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#039;');
      }

      function renderSummary(data) {
        const ok = !!(data && data.ok);
        const badge = ok
          ? '<span class="badge ok">OK</span>'
          : '<span class="badge err">ERROR</span>';

        const items = [
          ['Status', badge],
          ['HTTP status', data && data.status_code != null ? String(data.status_code) : ''],
          ['Invoked at', data && data.invoked_at ? escapeHtml(data.invoked_at) : ''],
          ['Action', data && data.request_action ? escapeHtml(data.request_action) : ''],
          ['Request count', data && data.request_count != null ? String(data.request_count) : ''],
          ['Webhook', data && data.webhook_url ? escapeHtml(data.webhook_url) : ''],
        ];

        const left = items.slice(0, 3);
        const right = items.slice(3);

        function box(title, rows) {
          const rowsHtml = rows.map(([k,v]) => (
            '<div class="kvrow"><div class="k">' + escapeHtml(k) + '</div><div class="v">' + v + '</div></div>'
          )).join('');
          return '<div class="kv"><h3>' + escapeHtml(title) + '</h3>' + rowsHtml + '</div>';
        }

        summary.innerHTML = box('Request', left) + box('Details', right);
        summary.style.display = 'grid';
      }

      function renderResultKV(obj) {
        if (obj === null || obj === undefined) {
          result.innerHTML = '<h3>n8n Result</h3><div class="muted">(empty)</div>';
          result.style.display = 'block';
          return;
        }
        if (typeof obj !== 'object') {
          result.innerHTML = '<h3>n8n Result</h3><div class="kvrow"><div class="k">value</div><div class="v">' + escapeHtml(obj) + '</div></div>';
          result.style.display = 'block';
          return;
        }

        function stringifyCell(v) {
          if (v === null || v === undefined) return '';
          if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return String(v);
          try {
            const s = JSON.stringify(v);
            return s.length > 220 ? (s.slice(0, 220) + '…') : s;
          } catch {
            return '[object]';
          }
        }

        function findArrayOfObjects(root) {
          // Prefer common keys first
          const preferredKeys = ['results', 'items', 'data', 'output', 'responses', 'analyses', 'properties', 'requests'];
          for (const k of preferredKeys) {
            const v = root[k];
            if (Array.isArray(v) && v.length && typeof v[0] === 'object') return { key: k, arr: v };
          }
          // Otherwise: first array of objects anywhere at top-level
          for (const k of Object.keys(root)) {
            const v = root[k];
            if (Array.isArray(v) && v.length && typeof v[0] === 'object') return { key: k, arr: v };
          }
          // If the root itself is an array of objects
          if (Array.isArray(root) && root.length && typeof root[0] === 'object') return { key: '(root)', arr: root };
          return null;
        }

        function renderTable(title, arr) {
          const rows = arr.slice(0, 200);
          const hiddenCols = new Set([
            'estimated_rent_month',
          ]);
          const keySet = new Set();
          for (const r of rows) {
            if (r && typeof r === 'object') {
              for (const k of Object.keys(r)) keySet.add(k);
            }
          }
          let cols = Array.from(keySet).filter((c) => !hiddenCols.has(String(c || '').toLowerCase()));
          // Make key fields appear first if present
          const priority = ['request_id', 'id', 'name', 'city', 'location', 'property_type', 'status', 'ok', 'message', 'score', 'roi', 'cap_rate', 'irr'];
          cols.sort((a, b) => {
            const ia = priority.indexOf(a);
            const ib = priority.indexOf(b);
            if (ia === -1 && ib === -1) return a.localeCompare(b);
            if (ia === -1) return 1;
            if (ib === -1) return -1;
            return ia - ib;
          });
          cols = cols.slice(0, 16);

          // Detect numeric-like columns (right-align)
          const numericCols = new Set();
          for (const c of cols) {
            let seen = 0;
            let numeric = 0;
            for (const r of rows) {
              const v = (r || {})[c];
              if (v === null || v === undefined || v === '') continue;
              seen += 1;
              if (typeof v === 'number') {
                numeric += 1;
              } else if (typeof v === 'string' && v.trim() && !isNaN(Number(v.replaceAll(',', '')))) {
                numeric += 1;
              }
              if (seen >= 12) break;
            }
            if (seen > 0 && numeric / seen >= 0.9) numericCols.add(c);
          }

          function cellClass(col, vStr) {
            // Make verbose columns wider
            if (col.toLowerCase().includes('option') || col.toLowerCase().includes('explain') || col.toLowerCase().includes('reason')) return 'cell wide';
            if ((vStr || '').length > 80) return 'cell wide';
            if ((vStr || '').length > 40) return 'cell';
            return 'cell small';
          }

          function safeColClass(col) {
            return String(col || '').toLowerCase().replaceAll(/[^a-z0-9_\-]+/g, '-');
          }

          function shouldNoWrap(col) {
            const c = String(col || '').toLowerCase();
            return (
              c === 'request_id' ||
              c === 'city' ||
              c === 'country' ||
              c === 'property_type' ||
              c === 'batch_size' ||
              c.endsWith('_psf') ||
              c.includes('psf')
            );
          }

          function shouldClamp(col) {
            const c = String(col || '').toLowerCase();
            return c.includes('best_option') || c.includes('recommend') || c.includes('summary') || c.includes('rationale');
          }

          function renderCell(col, rawVal) {
            // Show "action" as a pill if present
            if (col === 'action' && rawVal !== null && rawVal !== undefined && String(rawVal).trim()) {
              const v = String(rawVal);
              return '<span class="pill" title="' + escapeHtml(v) + '">' + escapeHtml(v) + '</span>';
            }
            const v = stringifyCell(rawVal);
            const base = cellClass(col, v);
            const nowrap = shouldNoWrap(col) ? ' nowrap' : '';
            const clamp = shouldClamp(col) ? ' clamp' : '';
            const colCls = ' col-' + safeColClass(col);
            return '<div class="' + base + nowrap + clamp + colCls + '" title="' + escapeHtml(v) + '">' + escapeHtml(v) + '</div>';
          }

          const thead = '<thead><tr>' + cols.map(c => {
            const colCls = ' col-' + safeColClass(c);
            const sticky = (String(c) === 'request_id') ? ' stickyCol' : '';
            const classAttr = ' class="' + (numericCols.has(c) ? 'num' : '') + colCls + sticky + '"';
            return '<th' + classAttr + '>' + escapeHtml(c) + '</th>';
          }).join('') + '</tr></thead>';
          const tbody = '<tbody>' + rows.map(r => {
            return '<tr>' + cols.map(c => {
              const colCls = ' col-' + safeColClass(c);
              const isNum = numericCols.has(c);
              const sticky = (String(c) === 'request_id') ? ' stickyCol' : '';
              const classAttr = ' class="' + (isNum ? 'num' : '') + colCls + sticky + '"';
              return '<td' + classAttr + '>' + renderCell(c, (r || {})[c]) + '</td>';
            }).join('') + '</tr>';
          }).join('') + '</tbody>';

          const note = arr.length > 200
            ? '<div class="muted" style="margin-top:8px">Showing first 200 rows. See raw JSON below for full output.</div>'
            : '';

          return (
            '<h3>n8n Result</h3>' +
            '<div class="muted">Table view from <b>' + escapeHtml(title) + '</b> (' + arr.length + ' items)</div>' +
            '<div class="tableWrap"><table>' + thead + tbody + '</table></div>' +
            note
          );
        }

        const found = findArrayOfObjects(obj);
        if (found && found.arr && found.arr.length) {
          result.innerHTML = renderTable(found.key, found.arr);
          result.style.display = 'block';
          return;
        }

        // Fallback: top-level key/value view
        const keys = Object.keys(obj).filter((k) => String(k || '').toLowerCase() !== 'estimated_rent_month');
        const rows = keys.slice(0, 40).map((k) => {
          return '<div class="kvrow"><div class="k">' + escapeHtml(k) + '</div><div class="v">' + escapeHtml(stringifyCell(obj[k])) + '</div></div>';
        }).join('');

        const note = keys.length > 40
          ? '<div class="muted" style="margin-top:8px">Showing first 40 fields. See raw JSON below for full output.</div>'
          : '';

        result.innerHTML = '<h3>n8n Result</h3>' + rows + note;
        result.style.display = 'block';
      }

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
        summary.style.display = 'none';
        result.style.display = 'none';
        rawBox.style.display = 'none';

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

          renderSummary(data);

          const n8nRes = data && data.n8n_result !== undefined ? data.n8n_result : null;
          renderResultKV(n8nRes);

          out.textContent = JSON.stringify(n8nRes, null, 2);
          rawBox.style.display = 'block';
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
