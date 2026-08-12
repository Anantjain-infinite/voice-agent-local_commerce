"""
dashboard.py — simple local dashboard for Bazaar Mitra's human escalations (Day 7).

Run with:
    python dashboard.py
Then open http://127.0.0.1:5000 in a browser. Shows open escalations by default, with
a button to mark each resolved. No login/accounts — meant for local/demo use, matching
what the escalations were sent to (a local SQLite table, per the Day 7 brief).

Requires Flask — install it if you don't already have it:
    pip install flask --break-system-packages
"""

import asyncio

from flask import Flask, redirect, render_template_string, request, url_for

import escalations

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Bazaar Mitra — Escalations</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #f7f7f5; color: #222; }
    h1 { margin-bottom: 0.15rem; font-size: 1.4rem; }
    .subtitle { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    th, td { text-align: left; padding: 0.6rem 0.7rem; border-bottom: 1px solid #eee; font-size: 0.88rem; vertical-align: top; }
    th { background: #fafafa; font-weight: 600; }
    .urgency-high { color: #b00020; font-weight: 700; }
    .urgency-medium { color: #b06000; font-weight: 700; }
    .urgency-low { color: #444; }
    .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.78rem; }
    .badge-open { background: #fde2e2; color: #a30000; }
    .badge-resolved { background: #e2f7e2; color: #1a7a1a; }
    button { padding: 0.35rem 0.7rem; border: none; border-radius: 4px; background: #222; color: white; cursor: pointer; font-size: 0.82rem; }
    button:hover { background: #444; }
    .empty { padding: 2.5rem; text-align: center; color: #888; background: white; }
    .toggle { margin-bottom: 1.2rem; }
    .toggle a { margin-right: 1.2rem; color: #555; text-decoration: none; font-size: 0.9rem; }
    .toggle a.active { font-weight: 700; color: #000; }
    .ref { font-family: ui-monospace, monospace; font-weight: 600; }
    small { color: #888; }
  </style>
</head>
<body>
  <h1>Bazaar Mitra — Human Escalations</h1>
  <div class="subtitle">Requests the agent couldn't resolve itself, sent here with the caller's permission.</div>
  <div class="toggle">
    <a href="{{ url_for('index', status='open') }}" class="{{ 'active' if status == 'open' else '' }}">Open</a>
    <a href="{{ url_for('index', status='resolved') }}" class="{{ 'active' if status == 'resolved' else '' }}">Resolved</a>
    <a href="{{ url_for('index', status='all') }}" class="{{ 'active' if status == 'all' else '' }}">All</a>
  </div>
  {% if rows %}
  <table>
    <tr>
      <th>Ref</th><th>Created</th><th>Reason</th><th>Caller</th><th>What happened</th>
      <th>What was checked</th><th>Urgency</th><th>Language</th><th>Follow-up</th><th>Status</th><th></th>
    </tr>
    {% for e in rows %}
    <tr>
      <td class="ref">ESC-{{ '%04d' % e['id'] }}</td>
      <td>{{ e['created_at'][:16].replace('T', ' ') }}</td>
      <td>{{ e['reason'] }}</td>
      <td>{{ e['caller_name'] or 'Unknown' }}<br><small>{{ e['caller_id'] }}</small></td>
      <td>{{ e['what_happened'] }}</td>
      <td>{{ e['what_agent_checked'] }}</td>
      <td class="urgency-{{ e['urgency'] }}">{{ e['urgency'] }}</td>
      <td>{{ e['caller_language'] }}</td>
      <td>{{ e['preferred_follow_up'] }}</td>
      <td><span class="badge badge-{{ e['status'] }}">{{ e['status'] }}</span></td>
      <td>
        {% if e['status'] == 'open' %}
        <form method="post" action="{{ url_for('resolve', reference_id='ESC-%04d' % e['id']) }}">
          <button type="submit">Mark resolved</button>
        </form>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div class="empty">No {{ status if status != 'all' else '' }} escalations.</div>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    status = request.args.get("status", "open")
    filter_status = status if status in ("open", "resolved") else None
    rows = asyncio.run(escalations.list_escalations(filter_status))
    return render_template_string(PAGE_TEMPLATE, rows=rows, status=status)


@app.route("/resolve/<reference_id>", methods=["POST"])
def resolve(reference_id):
    asyncio.run(escalations.resolve_escalation(reference_id))
    return redirect(url_for("index", status="open"))


if __name__ == "__main__":
    asyncio.run(escalations.init_db())
    app.run(debug=True, port=5000)