import joblib, json, os, sqlite3, time
from flask import Flask, request, jsonify, render_template_string, g, abort
import tldextract, validators

CFG = json.load(open("config.json"))
DB_PATH = CFG.get("DB_PATH", "phish_shield.db")

# Load model artifacts
if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
    raise RuntimeError("Run train_model.py first to create vectorizer.pkl and model.pkl")

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

app = Flask(__name__)

# Simple warning page template
WARNING_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚠️ Security Warning</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f8f9fa; color: #343a40; margin: 0; }
        .container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 500px; width: 90%; }
        h2 { font-size: 1.8em; color: #dc3545; }
        b { color: #0056b3; }
        button { background-color: #28a745; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1em; cursor: pointer; transition: background-color 0.2s; }
        button:hover { background-color: #218838; }
        a { color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Warning — Suspicious Site Detected</h2>
        <p>The site <b>{{domain}}</b> looks suspicious (score={{score:.2f}}).</p>
        <form method="post" action="/admin/override">
          <input type="hidden" name="domain" value="{{domain}}">
          <button type="submit">Proceed Anyway (temporary)</button>
        </form>
        <p><a href="/">Return Home</a></p>
    </div>
</body>
</html>
"""

# ---------- DB helpers ----------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scored_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        domain TEXT,
        score REAL,
        label TEXT,
        action TEXT,
        source TEXT,
        client_ip TEXT,
        ts INTEGER
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS blocked_domains (
        domain TEXT PRIMARY KEY,
        reason TEXT,
        ts INTEGER
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS whitelist (
        domain TEXT PRIMARY KEY,
        ts INTEGER
    )""")
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# ---------- utility ----------
def extract_domain(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = "http://" + url
        if not validators.url(url):
            return url # Return original string if it's not a valid URL structure
        ext = tldextract.extract(url)
        if ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return ext.domain.lower()
    except Exception:
        return url

def is_whitelisted(domain):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT domain FROM whitelist WHERE domain=?", (domain,))
    return cur.fetchone() is not None

def add_block(domain, reason="auto"):
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT OR REPLACE INTO blocked_domains(domain,reason,ts) VALUES (?,?,?)",
                (domain, reason, int(time.time())))
    db.commit()

def log_event(url, domain, score, label, action, source, client_ip):
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO scored_events(url,domain,score,label,action,source,client_ip,ts) VALUES (?,?,?,?,?,?,?,?)",
                (url, domain, score, label, action, source, client_ip, int(time.time())))
    db.commit()

# ---------- scoring endpoint ----------
@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "") or ""
    if not url:
        return jsonify({"error": "URL parameter is missing"}), 400

    source = data.get("source", "unknown")
    client_ip = data.get("client_ip", request.remote_addr)
    domain = extract_domain(url) or url

    # whitelist check first
    if is_whitelisted(domain):
        resp = {"url": url, "domain": domain, "score": 0.0, "label": "whitelisted", "action": "allow"}
        log_event(url, domain, resp["score"], resp["label"], resp["action"], source, client_ip)
        return jsonify(resp)

    # vectorize and predict
    x = vectorizer.transform([url])
    prob = float(model.predict_proba(x)[0][1])
    label = "phish" if prob >= 0.5 else "benign"

    # determine action
    if prob >= CFG.get("BLOCK_THRESHOLD", 0.95):
        action = "block"
        add_block(domain, reason="model")
    elif prob >= CFG.get("QUARANTINE_THRESHOLD", 0.6):
        action = "quarantine"
    else:
        action = "allow"

    resp = {"url": url, "domain": domain, "score": prob, "label": label, "action": action}
    log_event(url, domain, prob, label, action, source, client_ip)
    return jsonify(resp)

# ---------- warning / quarantine page ----------
@app.route("/warning/<path:domain>", methods=["GET"])
def warning(domain):
    # Render the warning page; domain is already cleaned after DNS redirect
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT score FROM scored_events WHERE domain=? ORDER BY ts DESC LIMIT 1", (domain,))
    row = cur.fetchone()
    score = row["score"] if row else 0.0
    return render_template_string(WARNING_PAGE, domain=domain, score=score)

# ---------- admin APIs ----------
@app.route("/admin/view_logs", methods=["GET"])
def admin_view_logs():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM scored_events ORDER BY ts DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    return jsonify(rows)

@app.route("/admin/blocked", methods=["GET"])
def admin_view_blocked():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM blocked_domains")
    rows = [dict(r) for r in cur.fetchall()]
    return jsonify(rows)

@app.route("/admin/whitelist", methods=["POST"])
def admin_whitelist():
    domain = request.form.get("domain") or (request.json and request.json.get("domain"))
    if not domain:
        return jsonify({"error":"no domain"}), 400
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT OR REPLACE INTO whitelist(domain,ts) VALUES (?,?)", (domain, int(time.time())))
    db.commit()
    return jsonify({"ok":True,"domain":domain})

@app.route("/admin/override", methods=["POST"])
def admin_override():
    # override posted from warning page form
    domain = request.form.get("domain")
    if not domain:
        abort(400)
    # temporary override = add to whitelist
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT OR REPLACE INTO whitelist(domain,ts) VALUES (?,?)", (domain, int(time.time())))
    db.commit()
    return f"Domain {domain} whitelisted temporarily. Go back."

# ---------- init DB and run ----------
if __name__ == "__main__":
    # FIX: Run init_db() inside an application context
    with app.app_context():
        init_db()
    print(f"Running scoring service on {CFG.get('HOST')}:{CFG.get('SCORE_PORT')}")
    app.run(host=CFG.get("HOST","0.0.0.0"), port=CFG.get("SCORE_PORT",5000))
