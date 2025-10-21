import json, os, sqlite3, time
from flask import Flask, request, jsonify, render_template_string, g, abort
import numpy as np

# NEW: Import TFLite runtime, with a fallback to full TensorFlow for compatibility
try:
    # Optimized for Raspberry Pi and other edge devices
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for development machines with full TensorFlow installed
    print("[!] TFLite runtime not found. Falling back to full TensorFlow.")
    import tensorflow as tf

    tflite = tf.lite
import joblib
from sklearn.preprocessing import StandardScaler
from functools import wraps
from urllib.parse import urlparse
import tldextract
import re
import validators
import socket

# --- Load Config and Constants ---
CFG = json.load(open("config.json"))
DB_PATH = CFG.get("DB_PATH", "phish_shield.db")
ADMIN_API_KEY = CFG.get("ADMIN_API_KEY")
# Get the local IP address to distinguish direct access from a DNS redirect
try:
    LOCAL_IP = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    LOCAL_IP = '127.0.0.1'


# -------------------------------------------------------------------
# Expert Feature Engineering Function (Matches training script)
# -------------------------------------------------------------------
def extract_features(url):
    features = []
    if not url.startswith(('http://', 'https://')):
        url = "http://" + url
    hostname, path, domain_parts = '', '', tldextract.extract('')
    try:
        parsed_url = urlparse(url)
        domain_parts = tldextract.extract(url)
        hostname = '.'.join(p for p in [domain_parts.subdomain, domain_parts.domain, domain_parts.suffix] if p)
        path = parsed_url.path
    except Exception:
        hostname, path = str(url), ''
    features.extend([
        len(url), len(hostname), url.count('.'), url.count('-'), url.count('_'),
        url.count('/'), url.count('?'), url.count('='), url.count('@'), url.count('&'),
        len(re.findall(r'\d', url)), len(re.findall(r'[a-zA-Z]', url)),
        sum(url.lower().count(word) for word in
            ['login', 'signin', 'verify', 'account', 'update', 'secure', 'banking', 'password', 'confirm']),
        1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname) else 0,
        hostname.count('.'), hostname.count('-'),
        len(domain_parts.subdomain.split('.')) if domain_parts.subdomain else 0,
        len(domain_parts.suffix), len(path), path.count('/'), path.count('.'),
        1 if '//' in path else 0
    ])
    return np.array(features, dtype=np.float32)


# --- Artifact Loading (for TFLite model and Scaler) ---
def load_artifacts():
    if not all(os.path.exists(f) for f in ["phishing_model.tflite", "scaler.pkl"]):
        raise RuntimeError("Run train_model_advanced.py first to create model and scaler files.")

    print("[+] Loading artifacts for TFLite...")
    interpreter = tflite.Interpreter(model_path="phishing_model.tflite")
    interpreter.allocate_tensors()
    scaler = joblib.load("scaler.pkl")
    blocklist = set()
    if os.path.exists("blocklist.txt"):
        with open("blocklist.txt", "r") as f:
            blocklist = set(line.strip() for line in f)

    print("[+] Artifacts loaded successfully.")
    return interpreter, blocklist, scaler


interpreter, BLOCKLIST, scaler = load_artifacts()
app = Flask(__name__)


# --- Prediction function for TFLite ---
def predict_with_tflite(features_scaled):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], features_scaled)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]


# --- Shared Prediction Function (2-tier system) ---
def predict_url_phishing(url):
    if url in BLOCKLIST:
        return (1.0, "Blocklist")
    try:
        features = extract_features(url)
        features_scaled = scaler.transform(np.reshape(features, (1, -1)))
        probability = predict_with_tflite(features_scaled.astype(np.float32))
        return (float(probability), "AI Model")
    except Exception as e:
        print(f"[!] Error during prediction for URL '{url}': {e}")
        return (0.0, "Error")


# --- Utility and Web Interface code ---
def extract_domain_utility(url):
    try:
        if not url.startswith(('http://', 'https://')): url = "http://" + url
        if not validators.url(url): return url
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()
    except Exception:
        return url


def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') != ADMIN_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated_function


# (HTML Templates, DB helpers, and Admin routes remain unchanged from previous version)
INDEX_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Phishing Detector</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; background-color: #f8f9fa; color: #343a40; margin: 0; padding-top: 50px; }
        .container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 600px; width: 90%; }
        h1 { font-size: 2em; color: #333; margin-bottom: 10px; }
        p.subtitle { color: #6c757d; margin-bottom: 30px; }
        form { display: flex; flex-direction: column; gap: 15px; margin-bottom: 30px; }
        input[type="text"] { padding: 12px; border: 1px solid #ccc; border-radius: 8px; font-size: 1em; }
        button { background-color: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1em; cursor: pointer; transition: background-color 0.2s; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 20px; border-radius: 8px; text-align: left; }
        .result.phish { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .result.benign { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .result p { margin: 5px 0; font-size: 1.1em;}
        .result b { word-break: break-all; }
        .result .source { font-style: italic; color: #6c757d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Phishing URL Detector</h1>
        <p class="subtitle">Enter a URL to check if it's a potential phishing site.</p>
        <form method="post" action="/">
          <input type="text" name="url" placeholder="e.g., https://example.com" required value="{{ url_to_check or '' }}">
          <button type="submit">Check URL</button>
        </form>

        {% if result %}
        <div class="result {{ result.label }}">
            <p><strong>URL Checked:</strong> <b>{{ result.url }}</b></p>
            <p><strong>Prediction:</strong> <strong style="text-transform: capitalize;">{{ result.label }}</strong></p>
            <p><strong>Phishing Probability:</strong> {{ "%.2f"|format(result.score * 100) }}%</p>
            <p class="source">Detection Source: {{ result.source }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
WARNING_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚠️ Site Blocked</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #fff0f1; color: #343a40; margin: 0; }
        .container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 500px; width: 90%; }
        h2 { font-size: 1.8em; color: #dc3545; }
        p { font-size: 1.1em; line-height: 1.6; }
        b { color: #0056b3; word-break: break-all; }
        .reason { font-style: italic; color: #6c757d; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Security Warning: Site Blocked</h2>
        <p>Access to the domain <b>{{domain}}</b> has been blocked by your network's Phishing Detector.</p>
        <p class="reason">Reason: This domain is either on a known blocklist or was identified as a potential threat by the AI model.</p>
    </div>
</body>
</html>
"""


def get_db():
    db = getattr(g, "_database", None)
    if db is None: db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False); db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db();
    cur = db.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS scored_events (id INTEGER PRIMARY KEY, url TEXT, domain TEXT, score REAL, label TEXT, action TEXT, source TEXT, client_ip TEXT, ts INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS blocked_domains (domain TEXT PRIMARY KEY, reason TEXT, ts INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS whitelist (domain TEXT PRIMARY KEY, ts INTEGER)")
    db.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None: db.close()


def is_whitelisted_db(domain):
    db = get_db();
    cur = db.cursor();
    cur.execute("SELECT domain FROM whitelist WHERE domain=?", (domain,));
    return cur.fetchone() is not None


def add_block(domain, reason="auto"):
    db = get_db();
    cur = db.cursor();
    cur.execute("INSERT OR REPLACE INTO blocked_domains VALUES (?,?,?)", (domain, reason, int(time.time())));
    db.commit()


def log_event(url, domain, score, label, action, source, client_ip):
    db = get_db();
    cur = db.cursor();
    cur.execute("INSERT INTO scored_events(url,domain,score,label,action,source,client_ip,ts) VALUES (?,?,?,?,?,?,?,?)",
                (url, domain, score, label, action, source, client_ip, int(time.time())));
    db.commit()


# --- Main Catch-All Route for Redirects and UI ---
@app.route('/', defaults={'path': ''}, methods=["GET", "POST"])
@app.route('/<path:path>', methods=["GET", "POST"])
def index(path):
    # Check if the request is for the detector's own IP/localhost to show the form
    # The host may include the port, so we check the start
    is_direct_access = request.host.startswith(LOCAL_IP) or request.host.startswith(
        '127.0.0.1') or request.host.startswith('localhost')

    if is_direct_access:
        if request.method == "POST":
            url = request.form.get("url")
            if not url: return render_template_string(INDEX_PAGE)
            prob, source = predict_url_phishing(url)
            label = "phish" if prob >= 0.5 else "benign"
            domain = extract_domain_utility(url)
            log_event(url, domain, prob, label, "predict", source, request.remote_addr)
            result_data = {"url": url, "score": prob, "label": label, "source": source}
            return render_template_string(INDEX_PAGE, result=result_data, url_to_check=url)
        return render_template_string(INDEX_PAGE)

    # If not direct access, it's a DNS redirect. Show the warning page.
    else:
        blocked_domain = request.host
        # Log the block event from the redirect
        log_event(f"http://{blocked_domain}", blocked_domain, 1.0, "phish", "block", "DNS Sinkhole",
                  request.remote_addr)
        return render_template_string(WARNING_PAGE, domain=blocked_domain)


# --- API endpoint for DNS Monitor ---
@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json(silent=True) or {};
    url = data.get("url", "");
    client_ip = data.get("client_ip", request.remote_addr)
    if not url: return jsonify({"error": "URL parameter is missing"}), 400
    domain = extract_domain_utility(url)
    if is_whitelisted_db(domain):
        resp = {"url": url, "domain": domain, "score": 0.0, "label": "whitelisted", "action": "allow"}
        log_event(url, domain, resp["score"], "whitelisted", "allow", "DB Whitelist", client_ip);
        return jsonify(resp)

    prob, detection_source = predict_url_phishing(url)
    label = "phish" if prob >= 0.5 else "benign"

    if detection_source == "Blocklist" or prob >= CFG.get("BLOCK_THRESHOLD", 0.95):
        action = "block"
        add_block(domain, reason=detection_source)
    else:
        action = "allow"

    resp = {"url": url, "domain": domain, "score": prob, "label": label, "action": action}
    log_event(url, domain, prob, resp['label'], action, detection_source, client_ip);
    return jsonify(resp)


# (Admin routes remain unchanged)
@app.route("/admin/view_logs", methods=["GET"])
@api_key_required
def admin_view_logs():
    db = get_db();
    cur = db.cursor();
    cur.execute("SELECT * FROM scored_events ORDER BY ts DESC LIMIT 200");
    rows = [dict(r) for r in cur.fetchall()];
    return jsonify(rows)


@app.route("/admin/blocked", methods=["GET"])
@api_key_required
def admin_view_blocked():
    db = get_db();
    cur = db.cursor();
    cur.execute("SELECT * FROM blocked_domains");
    rows = [dict(r) for r in cur.fetchall()];
    return jsonify(rows)


@app.route("/admin/whitelist", methods=["POST"])
@api_key_required
def admin_whitelist():
    domain = request.form.get("domain") or (request.json and request.json.get("domain"))
    if not domain: return jsonify({"error": "no domain"}), 400
    db = get_db();
    cur = db.cursor();
    cur.execute("INSERT OR REPLACE INTO whitelist VALUES (?,?)", (domain, int(time.time())));
    db.commit()
    return jsonify({"ok": True, "domain": domain})


# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        init_db()

    # The redirect feature requires the app to listen on the standard HTTP port (80)
    print("\n[+] Starting Phishing Detector Web Service...")
    print("      IMPORTANT: This script must be run with sudo to use port 80.")
    print(f"      Service will be available at http://{LOCAL_IP}:80")
    print(f"      DNS Monitor should point to this machine's IP ({LOCAL_IP}) for blocked domains.\n")

    # Use port 80 to catch browser redirects from the DNS sinkhole
    app.run(host="0.0.0.0", port=80)

