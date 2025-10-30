import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def _get_key():
    v = os.getenv("OPENAI_API_KEY")
    return v.strip() if v and v.strip() else None

def get_client():
    k = _get_key()
    # TEMP: allow header override for testing
    if not k:
        hdr = request.headers.get("x-openai-key")
        if hdr and hdr.strip():
            k = hdr.strip()
    return OpenAI(api_key=k) if k else None

@app.route("/")
def root():
    return jsonify({"ok": True, "message": "Backend is running"})

# shows whether OPENAI_API_KEY is present, plus length only (no value)
@app.route("/health")
def health():
    v = os.getenv("OPENAI_API_KEY")
    present = bool(v and v.strip())
    preview = (f"len={len(v.strip())}" if present else None)
    return jsonify({
        "ok": True,
        "openai_key_present": present,
        "preview": preview
    })

# lists env var NAMES only (no secrets), so we can confirm the key name exists
@app.route("/debug/env")
def debug_env():
    keys = sorted(os.environ.keys())
    # don’t return values; only names
    return jsonify({"keys": keys})

@app.route("/chat", methods=["POST"])
def chat():
    client = get_client()
    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY (or x-openai-key header)"}), 500

    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Missing 'message'"}), 400

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        temperature=0.7,
        max_tokens=400,
    )
    return jsonify({"reply": resp.choices[0].message.content})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
