import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def get_openai_key():
    # Single source of truth for both /health and /chat
    v = os.getenv("OPENAI_API_KEY")
    return v.strip() if v and v.strip() else None

def get_client():
    key = get_openai_key()
    # optional: allow a temporary header override for testing
    if not key:
        hdr = request.headers.get("x-openai-key")
        if hdr and hdr.strip():
            key = hdr.strip()
    return OpenAI(api_key=key) if key else None

@app.route("/")
def root():
    return jsonify({"ok": True, "message": "Backend is running"})

@app.route("/health")
def health():
    key = get_openai_key()
    return jsonify({
        "ok": True,
        "openai_key_present": bool(key),
        "openai_key_len": len(key) if key else None
    })

@app.route("/chat", methods=["POST"])
def chat():
    client = get_client()
    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY (or x-openai-key header)"}), 500

    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Missing 'message'"}), 400

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_message}],
            temperature=0.7,
            max_tokens=200
        )
        return jsonify({"reply": resp.choices[0].message.content})
    except Exception as e:
        app.logger.exception("OpenAI call failed")
        return jsonify({"error": f"OpenAI error: {e.__class__.__name__}: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
