import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or not key.strip():
        return None
    return OpenAI(api_key=key.strip())

@app.route("/health")
def health():
    present = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").strip())
    return jsonify({"ok": True, "openai_key_present": present})

@app.route("/chat", methods=["POST"])
def chat():
    client = get_client()
    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY"}), 500

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
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port)
