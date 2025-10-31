# server.py
import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import redis

app = Flask(__name__)
CORS(app)

CHAT_MODEL = "gpt-4o-mini"
MAX_TURNS = 12                   # keep last 12 user+assistant pairs
SESSION_TTL_SECONDS = 7*24*3600  # 7 days of inactivity

# --- OpenAI client ---
def get_openai():
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key.strip()) if key and key.strip() else None

# --- Redis client (optional until REDIS_URL set) ---
r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True) if os.getenv("REDIS_URL") else None
def _rk(sid): return f"chat:history:{sid}"

def load_history(sid: str):
    if not (r and sid): return []
    raw = r.get(_rk(sid))
    if not raw: return []
    try: return json.loads(raw)
    except: return []

def save_history(sid: str, msgs):
    if not (r and sid): return
    trimmed = msgs[-(MAX_TURNS*2):]  # user+assistant messages only
    r.set(_rk(sid), json.dumps(trimmed))
    r.expire(_rk(sid), SESSION_TTL_SECONDS)

@app.route("/")
def root():
    return jsonify({"ok": True, "message": "Backend is running"})

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").strip()),
        "redis_present": bool(r is not None)
    })

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True) or {}
    sid = (data.get("session_id") or "").strip()
    if not sid:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400
    if r:
        r.delete(_rk(sid))
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    client = get_openai()
    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY"}), 500

    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    sid = (data.get("session_id") or "").strip()

    if not msg:
        return jsonify({"error": "Missing 'message'"}), 400

    # Build conversation
    messages = [{
        "role": "system",
        "content": ("You are a helpful website assistant. Use conversation context to stay consistent. "
                    "If you don’t know, say so briefly and suggest contacting support.")
    }]

    # Load prior turns for this session
    history = load_history(sid)  # list of {role, content}
    messages.extend(history)

    # Add the new user message
    messages.append({"role": "user", "content": msg})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=300
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        app.logger.exception("OpenAI call failed")
        return jsonify({"error": f"OpenAI error: {e}"}), 500

    # Save updated history (only user/assistant, not system)
    new_hist = [m for m in messages if m["role"] in ("user","assistant")]
    new_hist.append({"role": "assistant", "content": reply})
    save_history(sid, new_hist)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
