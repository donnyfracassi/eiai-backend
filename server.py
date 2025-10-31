# server.py
import os, json, math
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import redis

app = Flask(__name__)
CORS(app)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MAX_TURNS = 12                    # keep last 12 user+assistant pairs
SESSION_TTL_SECONDS = 7*24*3600   # 7 days

# --- OpenAI client ---
def get_openai():
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key.strip()) if key and key.strip() else None

client = get_openai()

# --- Redis client (optional) ---
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
    trimmed = msgs[-(MAX_TURNS*2):]  # user+assistant only
    r.set(_rk(sid), json.dumps(trimmed))
    r.expire(_rk(sid), SESSION_TTL_SECONDS)

# ---------- FAQ RAG-lite ----------
KB = []
KB_EMB = []   # list of {"q","a","vec":[...]} built at startup

def _embed_text(text: str):
    e = client.embeddings.create(model=EMBED_MODEL, input=text)
    return e.data[0].embedding

def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-8
    nb = math.sqrt(sum(y*y for y in b)) + 1e-8
    return dot / (na * nb)

def _load_kb_and_build():
    """Loads faq.json and builds embeddings (question+answer)."""
    global KB, KB_EMB
    KB_EMB = []
    path = os.path.join(os.path.dirname(__file__), "faq.json")
    with open(path, "r", encoding="utf-8") as f:
        KB = json.load(f)
    for item in KB:
        text = f"Q: {item['q']}\nA: {item['a']}"
        KB_EMB.append({"q": item["q"], "a": item["a"], "vec": _embed_text(text)})

def _retrieve_top(user_query: str, k=3):
    if not KB_EMB:
        return []
    qvec = _embed_text(user_query)
    scored = [(_cosine(qvec, row["vec"]), row) for row in KB_EMB]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [row for _, row in scored[:k]]

# Build FAQ vectors at startup (if key present)
try:
    if client:
        _load_kb_and_build()
except Exception as e:
    app.logger.exception("FAQ embedding build failed: %s", e)
# ----------------------------------

@app.route("/")
def root():
    return jsonify({"ok": True, "message": "Backend is running"})

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").strip()),
        "redis_present": bool(r is not None),
        "faq_loaded": bool(len(KB_EMB) > 0),
        "faq_count": len(KB_EMB)
    })

@app.route("/reload", methods=["POST"])
def reload_kb():
    """Rebuild FAQ embeddings after updating faq.json (no code change needed)."""
    if not client:
        return jsonify({"ok": False, "error": "Missing OPENAI_API_KEY"}), 500
    try:
        _load_kb_and_build()
        return jsonify({"ok": True, "faq_count": len(KB_EMB)})
    except Exception as e:
        app.logger.exception("Reload failed: %s", e)
        return jsonify({"ok": False, "error": f"Reload failed: {e}"}), 500

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
    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY"}), 500

    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    sid = (data.get("session_id") or "").strip()
    if not msg:
        return jsonify({"error": "Missing 'message'"}), 400

    # System prompt sets policy: answer from FAQ when possible
    messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant for Erotic Icon Entertainment. "
            "Use the provided context to answer. If the context is not relevant, say you don't know and suggest contacting support."
        )
    }]

    # conversation memory
    history = load_history(sid)
    messages.extend(history)
    messages.append({"role": "user", "content": msg})

    # retrieve FAQ snippets
    context_items = _retrieve_top(msg, k=3)
    context = "\n\n".join([f"Q: {it['q']}\nA: {it['a']}" for it in context_items])

    user_with_context = (
        f"Context from the company FAQ (use only if relevant):\n{context}\n\n"
        f"User question: {msg}\n"
        f"Answer in a concise, friendly tone."
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[*messages[:-1], {"role": "user", "content": user_with_context}],
            temperature=0.2,
            max_tokens=350
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        app.logger.exception("OpenAI call failed")
        return jsonify({"error": f"OpenAI error: {e}"}), 500

    # Save updated history (user/assistant only)
    new_hist = [m for m in messages if m["role"] in ("user","assistant")]
    new_hist.append({"role": "assistant", "content": reply})
    save_history(sid, new_hist)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
