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

# ---------- Embedding helpers ----------
def _embed_text(text: str):
    e = client.embeddings.create(model=EMBED_MODEL, input=text)
    return e.data[0].embedding

def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-8
    nb = math.sqrt(sum(y*y for y in b)) + 1e-8
    return dot / (na * nb)

# ---------- FAQ KB ----------
FAQ = []
FAQ_EMB = []   # list of {"q","a","vec":[...]} built at startup

def _load_faq_and_build():
    """Loads faq.json and builds embeddings (question+answer)."""
    global FAQ, FAQ_EMB
    FAQ_EMB = []
    path = os.path.join(os.path.dirname(__file__), "faq.json")
    with open(path, "r", encoding="utf-8") as f:
        FAQ = json.load(f)
    for item in FAQ:
        text = f"Q: {item['q']}\nA: {item['a']}"
        FAQ_EMB.append({"q": item["q"], "a": item["a"], "vec": _embed_text(text)})

def _faq_top(user_query: str, k=3):
    if not FAQ_EMB:
        return []
    qvec = _embed_text(user_query)
    scored = [(_cosine(qvec, row["vec"]), row) for row in FAQ_EMB]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [row for _, row in scored[:k]]

# ---------- Pricing KB ----------
PRICING = []      # raw pricing items from pricing.json
PR_EMB = []       # list of {"title","desc","price","vec":[...]} built at startup

def _load_pricing_and_build():
    """Loads pricing.json and builds embeddings for each line item."""
    global PRICING, PR_EMB
    PR_EMB = []
    path = os.path.join(os.path.dirname(__file__), "pricing.json")
    if not os.path.exists(path):
        PRICING = []
        PR_EMB = []
        return
    with open(path, "r", encoding="utf-8") as f:
        PRICING = json.load(f)

    # Flatten categories into individual items with title/desc/price
    def items_from(pr):
        for cat in pr:
            cat_name = cat.get("category","")
            for show in cat.get("shows", []):
                title = show.get("name","").strip()
                price = show.get("price", None)
                desc  = show.get("description","").strip()
                yield {
                    "category": cat_name,
                    "title": title,
                    "price": price,
                    "desc": desc
                }

    for it in items_from(PRICING):
        # Embed title + description + price text for better recall
        price_text = f"${it['price']}" if it.get("price") not in (None, "") else "custom"
        blob = f"Title: {it['title']}\nCategory: {it['category']}\nPrice: {price_text}\nDescription: {it['desc']}"
        PR_EMB.append({
            "title": it["title"],
            "category": it["category"],
            "price": it["price"],
            "desc": it["desc"],
            "vec": _embed_text(blob)
        })

def _pricing_top(user_query: str, k=3):
    if not PR_EMB:
        return []
    qvec = _embed_text(user_query)
    scored = [(_cosine(qvec, row["vec"]), row) for row in PR_EMB]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [row for _, row in scored[:k]]

# ---------- Intent helper ----------
_PRICE_TERMS = (
    "price", "pricing", "cost", "how much", "rate", "rates", "fee", "fees",
    "charge", "charges", "budget", "spend", "host", "bartender", "hunk",
    "g-string", "bachelor", "bachelorette", "girls", "guys", "show"
)

def looks_like_pricing(q: str) -> bool:
    ql = q.lower()
    return any(term in ql for term in _PRICE_TERMS)

# ---------- Build both KBs at startup ----------
try:
    if client:
        _load_faq_and_build()
        _load_pricing_and_build()
except Exception as e:
    app.logger.exception("KB embedding build failed: %s", e)

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
        "faq_loaded": bool(len(FAQ_EMB) > 0),
        "faq_count": len(FAQ_EMB),
        "pricing_loaded": bool(len(PR_EMB) > 0),
        "pricing_count": len(PR_EMB)
    })

@app.route("/reload", methods=["POST"])
def reload_kb():
    """Rebuild embeddings after updating faq.json or pricing.json (no code change)."""
    if not client:
        return jsonify({"ok": False, "error": "Missing OPENAI_API_KEY"}), 500
    try:
        _load_faq_and_build()
        _load_pricing_and_build()
        return jsonify({
            "ok": True,
            "faq_count": len(FAQ_EMB),
            "pricing_count": len(PR_EMB)
        })
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

    # System policy
    messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant for Erotic Icon Entertainment. "
            "Use provided context (FAQ/pricing) to answer. If context is not relevant, say you don't know and suggest contacting support."
        )
    }]

    # Conversation memory
    history = load_history(sid)
    messages.extend(history)

    # Retrieve
    faq_hits = _faq_top(msg, k=3)
    pricing_hits = _pricing_top(msg, k=3) if looks_like_pricing(msg) else []

    # Build context blocks (FIXED)
    faq_ctx = "\n\n".join([f"Q: {it['q']}\nA: {it['a']}" for it in faq_hits]) if faq_hits else ""
    pr_ctx = "\n\n".join([
        f"Title: {row['title']}\nCategory: {row['category']}\n"
        f"Price: {'$'+str(row['price']) if row['price'] not in (None, '') else 'Custom'}\n"
        f"Details: {row['desc']}"
        for row in pricing_hits
    ]) if pricing_hits else ""

    ctx_parts = []
    if faq_ctx: ctx_parts.append("FAQ Context:\n" + faq_ctx)
    if pr_ctx:  ctx_parts.append("Pricing Context:\n" + pr_ctx)
    full_ctx = "\n\n".join(ctx_parts) if ctx_parts else "(no relevant context)"

    user_with_ctx = (
        f"{full_ctx}\n\n"
        f"User question: {msg}\n"
        f"Instructions: If pricing context is present, answer with the exact package names and prices. "
        f"If a price is 'Custom', say it's customizable and suggest contacting us for a quote. "
        f"Be concise and friendly."
    )

    messages.append({"role": "user", "content": user_with_ctx})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=350
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        app.logger.exception("OpenAI call failed")
        return jsonify({"error": f"OpenAI error: {e}"}), 500

    # Save user/assistant only
    new_hist = [m for m in messages if m["role"] in ("user","assistant")]
    new_hist.append({"role": "assistant", "content": reply})
    save_history(sid, new_hist)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
