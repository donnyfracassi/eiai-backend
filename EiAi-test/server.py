import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# Read your key from the environment (what you just set up)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
# Allow your front-end to call this server during development
CORS(app, resources={r"/chat": {"origins": "*"}})

# Simple in-memory rate limit (per-process; good enough for local dev)
from time import time
last_call = 0

@app.route("/chat", methods=["POST"])
def chat():
    global last_call
    data = request.get_json(force=True)
    user_message = (data or {}).get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Missing 'message'"}), 400

    # super basic 0.5s throttle so you don't fat-finger and spam requests
    now = time()
    if now - last_call < 0.5:
        return jsonify({"error": "Slow down"}), 429
    last_call = now

    # Compose messages; you can add system guidance here
    messages = [
        {"role": "system", "content": "You are a friendly, concise website assistant."},
        {"role": "user", "content": user_message}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",        # great quality/price for assistants
        temperature=0.7,            # adjust style/creativity
        max_tokens=400,             # control cost
        messages=messages,
    )

    reply = resp.choices[0].message.content
    return jsonify({"reply": reply})
    

if __name__ == "__main__":
    # Run locally: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
