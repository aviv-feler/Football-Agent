"""
app.py
ScoutAI Flask server - main entry point.
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from agent import load_resources, build_agent

app = Flask(__name__)

# Load resources once at startup.
print("[app] Initializing ScoutAI...", flush=True)
_engine, _national_strength, _schedule = load_resources()
_agent = build_agent(_engine, _national_strength, _schedule)
print("[app] ScoutAI is ready.", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = _agent.invoke(user_msg)
    except Exception as e:
        print(f"[app] Agent error: {e}", flush=True)
        response = (
            "Sorry, an error occurred while processing your request. "
            "Please try rephrasing the question."
        )

    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history."""
    _agent.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
