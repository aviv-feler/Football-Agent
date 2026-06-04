"""
app.py
ScoutAI Flask server - main entry point.
"""

import os
import threading
import uuid
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from agent import load_resources, build_agent

app = Flask(__name__)

# Load resources once at startup.
print("[app] Initializing ScoutAI...", flush=True)
_engine, _national_strength, _schedule = load_resources()
_agent = build_agent(_engine, _national_strength, _schedule)
_agent_lock = threading.RLock()
print("[app] ScoutAI is ready.", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip() or str(uuid.uuid4())
    if len(session_id) > 128:
        session_id = str(uuid.uuid4())

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        with _agent_lock:
            response = _agent.invoke(user_msg, session_id=session_id)
    except Exception as e:
        print(f"[app] Agent error: {e}", flush=True)
        response = (
            "Sorry, an error occurred while processing your request. "
            "Please try rephrasing the question."
        )

    return jsonify({"response": response, "session_id": session_id})


@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history — new chat."""
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    with _agent_lock:
        _agent.reset(session_id=session_id or None)
    return jsonify({"ok": True, "session_id": session_id})


@app.route("/healthz")
def healthz():
    """Health check for hosting platforms."""
    return jsonify({
        "ok": True,
        "players": int(len(_engine.df)),
        "world_cup_matches": int(len(_schedule)),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
