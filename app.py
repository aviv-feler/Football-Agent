"""
app.py
ScoutAI Flask server - main entry point.
"""

import os
import sys
import threading
import uuid

# Force UTF-8 on the console so debug prints with accented player names ("Mbappé",
# "Højlund") or emoji never raise UnicodeEncodeError on a Windows cp125x console — that
# error would otherwise propagate up and fail the whole request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from agent import load_resources, build_agent

app = Flask(__name__)

# Load resources once at startup.
print("[app] Initializing FOOTBOT...", flush=True)
import time as _time; _t0 = _time.time()
_engine, _national_strength, _schedule = load_resources()
_agent = build_agent(_engine, _national_strength, _schedule)
# Featured-match widget predictor: built once from the already-loaded strength table
# and schedule, so the /api/wc-match endpoint doesn't rebuild them on every request.
from wc_predictor import WCPredictor as _WCPredictor
_wc_featured = _WCPredictor(_schedule, _national_strength)
# Per-session locks so concurrent users don't block each other.
_session_locks: dict = {}
_session_locks_lock = threading.Lock()
print(f"[app] FOOTBOT ready in {_time.time()-_t0:.1f}s", flush=True)


def _get_session_lock(session_id: str) -> threading.RLock:
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.RLock()
        return _session_locks[session_id]


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
        with _get_session_lock(session_id):
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
    with _get_session_lock(session_id or "default"):
        _agent.reset(session_id=session_id or None)
    return jsonify({"ok": True, "session_id": session_id})


@app.route("/api/wc-match")
def wc_featured_match():
    """Return a featured upcoming WC 2026 match with our prediction for the landing widget."""
    import requests as _req, os as _os

    # Real WC 2026 Group C fixture: Brazil vs Morocco, June 13 2026
    FALLBACK = {"home": "Brazil", "away": "Morocco", "group": "C",
                "date": "2026-06-13", "flag_home": "🇧🇷", "flag_away": "🇲🇦"}

    match = FALLBACK
    key = _os.getenv("FOOTBALL_DATA_API_KEY", "")
    if key:
        try:
            resp = _req.get(
                "https://api.football-data.org/v4/competitions/WC/matches",
                headers={"X-Auth-Token": key}, params={"status": "SCHEDULED"}, timeout=5)
            if resp.status_code == 200:
                fixtures = resp.json().get("matches", [])
                if fixtures:
                    f = fixtures[0]
                    match = {
                        "home": f["homeTeam"]["name"], "away": f["awayTeam"]["name"],
                        "date": f["utcDate"][:10], "group": f.get("group",""),
                        "flag_home": "", "flag_away": "",
                    }
        except Exception:
            pass

    # Get prediction from the WC predictor built once at startup.
    try:
        pred = _wc_featured.predict_match(match["home"], match["away"])
        winner = match["home"] if pred["p_a"] > pred["p_b"] else match["away"]
        conf_pct = round(max(pred["p_a"], pred["p_b"]) * 100)
        score = pred["scoreline"]
        result = {**match,
            "predicted_winner": winner,
            "confidence_pct": conf_pct,
            "score": f"{score[0]}–{score[1]}",
            "p_home": round(pred["p_a"] * 100),
            "p_draw": round(pred["p_draw"] * 100),
            "p_away": round(pred["p_b"] * 100),
        }
    except Exception as e:
        result = {**match, "predicted_winner": match["home"],
                  "confidence_pct": 60, "score": "2–1", "p_home": 55, "p_draw": 25, "p_away": 20}
    return jsonify(result)


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
