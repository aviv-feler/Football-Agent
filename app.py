"""
app.py
ScoutAI Flask server - main entry point.
"""

import os
import sys
import threading
import uuid
import subprocess
import datetime

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


def _compute_version() -> dict:
    """Build a version stamp that increments per commit + a build timestamp, so a
    refresh after deploy confirms the newest code is live. Uses the git commit
    count/hash when available, falling back to env vars (set by the host) or 'dev'."""
    here = os.path.dirname(os.path.abspath(__file__))
    def _git(args):
        try:
            return subprocess.check_output(["git", *args], cwd=here,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    commit = (os.getenv("RENDER_GIT_COMMIT") or os.getenv("SOURCE_VERSION")
              or _git(["rev-parse", "HEAD"]) or "")
    commit = commit[:7] if commit else "dev"
    count = _git(["rev-list", "--count", "HEAD"]) or "0"
    build_time = os.getenv("BUILD_TIME") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "version": f"v1.0.{count}",
        "commit": commit,
        "build_time": build_time,
        "label": f"v1.0.{count} · {commit} · {build_time}",
    }


_VERSION = _compute_version()
print(f"[app] Build {_VERSION['label']}", flush=True)

# Load resources once at startup.
print("[app] Initializing FOOTBOT...", flush=True)
import time as _time; _t0 = _time.time()
_engine, _national_strength, _schedule = load_resources()
_agent = build_agent(_engine, _national_strength, _schedule)
# Featured-match widget predictor: built once from the already-loaded strength table
# and schedule, so the /api/wc-match endpoint doesn't rebuild them on every request.
from wc_predictor import WCPredictor as _WCPredictor
_wc_featured = _WCPredictor(_schedule, _national_strength)
# Trained Logistic Regression predictor for the featured-match widget.
try:
    from match_predictor import MatchPredictor as _MatchPredictor
    _match_predictor = _MatchPredictor()
except Exception as _e:
    print(f"[app] Match predictor unavailable for widget: {_e}", flush=True)
    _match_predictor = None
# Per-session locks so concurrent users don't block each other.
_session_locks: dict = {}
_session_locks_lock = threading.Lock()
# Warm the league-table carousel cache in the background (non-blocking).
import live_tables as _live_tables
_live_tables.prefetch(os.getenv("FOOTBALL_DATA_API_KEY", ""))
print(f"[app] FOOTBOT ready in {_time.time()-_t0:.1f}s", flush=True)


def _get_session_lock(session_id: str) -> threading.RLock:
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.RLock()
        return _session_locks[session_id]


@app.route("/")
def index():
    return render_template("index.html", app_version=_VERSION["label"])


@app.route("/version")
def version():
    """Build/version stamp — refresh after deploy to confirm the newest code is live."""
    return jsonify(_VERSION)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip() or str(uuid.uuid4())
    if len(session_id) > 128:
        session_id = str(uuid.uuid4())

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    viz = None
    try:
        with _get_session_lock(session_id):
            response, viz = _agent.invoke(user_msg, session_id=session_id)
    except Exception as e:
        print(f"[app] Agent error: {e}", flush=True)
        response = (
            "Sorry, an error occurred while processing your request. "
            "Please try rephrasing the question."
        )

    return jsonify({"response": response, "viz": viz, "session_id": session_id})


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

    def _winner(sh, sa, home, away):
        # A draw scoreline must never report a team as the winner.
        if sh > sa: return home
        if sa > sh: return away
        return "Draw"

    result = None
    # Preferred: trained Logistic Regression squad-strength model.
    try:
        if (_match_predictor is not None and _match_predictor.has_team(match["home"])
                and _match_predictor.has_team(match["away"])):
            p = _match_predictor.predict(match["home"], match["away"])
            if p is not None:
                sh, sa = p["score"]
                result = {**match,
                    "predicted_winner": _winner(sh, sa, match["home"], match["away"]),
                    "confidence_pct": round(max(p["p_win"], p["p_draw"], p["p_loss"]) * 100),
                    "score": f"{sh}–{sa}",
                    "p_home": round(p["p_win"] * 100),
                    "p_draw": round(p["p_draw"] * 100),
                    "p_away": round(p["p_loss"] * 100),
                }
    except Exception as e:
        print(f"[app] widget LR predict failed: {e}", flush=True)

    # Fallback: hybrid strength model (also with the draw rule applied).
    if result is None:
        try:
            pred = _wc_featured.predict_match(match["home"], match["away"])
            sh, sa = pred["scoreline"]
            result = {**match,
                "predicted_winner": _winner(sh, sa, match["home"], match["away"]),
                "confidence_pct": round(max(pred["p_a"], pred["p_b"]) * 100),
                "score": f"{sh}–{sa}",
                "p_home": round(pred["p_a"] * 100),
                "p_draw": round(pred["p_draw"] * 100),
                "p_away": round(pred["p_b"] * 100),
            }
        except Exception:
            result = {**match, "predicted_winner": match["home"],
                      "confidence_pct": 60, "score": "2–1", "p_home": 55, "p_draw": 25, "p_away": 20}
    return jsonify(result)


@app.route("/api/league-tables")
def league_tables():
    """Top-5 of the big-5 leagues for the landing carousel (cached, prefetched)."""
    return jsonify({"leagues": _live_tables.get_tables(os.getenv("FOOTBALL_DATA_API_KEY", ""))})


@app.route("/healthz")
def healthz():
    """Health check for hosting platforms."""
    return jsonify({
        "ok": True,
        "players": int(len(_engine.df)),
        "world_cup_matches": int(len(_schedule)),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "version": _VERSION["label"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
