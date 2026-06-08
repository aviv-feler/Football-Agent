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

from agent import load_resources, build_agent, DEMO_QUERIES

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
# Background demo warm-up state. Running all ~26 demo questions synchronously inside the
# /warmup request would blow past gunicorn's --timeout and get the single worker SIGKILLed
# (→ "Internal Server Error", nothing cached). So /warmup starts a daemon thread and the
# progress is polled via /warmup/status.
_warmup_lock = threading.Lock()
_warmup_state: dict = {
    "running": False, "done": False, "started_at": None,
    "elapsed": 0.0, "cached": 0, "total": len(DEMO_QUERIES),
    "results": None, "error": None,
}
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


def _public_warmup_state() -> dict:
    """Snapshot of the warm-up state, with live progress, safe to return as JSON."""
    with _warmup_lock:
        st = dict(_warmup_state)
    st["cache_size"] = len(_agent.response_cache)   # live: answers cached so far
    if st["running"] and st["started_at"]:
        st["elapsed"] = round(_time.time() - st["started_at"], 1)
    if not st["done"]:
        st.pop("results", None)                     # the full report only once finished
    return st


def _run_warmup_bg():
    """Run the (slow) demo warm-up off the request thread so it can't trip the gunicorn
    timeout. Always clears `running` so a failure can't wedge the state forever."""
    t0 = _time.time()
    error = None
    report = []
    try:
        report = _agent.warmup()
    except Exception as e:                           # never let the thread die silently
        error = str(e)
        print(f"[app] Warmup thread error: {e}", flush=True)
    ok = sum(1 for r in report if r.get("ok"))
    with _warmup_lock:
        _warmup_state.update(running=False, done=True, elapsed=round(_time.time() - t0, 1),
                             cached=ok, results=report, error=error)
    print(f"[app] Warmup finished in {_time.time()-t0:.1f}s (cached {ok}/{len(report)}).", flush=True)


@app.route("/warmup", methods=["GET", "POST"])
def warmup():
    """Pre-compute + cache the fixed demo questions so they answer INSTANTLY on stage.
    Call this once before the presentation, then poll /warmup/status until done=true.

    Runs in the BACKGROUND: executing ~26 demo queries synchronously here would exceed
    gunicorn's --timeout and get the worker killed (the old behaviour → 500). This starts a
    daemon thread and returns immediately. Pass ?force=1 to re-run after it has finished."""
    force = (request.args.get("force") or "").lower() in ("1", "true", "yes")
    with _warmup_lock:
        if _warmup_state["running"]:
            running_state = True
            already_done = False
        else:
            already_done = _warmup_state["done"] and not force
            running_state = False
            if not already_done:
                _warmup_state.update(running=True, done=False, started_at=_time.time(),
                                     elapsed=0.0, results=None, error=None,
                                     cached=len(_agent.response_cache))
    if running_state:
        return jsonify({"status": "already_running", **_public_warmup_state()}), 202
    if already_done:
        return jsonify({"status": "already_done", **_public_warmup_state()})
    threading.Thread(target=_run_warmup_bg, name="warmup", daemon=True).start()
    return jsonify({
        "status": "started",
        "message": "Warm-up running in the background. Poll /warmup/status until \"done\": true.",
        "total": _warmup_state["total"],
        "status_url": "/warmup/status",
    }), 202


@app.route("/warmup/status")
def warmup_status():
    """Progress/result of the background warm-up started by /warmup."""
    return jsonify(_public_warmup_state())


@app.route("/healthz")
def healthz():
    """Health check for hosting platforms."""
    return jsonify({
        "ok": True,
        "players": int(len(_engine.df)),
        "world_cup_matches": int(len(_schedule)),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "demo_cached": len(_agent.response_cache),
        "version": _VERSION["label"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
