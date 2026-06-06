"""
live_tables.py
Cached top-5 standings for the big-5 European leagues (football-data.org), used by
the landing-page league-table carousel.

Speed: results are cached in-process (6h TTL) and prefetched in a background thread
at startup, so the /api/league-tables endpoint serves instantly and never blocks the
UI. Team crests + the league emblem come straight from the API response (no separate
logo dataset needed).
"""

from __future__ import annotations

import os
import time
import threading

import requests

API_BASE = "https://api.football-data.org/v4"

# (football-data.org competition code, display name, prompt phrase, local logo slug)
LEAGUES = [
    ("PL",  "Premier League", "Premier League", "premier-league"),
    ("PD",  "La Liga",        "La Liga",        "la-liga"),
    ("SA",  "Serie A",        "Serie A",        "serie-a"),
    ("BL1", "Bundesliga",     "Bundesliga",     "bundesliga"),
    ("FL1", "Ligue 1",        "Ligue 1",        "ligue-1"),
]

TTL_SECONDS = 6 * 3600
TOP_N = 5

_cache: dict = {"data": None, "ts": 0.0}
_lock = threading.Lock()


def _fetch_one(code: str, name: str, prompt: str, slug: str, key: str) -> dict | None:
    r = requests.get(f"{API_BASE}/competitions/{code}/standings",
                     headers={"X-Auth-Token": key}, timeout=10)
    if r.status_code != 200:
        return None
    d = r.json()
    comp = d.get("competition", {})
    standings = d.get("standings", [])
    rows = next((s.get("table", []) for s in standings if s.get("type") == "TOTAL"),
                standings[0].get("table", []) if standings else [])
    table = []
    for row in rows[:TOP_N]:
        t = row.get("team", {})
        table.append({
            "pos":    row.get("position"),
            "team":   t.get("shortName") or t.get("name", ""),
            "crest":  t.get("crest", ""),
            "points": row.get("points", 0),
        })
    if not table:
        return None
    return {"code": code, "name": name, "prompt": prompt,
            "logo": f"/static/logos/competitions/{slug}.png",
            "emblem": comp.get("emblem", ""), "table": table}


def get_tables(key: str | None = None, force: bool = False) -> list[dict]:
    """Return cached league tables, refreshing past the TTL (or when forced)."""
    key = key if key is not None else os.getenv("FOOTBALL_DATA_API_KEY", "")
    with _lock:
        fresh = _cache["data"] is not None and (time.time() - _cache["ts"]) < TTL_SECONDS
        if fresh and not force:
            return _cache["data"]
    if not key:
        return _cache["data"] or []
    out = []
    for code, name, prompt, slug in LEAGUES:
        try:
            t = _fetch_one(code, name, prompt, slug, key)
            if t:
                out.append(t)
        except requests.RequestException:
            pass
    if out:
        with _lock:
            _cache["data"] = out
            _cache["ts"] = time.time()
        return out
    # On total failure keep any stale data we already had.
    return _cache["data"] or []


def prefetch(key: str | None = None) -> None:
    """Warm the cache in a background thread so startup isn't blocked."""
    key = key if key is not None else os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not key:
        return
    threading.Thread(target=lambda: get_tables(key, force=True),
                     daemon=True, name="league-tables-prefetch").start()
