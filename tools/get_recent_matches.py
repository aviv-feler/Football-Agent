"""
Recent finished club matches / head-to-head lookup via football-data.org.

The tool is intentionally conservative: it returns only verified match metadata
and scores from the API. football-data.org's regular match payload does not
include goal scorers, so the tool states that verified scorer data is unavailable
instead of guessing.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata

import requests
from langchain.tools import tool

API_BASE = "https://api.football-data.org/v4"

# Fast path for the clubs most likely to appear in class/demo questions.
TEAM_ID_ALIASES = {
    "arsenal": 57,
    "manchester city": 65,
    "man city": 65,
    "city": 65,
    "manchester united": 66,
    "man united": 66,
    "man utd": 66,
    "liverpool": 64,
    "chelsea": 61,
    "tottenham": 73,
    "spurs": 73,
    "barcelona": 81,
    "barca": 81,
    "real madrid": 86,
    "atletico madrid": 78,
    "bayern munich": 5,
    "bayern": 5,
    "borussia dortmund": 4,
    "dortmund": 4,
    "psg": 524,
    "paris saint germain": 524,
    "paris sg": 524,
    "inter": 108,
    "inter milan": 108,
    "juventus": 109,
    "milan": 98,
    "ac milan": 98,
    "napoli": 113,
    "roma": 100,
    # Hebrew aliases.
    "ארסנל": 57,
    "מנצסטר סיטי": 65,
    "מנצ'סטר סיטי": 65,
    "סיטי": 65,
    "מנצסטר יונייטד": 66,
    "מנצ'סטר יונייטד": 66,
    "ברצלונה": 81,
    "בארסה": 81,
    "ריאל מדריד": 86,
    "צלסי": 61,
    "צ'לסי": 61,
    "ליברפול": 64,
    "באיירן": 5,
    "דורטמונד": 4,
    "פסז": 524,
}

COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "CL"]
_team_cache: dict[str, tuple[float, dict[str, int]]] = {}
_CACHE_TTL = 6 * 3600


def _norm(value: str) -> str:
    s = str(value or "").strip().lower().replace("&", " and ")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9א-ת'\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY", "")}


def _load_team_ids() -> dict[str, int]:
    """Build a small in-memory team-name index from football-data competitions."""
    key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    now = time.time()
    cached = _team_cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    out = dict(TEAM_ID_ALIASES)
    if key:
        for comp in COMPETITIONS:
            try:
                r = requests.get(f"{API_BASE}/competitions/{comp}/teams", headers=_headers(), timeout=8)
                if r.status_code != 200:
                    continue
                for t in r.json().get("teams", []):
                    tid = t.get("id")
                    if not tid:
                        continue
                    for name in (t.get("name"), t.get("shortName"), t.get("tla")):
                        if name:
                            out[_norm(name)] = int(tid)
            except requests.RequestException:
                continue
    _team_cache[key] = (now, out)
    return out


def _resolve_team(name: str) -> int | None:
    q = _norm(name)
    if q in TEAM_ID_ALIASES:
        return TEAM_ID_ALIASES[q]
    for n, tid in TEAM_ID_ALIASES.items():
        if q and (q in n or n in q):
            return tid

    ids = _load_team_ids()
    if q in ids:
        return ids[q]
    for n, tid in ids.items():
        if q and (q in n or n in q):
            return tid
    return None


def _match_team_names(match: dict) -> tuple[str, str]:
    home = match.get("homeTeam", {}).get("name") or "?"
    away = match.get("awayTeam", {}).get("name") or "?"
    return home, away


def _is_vs(match: dict, opponent_id: int) -> bool:
    return (
        match.get("homeTeam", {}).get("id") == opponent_id
        or match.get("awayTeam", {}).get("id") == opponent_id
    )


def _fmt_match(match: dict) -> str:
    home, away = _match_team_names(match)
    ft = match.get("score", {}).get("fullTime", {})
    hs = ft.get("home")
    aw = ft.get("away")
    score = f"{hs}-{aw}" if hs is not None and aw is not None else "score unavailable"
    comp = match.get("competition", {}).get("name") or "Competition"
    date = (match.get("utcDate") or "")[:10] or "date unavailable"
    return f"{date} | {comp} | {home} {score} {away}"


def lookup_team_matches(team: str, opponent: str = "", limit: int = 2, status: str = "FINISHED") -> tuple[str, list[dict]]:
    """Return formatted text plus raw selected matches for agent-side routing."""
    if not os.getenv("FOOTBALL_DATA_API_KEY", ""):
        return (
            "I don't have access to verified historical data for that specific match."
            "\n\n🔍 Method: Verified historical match lookup via football-data.org API.",
            [],
        )

    team_id = _resolve_team(team)
    if team_id is None:
        return (
            f"I don't have access to verified historical data for '{team}'."
            "\n\n🔍 Method: Verified historical match lookup via football-data.org API.",
            [],
        )

    opp_id = _resolve_team(opponent) if opponent else None
    if opponent and opp_id is None:
        return (
            f"I don't have access to verified historical data for '{opponent}'."
            "\n\n🔍 Method: Verified historical match lookup via football-data.org API.",
            [],
        )

    status = (status or "FINISHED").upper()
    try:
        r = requests.get(
            f"{API_BASE}/teams/{team_id}/matches",
            headers=_headers(),
            params={"status": status, "limit": 100},
            timeout=10,
        )
        if r.status_code != 200:
            return (
                f"I don't have access to verified historical data for that specific match "
                f"(API status {r.status_code})."
                "\n\n🔍 Method: Verified historical match lookup via football-data.org API.",
                [],
            )
        matches = r.json().get("matches", [])
    except requests.RequestException as e:
        return (
            f"I don't have access to verified historical data right now: {e}"
            "\n\n🔍 Method: Verified historical match lookup via football-data.org API.",
            [],
        )

    if opp_id is not None:
        matches = [m for m in matches if _is_vs(m, opp_id)]

    reverse = status == "FINISHED"
    matches = sorted(matches, key=lambda m: m.get("utcDate", ""), reverse=reverse)
    n = max(1, min(int(limit or 2), 10))
    selected = matches[:n]
    if not selected:
        scope = f"{team} vs {opponent}" if opponent else team
        word = "historical" if status == "FINISHED" else "scheduled"
        return (
            f"I don't have access to verified {word} data for {scope}."
            "\n\n🔍 Method: Verified match lookup via football-data.org API.",
            [],
        )

    if status == "FINISHED":
        title = f"Verified recent H2H: {team} vs {opponent}" if opponent else f"Verified recent matches: {team}"
    else:
        title = f"Verified upcoming match: {team}"
    lines = [f"**{title}**\n"]
    for i, m in enumerate(selected, 1):
        lines.append(f"{i}. {_fmt_match(m)}")
    if status == "FINISHED":
        lines.append("\nI have the verified scoreline, but I do not have a verified scorer source for these matches.")
    lines.append("\n🔍 Method: Verified match lookup via football-data.org API.")
    return "\n".join(lines), selected


def make_get_recent_matches_tool():
    @tool
    def get_recent_matches(team: str, opponent: str = "", limit: int = 2, status: str = "FINISHED") -> str:
        """
        Fetch verified finished matches for a club from football-data.org.
        Use for historical results, latest match, last N matches, and recent
        head-to-head questions. If opponent is provided, returns recent H2H only.
        Does not guess scorers; says scorer data is unavailable when not present.
        """
        text, _ = lookup_team_matches(team, opponent=opponent, limit=limit, status=status)
        return text

    return get_recent_matches
