"""
Fetch LIVE current-season top scorers from the football-data.org API.
This is real, fresh data (not the local dataset) for "who is the top scorer right now"
style questions.
"""

import os
import requests
from langchain.tools import tool

API_BASE = "https://api.football-data.org/v4"

COMPETITION_MAP = {
    "premier league":   "PL",
    "epl":              "PL",
    "english league":   "PL",
    "הליגה האנגלית":    "PL",
    "ליגה אנגלית":      "PL",
    "פריימר ליג":       "PL",
    "פרמייר ליג":       "PL",
    "la liga":          "PD",
    "הליגה הספרדית":    "PD",
    "לה ליגה":          "PD",
    "bundesliga":       "BL1",
    "בונדסליגה":        "BL1",
    "serie a":          "SA",
    "סריה א":           "SA",
    "סדרה א":           "SA",
    "ligue 1":          "FL1",
    "ליגה צרפתית":      "FL1",
    "champions league": "CL",
    "ליגת האלופות":     "CL",
    "eredivisie":       "DED",
    "primeira liga":    "PPL",
    "championship":     "ELC",
}


def make_get_top_scorers_tool():
    """Factory for the live top-scorers tool."""

    @tool
    def get_top_scorers(competition: str) -> str:
        """
        Get the LIVE current-season top scorers for a league from the football-data.org API.
        IMPORTANT: this tool only has CURRENT-SEASON data. It cannot return last season's
        top scorers. If the user asks for 'last season', answer with the current season and
        note that only current-season data is available.
        Use this for "top scorer / golden boot / who scored the most this season/last season"
        questions. Examples: 'Premier League', 'La Liga', 'Serie A', 'Bundesliga'.
        Input is the competition name as a string.
        """
        key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not key:
            return (
                "FOOTBALL_DATA_API_KEY is not configured, so live top scorers are unavailable. "
                "Ask for top scorers from the local dataset instead."
                "\n\n🔍 Method: Live top-scorer lookup via football-data.org API."
            )

        comp_lower = competition.strip().lower()
        comp_id = next((cid for name, cid in COMPETITION_MAP.items()
                        if name in comp_lower or comp_lower in name), None)
        if not comp_id:
            return (
                f"Competition '{competition}' is not available for live scorers. "
                f"Available: {', '.join(sorted(set(COMPETITION_MAP.keys())))}."
                "\n\n🔍 Method: Live top-scorer lookup via football-data.org API."
            )

        try:
            resp = requests.get(
                f"{API_BASE}/competitions/{comp_id}/scorers",
                headers={"X-Auth-Token": key},
                params={"limit": 10},
                timeout=10,
            )
            if resp.status_code != 200:
                return (
                    f"Could not fetch live scorers for {competition} (status {resp.status_code})."
                    "\n\n🔍 Method: Live top-scorer lookup via football-data.org API."
                )
            scorers = resp.json().get("scorers", [])
            if not scorers:
                return (
                    f"No live scorer data returned for {competition} yet."
                    "\n\n🔍 Method: Live top-scorer lookup via football-data.org API."
                )
            lines = [f"**Live top scorers — {competition} (current season):**\n"]
            for rank, s in enumerate(scorers[:10], 1):
                player = s.get("player", {}).get("name", "?")
                team = s.get("team", {}).get("name", "?")
                goals = s.get("goals", 0) or 0
                assists = s.get("assists")
                tail = f" | assists {assists}" if assists is not None else ""
                lines.append(f"{rank}. {player} ({team}) — {goals} goals{tail}")
            lines.append("\n🔍 Method: Live top-scorer lookup via football-data.org API.")
            return "\n".join(lines)
        except requests.RequestException as e:
            return (
                f"Network error fetching live scorers: {e}"
                "\n\n🔍 Method: Live top-scorer lookup via football-data.org API."
            )

    return get_top_scorers
