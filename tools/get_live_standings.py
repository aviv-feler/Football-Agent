"""
tools/get_live_standings.py
כלי לשליפת טבלאות ותוצאות חיות מ-football-data.org API
"""

import os
import requests
from langchain.tools import tool

API_BASE = "https://api.football-data.org/v4"

# מיפוי שמות תחרויות ל-ID ב-API
COMPETITION_MAP = {
    "premier league":    "PL",
    "la liga":           "PD",
    "bundesliga":        "BL1",
    "serie a":           "SA",
    "ligue 1":           "FL1",
    "champions league":  "CL",
    "world cup":         "WC",
    "world cup 2026":    "WC",
    "euro":              "EC",
    "euros":             "EC",
    "copa america":      "CLI",
    "eredivisie":        "DED",
    "primeira liga":     "PPL",
}


def _get_headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY", "")}


def make_get_live_standings_tool():
    """Factory – יוצר את הכלי לשליפת טבלאות חיות."""

    @tool
    def get_live_standings(competition: str) -> str:
        """
        Fetch live standings or upcoming fixtures for a football competition.
        Examples: 'Premier League', 'World Cup 2026', 'Champions League', 'La Liga'.
        Input is the competition name as a string.
        """
        key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not key:
            return (
                "FOOTBALL_DATA_API_KEY לא הוגדר. "
                "הוסף את המפתח כמשתנה סביבה כדי לקבל נתונים חיים."
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

        comp_lower = competition.strip().lower()
        comp_id    = None
        for name, cid in COMPETITION_MAP.items():
            if name in comp_lower or comp_lower in name:
                comp_id = cid
                break

        if not comp_id:
            return (
                f"לא נמצאה תחרות בשם '{competition}'. "
                f"תחרויות זמינות: {', '.join(COMPETITION_MAP.keys())}"
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

        headers = _get_headers()

        # נסיון לשלוף טבלה
        try:
            resp = requests.get(
                f"{API_BASE}/competitions/{comp_id}/standings",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                data    = resp.json()
                season  = data.get("season", {})
                season_str = f"{season.get('startDate','?')} – {season.get('endDate','?')}"
                standings = data.get("standings", [])
                table_data = next(
                    (s.get("table", []) for s in standings if s.get("type") == "TOTAL"),
                    []
                )

                if not table_data:
                    table_data = standings[0].get("table", []) if standings else []

                if table_data:
                    lines = [f"**טבלת {competition} | עונה: {season_str}**\n",
                             f"{'#':<3} {'קבוצה':<25} {'מ':<4} {'נ':<4} {'ת':<4} {'ה':<4} {'גבצ':<6} {'נגד':<6} {'נק':<5}"]
                    lines.append("-" * 60)
                    for row in table_data[:20]:
                        pos   = row.get("position", "")
                        team  = row.get("team", {}).get("name", "")[:24]
                        played= row.get("playedGames", 0)
                        won   = row.get("won", 0)
                        draw  = row.get("draw", 0)
                        lost  = row.get("lost", 0)
                        gf    = row.get("goalsFor", 0)
                        ga    = row.get("goalsAgainst", 0)
                        pts   = row.get("points", 0)
                        lines.append(f"{pos:<3} {team:<25} {played:<4} {won:<4} {draw:<4} {lost:<4} {gf:<6} {ga:<6} {pts:<5}")
                    lines.append("\n🔍 Method: Live standings lookup via football-data.org API.")
                    return "\n".join(lines)

            # fallback: fixtures
            resp2 = requests.get(
                f"{API_BASE}/competitions/{comp_id}/matches",
                headers=headers, timeout=10,
                params={"status": "SCHEDULED", "limit": 10},
            )
            if resp2.status_code == 200:
                matches = resp2.json().get("matches", [])
                if matches:
                    lines = [f"**משחקים קרובים – {competition}:**\n"]
                    for m in matches:
                        date  = m.get("utcDate", "")[:10]
                        home  = m.get("homeTeam", {}).get("name", "?")
                        away  = m.get("awayTeam", {}).get("name", "?")
                        lines.append(f"• {date}: {home} vs {away}")
                    lines.append("\n🔍 Method: Live fixture lookup via football-data.org API.")
                    return "\n".join(lines)

            return (
                f"לא הצלחתי לשלוף נתונים עבור {competition} (קוד: {resp.status_code})."
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

        except requests.RequestException as e:
            return (
                f"שגיאת רשת בעת גישה ל-API: {e}"
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

    return get_live_standings
