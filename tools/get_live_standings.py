"""
Fetch live standings and fixtures from the football-data.org API.
"""

import os
import requests
from langchain.tools import tool

API_BASE = "https://api.football-data.org/v4"

# Competition-name mapping for the API (English + Hebrew aliases, lowercased).
COMPETITION_MAP = {
    "premier league":    "PL",
    "epl":               "PL",
    "english league":    "PL",
    "הליגה האנגלית":     "PL",
    "ליגה אנגלית":       "PL",
    "פריימר ליג":        "PL",
    "פרמייר ליג":        "PL",
    "ליגת העל האנגלית":  "PL",
    "אליפות אנגליה":     "PL",
    "la liga":           "PD",
    "הליגה הספרדית":     "PD",
    "לה ליגה":           "PD",
    "ליגה ספרדית":       "PD",
    "bundesliga":        "BL1",
    "בונדסליגה":         "BL1",
    "ליגה גרמנית":       "BL1",
    "serie a":           "SA",
    "סריה א":            "SA",
    "סדרה א":            "SA",
    "ליגה איטלקית":      "SA",
    "ligue 1":           "FL1",
    "ליגה צרפתית":       "FL1",
    "champions league":  "CL",
    "ליגת האלופות":      "CL",
    "world cup":         "WC",
    "world cup 2026":    "WC",
    "מונדיאל":           "WC",
    "גביע העולם":        "WC",
    "euro":              "EC",
    "euros":             "EC",
    "copa america":      "CLI",
    "eredivisie":        "DED",
    "primeira liga":     "PPL",
}


def _get_headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY", "")}


def make_get_live_standings_tool():
    """Factory for the live standings tool."""

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
                "FOOTBALL_DATA_API_KEY is not configured. "
                "Add it as an environment variable to fetch live data."
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
                f"Competition '{competition}' was not found. "
                f"Available competitions: {', '.join(COMPETITION_MAP.keys())}"
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

        headers = _get_headers()

        # Try standings first.
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
                    from viz import embed_viz
                    champion = table_data[0].get("team", {}).get("name", "") if table_data else ""
                    top_pts  = table_data[0].get("points", 0) if table_data else 0
                    played   = table_data[0].get("playedGames", 0) if table_data else 0
                    bottom   = table_data[-1].get("team", {}).get("name", "") if table_data else ""
                    # Plain text is a concise summary only — the viz card shows the full table.
                    # Avoid duplicating all 20 rows in both text and card.
                    summary = (
                        f"**{competition} | {season_str}**\n"
                        f"🏆 Leader: {champion} — {top_pts} pts ({played} played)\n"
                        f"🔻 Bottom: {bottom}\n"
                        f"\n🔍 Method: Live standings lookup via football-data.org API."
                    )
                    viz_rows = []
                    for row in table_data[:20]:
                        pos  = row.get("position", "")
                        team = row.get("team", {}).get("name", "")
                        p    = row.get("playedGames", 0)
                        w, d, l = row.get("won", 0), row.get("draw", 0), row.get("lost", 0)
                        gf, ga  = row.get("goalsFor", 0), row.get("goalsAgainst", 0)
                        pts  = row.get("points", 0)
                        viz_rows.append({
                            "pos": pos, "team": team, "p": p, "w": w, "d": d,
                            "l": l, "gf": gf, "ga": ga, "gd": gf - ga, "pts": pts,
                        })
                    viz = {
                        "type": "standings",
                        "competition": competition,
                        "season": season_str,
                        "champion": champion,
                        "table": viz_rows,
                    }
                    return embed_viz(summary, viz)

            # fallback: fixtures
            resp2 = requests.get(
                f"{API_BASE}/competitions/{comp_id}/matches",
                headers=headers, timeout=10,
                params={"status": "SCHEDULED", "limit": 10},
            )
            if resp2.status_code == 200:
                matches = resp2.json().get("matches", [])
                if matches:
                    lines = [f"**Upcoming matches - {competition}:**\n"]
                    for m in matches:
                        date  = m.get("utcDate", "")[:10]
                        home  = m.get("homeTeam", {}).get("name", "?")
                        away  = m.get("awayTeam", {}).get("name", "?")
                        lines.append(f"• {date}: {home} vs {away}")
                    lines.append("\n🔍 Method: Live fixture lookup via football-data.org API.")
                    return "\n".join(lines)

            return (
                f"Could not fetch data for {competition} (status: {resp.status_code})."
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

        except requests.RequestException as e:
            return (
                f"Network error while accessing the API: {e}"
                "\n\n🔍 Method: Live standings lookup via football-data.org API."
            )

    return get_live_standings
