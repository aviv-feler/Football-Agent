"""
tools/world_cup.py
כלי למידע על מונדיאל 2026 מתוך לוח המשחקים הרשמי (fwc26_match_schedule_agent.csv).
תומך בשאילתות לפי נבחרת, בית, שלב, עיר ותאריך.
"""

import pandas as pd
from langchain.tools import tool


def make_world_cup_tool(schedule: pd.DataFrame):
    """Factory – schedule הוא DataFrame של לוח המשחקים."""

    @tool
    def world_cup_info(query: str) -> str:
        """
        Get FIFA World Cup 2026 schedule information: fixtures by team, by group
        (e.g. 'Group A'), by stage ('Group Stage', 'Final', 'round of 16'), by city,
        or upcoming matches. Examples: 'Brazil matches', 'Group D', 'World Cup final',
        'matches in Mexico City'. Input is the query as plain text.
        """
        q = query.lower().strip()
        sched = schedule.copy()

        # ── לפי בית (Group X) ──
        import re
        gmatch = re.search(r"group\s+([a-l])\b|בית\s+([a-l])", q)
        if gmatch:
            grp = (gmatch.group(1) or gmatch.group(2)).upper()
            res = sched[sched["group"].astype(str).str.upper() == grp]
            return _format(res, f"בית {grp} – מונדיאל 2026")

        # ── לפי שלב ──
        stage_map = {
            "final": "Final", "גמר": "Final",
            "semi": "Semi-finals", "חצי גמר": "Semi-finals",
            "quarter": "Quarter-finals", "רבע גמר": "Quarter-finals",
            "round of 16": "Round of 16", "שמינית": "Round of 16",
            "group stage": "Group Stage", "בתים": "Group Stage",
        }
        for kw, stage_val in stage_map.items():
            if kw in q:
                res = sched[sched["stage"].str.contains(stage_val, case=False, na=False)]
                return _format(res, f"{stage_val} – מונדיאל 2026")

        # ── לפי עיר ──
        for city in sched["city"].dropna().unique():
            if isinstance(city, str) and city.lower() in q:
                res = sched[sched["city"] == city]
                return _format(res, f"משחקים ב-{city}")

        # ── לפי נבחרת ──
        for col in ["team1_name", "team2_name"]:
            for team in sched[col].dropna().unique():
                if isinstance(team, str) and team.lower() in q:
                    res = sched[(sched["team1_name"] == team) | (sched["team2_name"] == team)]
                    return _format(res, f"משחקי {team} במונדיאל 2026")

        # ── ברירת מחדל: המשחקים הראשונים ──
        return _format(sched.head(10), "משחקי הפתיחה של מונדיאל 2026")

    return world_cup_info


def _format(res: pd.DataFrame, title: str) -> str:
    if res.empty:
        return (
            f"{title}: לא נמצאו משחקים תואמים."
            "\n\n🔍 Method: Schedule lookup from the FIFA World Cup 2026 fixture CSV."
        )
    lines = [f"**{title}** ({len(res)} משחקים):\n"]
    for _, m in res.head(20).iterrows():
        stage = m.get("stage", "")
        grp   = m.get("group", "")
        grp_s = f" (בית {grp})" if isinstance(grp, str) and grp.strip() and grp.lower() != "nan" else ""
        lines.append(
            f"• משחק {m.get('match_number','?')} | {m.get('date','?')} {m.get('kickoff_time_et','')} ET "
            f"| {m.get('team1_name','?')} vs {m.get('team2_name','?')} "
            f"| {m.get('city','?')}, {m.get('venue_country','?')} | {stage}{grp_s}"
        )
    lines.append("\n🔍 Method: Schedule lookup from the FIFA World Cup 2026 fixture CSV.")
    return "\n".join(lines)
