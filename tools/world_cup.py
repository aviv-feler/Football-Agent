"""
World Cup 2026 schedule tool based on fwc26_match_schedule_agent.csv.
Supports queries by team, group, stage, city, and date.
"""

import pandas as pd
from langchain.tools import tool


def make_world_cup_tool(schedule: pd.DataFrame):
    """Factory for the World Cup schedule tool."""

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

        import re
        gmatch = re.search(r"group\s+([a-l])\b|בית\s+([a-l])", q)
        if gmatch:
            grp = (gmatch.group(1) or gmatch.group(2)).upper()
            res = sched[sched["group"].astype(str).str.upper() == grp]
            return _format(res, f"Group {grp} - World Cup 2026")

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
                return _format(res, f"{stage_val} - World Cup 2026")

        for city in sched["city"].dropna().unique():
            if isinstance(city, str) and city.lower() in q:
                res = sched[sched["city"] == city]
                return _format(res, f"Matches in {city}")

        for col in ["team1_name", "team2_name"]:
            for team in sched[col].dropna().unique():
                if isinstance(team, str) and team.lower() in q:
                    res = sched[(sched["team1_name"] == team) | (sched["team2_name"] == team)]
                    return _format(res, f"{team} matches - World Cup 2026")

        return _format(sched.head(10), "Opening matches - World Cup 2026")

    return world_cup_info


def _format(res: pd.DataFrame, title: str) -> str:
    if res.empty:
        return f"{title}: no matching matches were found."
    lines = [f"**{title}** ({len(res)} matches):\n"]
    for _, m in res.head(20).iterrows():
        stage = m.get("stage", "")
        grp   = m.get("group", "")
        grp_s = f" (Group {grp})" if isinstance(grp, str) and grp.strip() and grp.lower() != "nan" else ""
        lines.append(
            f"- Match {m.get('match_number','?')} | {m.get('date','?')} {m.get('kickoff_time_et','')} ET "
            f"| {m.get('team1_name','?')} vs {m.get('team2_name','?')} "
            f"| {m.get('city','?')}, {m.get('venue_country','?')} | {stage}{grp_s}"
        )
    return "\n".join(lines)
