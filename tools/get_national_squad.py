"""
get_national_squad
Return a national team's official FIFA World Cup 2026 squad from the committed
squads file (data/world_cup_2026_squads.csv) — the source of truth for rosters.
"""

import re
import unicodedata
import datetime as dt

import pandas as pd
from langchain.tools import tool

_POS_ORDER = ["GK", "DF", "MF", "FW"]
_POS_LABEL = {"GK": "Goalkeepers", "DF": "Defenders", "MF": "Midfielders", "FW": "Forwards"}

# free-text alias -> squads-CSV team name
_ALIASES = {
    "south korea": "Korea Republic", "korea": "Korea Republic",
    "usa": "USA", "united states": "USA", "america": "USA",
    "ivory coast": "Côte D'Ivoire", "cote d'ivoire": "Côte D'Ivoire",
    "iran": "IR Iran", "czech republic": "Czechia", "czech": "Czechia",
    "cape verde": "Cabo Verde", "dr congo": "Congo DR", "congo": "Congo DR",
    "bosnia": "Bosnia And Herzegovina", "bosnia and herzegovina": "Bosnia And Herzegovina",
    "bosnia & herzegovina": "Bosnia And Herzegovina", "turkey": "Türkiye",
    "holland": "Netherlands",
}


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _age(dob: str) -> str:
    try:
        d = dt.datetime.strptime(str(dob).strip(), "%d/%m/%Y").date()
        ref = dt.date(2026, 6, 11)   # WC 2026 kickoff
        return str(ref.year - d.year - ((ref.month, ref.day) < (d.month, d.day)))
    except Exception:
        return "?"


def _clean_club(club: str) -> str:
    return re.sub(r"\s*\([A-Z]{3}\)\s*$", "", str(club)).strip()


def _disp_name(r) -> str:
    first = str(r.get("first_names", "") or "").strip()
    last = str(r.get("last_names", "") or "").strip().title()
    name = f"{first} {last}".strip()
    return name or str(r.get("player_name", "Unknown"))


def make_get_national_squad_tool(squads_df: pd.DataFrame):
    """Factory for the get_national_squad tool. squads_df = world_cup_2026_squads.csv."""

    available = squads_df is not None and not squads_df.empty
    if available:
        teams = sorted(squads_df["team"].dropna().unique())
        norm_map = {_norm(t): t for t in teams}

    def _resolve(name: str):
        q = name.strip()
        if not available:
            return None
        if q in norm_map.values():
            return q
        nq = _norm(q)
        if nq in norm_map:
            return norm_map[nq]
        alias = _ALIASES.get(q.lower())
        if alias:
            return alias
        # substring / contains
        for k, t in norm_map.items():
            if nq and (nq in k or k in nq):
                return t
        return None

    @tool
    def get_national_squad(team_name: str) -> str:
        """
        Get the OFFICIAL FIFA World Cup 2026 squad / roster / called-up players for a
        national team. This is the source of truth for "who is in [country]'s squad",
        "[country] World Cup roster", "is [player] in the squad", squad numbers, the
        head coach, and the clubs each player currently plays for.
        Use for any question about a national team's 2026 squad or player list.
        team_name: the national team (e.g. 'Brazil', 'USA', 'South Korea').
        """
        if not available:
            return "The World Cup 2026 squad list is not loaded on this server."
        team = _resolve(team_name)
        if team is None:
            sample = ", ".join(sorted(squads_df["team"].unique())[:8])
            return (f"I don't have a World Cup 2026 squad for '{team_name}'. "
                    f"The 48 qualified teams include: {sample}, …")

        grp = squads_df[squads_df["team"] == team]
        coach_first = str(grp["head_coach_first_names"].iloc[0]) if "head_coach_first_names" in grp else ""
        coach_last = str(grp["head_coach_last_names"].iloc[0]).title() if "head_coach_last_names" in grp else ""
        coach = f"{coach_first} {coach_last}".strip() or str(grp["head_coach"].iloc[0])

        lines = [f"**{team} — FIFA World Cup 2026 squad** ({len(grp)} players)",
                 f"_Head coach: {coach}_\n"]
        for pos in _POS_ORDER:
            block = grp[grp["position"] == pos].sort_values("squad_number")
            if block.empty:
                continue
            lines.append(f"**{_POS_LABEL[pos]}:**")
            for _, r in block.iterrows():
                num = r.get("squad_number", "")
                num = f"{int(num)}. " if pd.notna(num) else ""
                lines.append(f"- {num}{_disp_name(r)} ({_clean_club(r.get('club',''))}, age {_age(r.get('date_of_birth'))})")
            lines.append("")
        lines.append("🔍 Method: Official FIFA World Cup 2026 squad lists "
                     "(data/world_cup_2026_squads.csv) — committed roster data, not model output.")
        return "\n".join(lines)

    return get_national_squad
