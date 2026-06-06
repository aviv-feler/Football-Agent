"""
Compute final league standings from the local games.csv dataset.
Covers all competitions in data/games.csv, up to season 2025 (= 2025/26 campaign).
Season label convention in the dataset: season N = the campaign that starts in year N-1
and ends in year N (e.g. season=2025 → 2024/25 or 2025/26 depending on competition).
For European leagues that run Aug→May, season=2025 always means 2025/26.
"""

import os
import pandas as pd
from langchain.tools import tool

_GAMES_CSV = os.path.join("data", "games.csv")

# Map human-readable names → competition_id values that appear in games.csv
COMP_ALIASES: dict[str, str] = {
    # English
    "premier league": "GB1", "epl": "GB1", "english premier league": "GB1",
    "fa cup": "FAC",
    "efl cup": "CGB", "league cup": "CGB", "carabao cup": "CGB",
    # Spanish
    "la liga": "ES1", "laliga": "ES1", "primera division": "ES1",
    "copa del rey": "CDR",
    # German
    "bundesliga": "L1", "german bundesliga": "L1",
    "dfb pokal": "DFB", "dfb-pokal": "DFB",
    # Italian
    "serie a": "IT1", "serie-a": "IT1",
    "coppa italia": "CIT", "italy cup": "CIT",
    # French
    "ligue 1": "FR1", "ligue1": "FR1",
    # Dutch
    "eredivisie": "NL1",
    # Portuguese
    "primeira liga": "PO1", "liga portugal": "PO1",
    # Scottish
    "scottish premiership": "SC1",
    # Belgian
    "jupiler pro league": "BE1", "belgian first division": "BE1",
    # Brazilian
    "brasileirao": "BRA1", "campeonato brasileiro": "BRA1",
    # Champions League / European
    "champions league": "CL", "ucl": "CL", "uefa champions league": "CL",
    "europa league": "EL", "uel": "EL",
    "conference league": "UCOL",
    # International
    "world cup": "FIWC",
    # Hebrew
    "פריימר ליג": "GB1", "פרמייר ליג": "GB1", "פרמיר ליג": "GB1",
    "ליגה האנגלית": "GB1", "ליגת פריימר": "GB1",
    "לה ליגה": "ES1", "לה-ליגה": "ES1", "ליגה הספרדית": "ES1",
    "בונדסליגה": "L1", "ליגה הגרמנית": "L1",
    "סרייה א": "IT1", "סריה א": "IT1", "ליגה האיטלקית": "IT1",
    "ליג 1": "FR1", "ליג'1": "FR1", "ליגה הצרפתית": "FR1",
    "ארדיביזיה": "NL1", "ליגה ההולנדית": "NL1",
    "ליגת האלופות": "CL", "ליגת אלופות": "CL",
    "ליגה אירופית": "EL", "ליגה קונפרנס": "UCOL",
}

# Which competitions are knockout-only (no league table to compute)
_KNOCKOUT_ONLY = {"FAC", "CGB", "CDR", "DFB", "CIT", "FIWC", "AFAC", "AFCN",
                  "COPA", "EURO", "CLQ", "ELQ", "ECLQ", "USC", "KLUB"}

# Season display helper: season label → human-readable string
def _season_label(season: int, comp_id: str) -> str:
    """Most European leagues: season N started in N-1 (e.g. 2025 → 2025/26)."""
    summer_leagues = {"BRA1", "ARG1", "MLS1", "AUS1", "MEX1"}
    if comp_id in summer_leagues:
        return str(season)
    return f"{season - 1}/{str(season)[-2:]}"


def _resolve_competition(query: str) -> tuple[str | None, str | None]:
    """Return (competition_id, display_name) for a user query, or (None, None)."""
    q = query.strip().lower()
    # Exact alias match
    if q in COMP_ALIASES:
        cid = COMP_ALIASES[q]
        return cid, query.title()
    # Partial alias match
    for alias, cid in COMP_ALIASES.items():
        if alias in q or q in alias:
            return cid, alias.title()
    # Direct competition_id match (e.g. "GB1")
    q_upper = query.strip().upper()
    if len(q_upper) <= 5:
        return q_upper, q_upper
    return None, None


def _compute_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a league table from a slice of games.csv."""
    rows: list[dict] = []
    for _, g in df.iterrows():
        h = g["home_club_name"]
        a = g["away_club_name"]
        try:
            hg = int(g["home_club_goals"])
            ag = int(g["away_club_goals"])
        except (ValueError, TypeError):
            continue  # skip matches with missing scores
        if h:
            rows.append({"team": h, "gf": hg, "ga": ag,
                         "w": int(hg > ag), "d": int(hg == ag), "l": int(hg < ag)})
        if a:
            rows.append({"team": a, "gf": ag, "ga": hg,
                         "w": int(ag > hg), "d": int(ag == hg), "l": int(ag < hg)})

    if not rows:
        return pd.DataFrame()

    tbl = (
        pd.DataFrame(rows)
        .groupby("team", as_index=False)
        .agg(p=("w", "count"), w=("w", "sum"), d=("d", "sum"),
             l=("l", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    )
    tbl["pts"] = tbl["w"] * 3 + tbl["d"]
    tbl["gd"] = tbl["gf"] - tbl["ga"]
    tbl = tbl.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
    tbl.index += 1
    return tbl


def make_get_historical_standings_tool():
    """Factory — returns the get_historical_standings LangChain tool."""

    @tool
    def get_historical_standings(query: str) -> str:
        """
        Return the final league standings (champion, top-4, full table) for any
        competition and season from the local historical dataset.
        Use this for questions like:
          - "who won the Premier League last season / in 2025/26?"
          - "who is the Premier League champion?"
          - "final La Liga table 2024/25"
          - "did Arsenal win the league?"
        Input format: "<competition> [season]"
        Examples: "Premier League", "Premier League 2025", "La Liga 2024", "Bundesliga"
        Omitting the season returns the most recent completed season.
        """
        if not os.path.exists(_GAMES_CSV):
            return "Local match data (games.csv) not found.\n\n🔍 Method: Historical standings from local games.csv dataset."

        games = pd.read_csv(_GAMES_CSV, low_memory=False)

        # Parse optional season number from the end of the query
        import re
        season_match = re.search(r"\b(20\d{2})\b", query)
        requested_season: int | None = int(season_match.group(1)) if season_match else None
        comp_query = re.sub(r"\b20\d{2}\b", "", query).strip(" ,/-")

        comp_id, display_name = _resolve_competition(comp_query)
        if comp_id is None:
            available = ", ".join(sorted({v for v in COMP_ALIASES.values()}))
            return (
                f"Competition '{comp_query}' not recognised. "
                f"Try names like: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, "
                f"Champions League, Eredivisie.\n\n"
                f"🔍 Method: Historical standings from local games.csv dataset."
            )

        if comp_id in _KNOCKOUT_ONLY:
            return (
                f"{display_name} is a knockout competition — no league table exists. "
                f"Ask about a league competition instead.\n\n"
                f"🔍 Method: Historical standings from local games.csv dataset."
            )

        subset = games[games["competition_id"] == comp_id]
        if subset.empty:
            return (
                f"No data found for {display_name} (id={comp_id}) in the local dataset.\n\n"
                f"🔍 Method: Historical standings from local games.csv dataset."
            )

        available_seasons = sorted(subset["season"].dropna().unique().astype(int))

        if requested_season is not None:
            if requested_season not in available_seasons:
                # Try to be helpful: map 2026 → 2025 (season label convention)
                alt = requested_season - 1
                if alt in available_seasons:
                    requested_season = alt
                else:
                    return (
                        f"Season {requested_season} not found for {display_name}. "
                        f"Available seasons: {', '.join(str(s) for s in available_seasons[-5:])}.\n\n"
                        f"🔍 Method: Historical standings from local games.csv dataset."
                    )
            season = requested_season
        else:
            season = available_seasons[-1]  # most recent

        season_df = subset[subset["season"] == season].copy()
        tbl = _compute_standings(season_df)

        if tbl.empty:
            return (
                f"Could not compute standings for {display_name} "
                f"season {_season_label(season, comp_id)} — match scores may be missing.\n\n"
                f"🔍 Method: Historical standings from local games.csv dataset."
            )

        champion = tbl.iloc[0]["team"]
        season_str = _season_label(season, comp_id)
        n_matches = len(season_df)

        lines = [
            f"**{display_name} — {season_str} final standings** "
            f"({n_matches} matches)\n",
            f"**CHAMPION: {champion}**\n",
            f"{'#':<3} {'Team':<32} {'P':<4} {'W':<4} {'D':<4} {'L':<4} "
            f"{'GF':<5} {'GA':<5} {'GD':<6} {'Pts'}",
            "-" * 72,
        ]
        for pos, row in tbl.iterrows():
            lines.append(
                f"{pos:<3} {row['team'][:31]:<32} {int(row['p']):<4} {int(row['w']):<4} "
                f"{int(row['d']):<4} {int(row['l']):<4} {int(row['gf']):<5} "
                f"{int(row['ga']):<5} {int(row['gd']):<+6} {int(row['pts'])}"
            )

        lines.append(
            f"\n🔍 Method: Historical standings computed from local games.csv dataset "
            f"(Transfermarkt / open-football data, season label {season})."
        )
        return "\n".join(lines)

    return get_historical_standings
