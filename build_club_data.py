"""
build_club_data.py
Parse the football-data.co.uk league CSVs (top-5 leagues, seasons 16-17..25-26) into
one committed tidy file the club prediction model reads.

Output: data/club_matches.csv  (one row per match: date, season, league, teams, goals,
result, shots, and pre-match odds where available).

Input: data/<league folder>/*.csv  (E0/D1/F1/I1/SP1)
"""

import os
import re
import glob
import pandas as pd

OUTPUT_CSV = "data/club_matches.csv"

DIV_TO_LEAGUE = {
    "E0": "Premier League", "D1": "Bundesliga", "F1": "Ligue 1",
    "I1": "Serie A", "SP1": "La Liga",
}
# Preferred odds columns in priority order (different seasons carry different sets).
ODDS_SETS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("BbAvH", "BbAvD", "BbAvA")]


def log(msg: str) -> None:
    print(f"[club_data] {msg}", flush=True)


def season_from_path(path: str) -> str:
    m = re.search(r"(\d{2})\s*-\s*(\d{2})", os.path.basename(path))
    if not m:
        return ""
    return f"20{m.group(1)}-{m.group(2)}"


def main() -> None:
    files = sorted(glob.glob("data/*/*.csv"))
    if not files:
        log("ERROR: no league CSVs found under data/<league>/.")
        raise SystemExit(1)

    rows = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}.issubset(df.columns):
            log(f"  skip (missing core cols): {f}")
            continue

        odds = next((s for s in ODDS_SETS if all(c in df.columns for c in s)), None)
        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce"),
            "season": season_from_path(f),
            "div": df["Div"] if "Div" in df.columns else "",
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
            "fthg": pd.to_numeric(df["FTHG"], errors="coerce"),
            "ftag": pd.to_numeric(df["FTAG"], errors="coerce"),
            "ftr": df["FTR"].astype(str).str.strip(),
            "home_shots": pd.to_numeric(df.get("HS"), errors="coerce"),
            "away_shots": pd.to_numeric(df.get("AS"), errors="coerce"),
            "home_sot": pd.to_numeric(df.get("HST"), errors="coerce"),
            "away_sot": pd.to_numeric(df.get("AST"), errors="coerce"),
            "odds_home": pd.to_numeric(df[odds[0]], errors="coerce") if odds else pd.NA,
            "odds_draw": pd.to_numeric(df[odds[1]], errors="coerce") if odds else pd.NA,
            "odds_away": pd.to_numeric(df[odds[2]], errors="coerce") if odds else pd.NA,
        })
        out["league"] = out["div"].map(DIV_TO_LEAGUE).fillna(out["div"])
        rows.append(out)

    matches = pd.concat(rows, ignore_index=True)
    matches = matches.dropna(subset=["home", "away", "fthg", "ftag", "date"])
    matches = matches[matches["home"].str.len() > 0].sort_values("date").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    matches.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(matches)} matches -> {OUTPUT_CSV}")
    log(f"Leagues: {sorted(matches['league'].unique())}")
    log(f"Seasons: {sorted(s for s in matches['season'].unique() if s)}")
    log(f"Matches with odds: {matches['odds_home'].notna().sum()} / {len(matches)}")
    log(f"Distinct clubs: {pd.concat([matches['home'], matches['away']]).nunique()}")


if __name__ == "__main__":
    main()
