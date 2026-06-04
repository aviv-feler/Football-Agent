"""
build_national_data.py
Convert the historical World Cup workbook (2018/2014/2010) into a committed CSV the
app can read without openpyxl at runtime.

Output: data/national_matches.csv with one row per international match, team names
canonicalized to the player-data nationality space (so Elo ratings can be merged onto
squad-strength), the 90-minute result, stage, and bookmaker odds where available.

Input:  data/WorldCup_2018_2014_2010.xlsx
"""

import os
import re
import sys
import glob
import pandas as pd

from ds_engine import normalize_nation

PLAYERS_CSV = "data/players.csv"
OUTPUT_CSV = "data/national_matches.csv"
SHEET_RE = re.compile(r"WorldCup\d{4}", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[national_data] {msg}", flush=True)


def find_workbook() -> str:
    """Locate the World Cup workbook (any data/WorldCup*.xlsx)."""
    candidates = sorted(glob.glob("data/WorldCup*.xlsx"))
    if not candidates:
        log("ERROR: no data/WorldCup*.xlsx workbook found.")
        sys.exit(1)
    if len(candidates) > 1:
        log(f"WARNING: multiple workbooks found, using the last: {candidates}")
    return candidates[-1]


def known_nationalities() -> set:
    """The canonical nationality names used in the player data."""
    df = pd.read_csv(PLAYERS_CSV, usecols=["country_of_citizenship"], low_memory=False)
    return set(df["country_of_citizenship"].dropna().astype(str).unique())


def canon(name, known: set, cache: dict) -> str:
    raw = str(name).strip()
    if raw in cache:
        return cache[raw]
    mapped = normalize_nation(raw, known) or raw
    cache[raw] = mapped
    return mapped


def main() -> None:
    wc_xlsx = find_workbook()
    sheets = [s for s in pd.ExcelFile(wc_xlsx).sheet_names if SHEET_RE.fullmatch(s)]
    if not sheets:
        log(f"ERROR: no 'WorldCup<year>' sheets in {wc_xlsx}.")
        sys.exit(1)
    log(f"Workbook: {wc_xlsx} | tournament sheets: {sorted(sheets)}")

    known = known_nationalities()
    cache: dict = {}

    frames = []
    for sheet in sheets:
        df = pd.read_excel(wc_xlsx, sheet_name=sheet)
        df = df.rename(columns={
            "Competition": "tournament", "Home": "home", "Away": "away",
            "HGFT": "home_goals", "AGFT": "away_goals", "Date": "date", "Stage": "stage",
            "bet365-H": "odds_home", "bet365-D": "odds_draw", "bet365-A": "odds_away",
            "H-Avg": "avg_home", "D-Avg": "avg_draw", "A-Avg": "avg_away",
        })
        keep = ["tournament", "home", "away", "home_goals", "away_goals", "date", "stage",
                "odds_home", "odds_draw", "odds_away", "avg_home", "avg_draw", "avg_away"]
        df = df[[c for c in keep if c in df.columns]].copy()
        frames.append(df)

    wc = pd.concat(frames, ignore_index=True)
    wc = wc.dropna(subset=["home", "away", "home_goals", "away_goals"])
    wc["home_goals"] = pd.to_numeric(wc["home_goals"], errors="coerce")
    wc["away_goals"] = pd.to_numeric(wc["away_goals"], errors="coerce")
    wc = wc.dropna(subset=["home_goals", "away_goals"])

    # Canonicalize team names to the player-data nationality space.
    unmapped = set()
    for col in ["home", "away"]:
        wc[col + "_raw"] = wc[col]
        wc[col] = wc[col].map(lambda n: canon(n, known, cache))
    for raw, mapped in cache.items():
        if mapped == raw and raw not in known:
            unmapped.add(raw)

    # 90-minute result from the home team's perspective: H / D / A.
    wc["result"] = "D"
    wc.loc[wc["home_goals"] > wc["away_goals"], "result"] = "H"
    wc.loc[wc["home_goals"] < wc["away_goals"], "result"] = "A"
    wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
    wc = wc.sort_values("date").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    wc.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(wc)} matches -> {OUTPUT_CSV}")
    log(f"Distinct teams: {pd.concat([wc['home'], wc['away']]).nunique()}")
    log(f"Matches with odds: {wc['odds_home'].notna().sum()}")
    if unmapped:
        log(f"WARNING: {len(unmapped)} team names did not map to a player-data nationality "
            f"(kept as-is): {sorted(unmapped)}")
    else:
        log("All team names mapped to player-data nationalities.")


if __name__ == "__main__":
    main()
