"""
Centralized data management for ScoutAI.

All chatbot data access should go through this module. It defines source priority,
normalizes current player profiles, and prevents random stale CSV reads.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ds_engine import jaccard


DATA_DIR = Path("data")
MAIN_PLAYERS_FILE = DATA_DIR / "players.csv"
PLAYER_PROFILES_FILE = DATA_DIR / "player_profiles.csv"
PLAYERS_CLEAN_FILE = DATA_DIR / "players_clean.csv"
APPEARANCES_FILE = DATA_DIR / "appearances.csv"
CLUBS_FILE = DATA_DIR / "clubs.csv"
GAMES_FILE = DATA_DIR / "games.csv"
FEATURES_FILE = DATA_DIR / "player_features.npy"
SOURCE_MAP_FILE = DATA_DIR / "data_source_map.json"
FOOTBALL_WORKBOOK_FILE = DATA_DIR / "Football_clubs_players_full.xlsx"
CURRENT_STATS_SHEET = "players_statistics_final"

DATA_VERSION = "2026-06-03-source-priority-v2"


SOURCE_PRIORITY: dict[str, Any] = {
    "current_player_rankings": [
        {
            "priority": 1,
            "file": "data/Football_clubs_players_full.xlsx",
            "sheet": "players_statistics_final",
            "reason": "Current/season-style player statistics for ranking questions.",
        },
        {"priority": 2, "file": "Football-Data API", "reason": "Fresh scorers/standings when relevant and available."},
        {"priority": 3, "file": "data/player_profiles.csv", "reason": "Fallback only; contains aggregated profile context."},
    ],
    "current_player_identity": [
        {"priority": 1, "file": "data/players.csv", "reason": "Raw Transfermarkt current player club/league fields."},
        {"priority": 2, "file": "data/player_profiles.csv", "reason": "Generated from players.csv with metadata."},
        {"priority": 3, "file": "data/players_clean.csv", "reason": "Generated feature file, used only as fallback/features."},
    ],
    "player_similarity": [
        {"priority": 1, "file": "data/player_profiles.csv", "reason": "Rebuilt profiles with position-aware scores and clusters."},
        {"priority": 2, "file": "data/player_features.npy", "reason": "Legacy numeric vectors; used only while row-aligned."},
        {"priority": 3, "file": "data/feature_meta.json", "reason": "K-Means metadata."},
    ],
    "player_stats": [
        {"priority": 1, "file": "data/appearances.csv", "reason": "Match appearance aggregates."},
        {"priority": 2, "file": "data/player_profiles.csv", "reason": "Pre-aggregated generated profile."},
    ],
    "teams_clubs": [
        {"priority": 1, "file": "data/clubs.csv", "reason": "Club metadata and domestic competition."},
        {"priority": 2, "file": "data/player_profiles.csv", "reason": "Player-derived club squads."},
    ],
    "matches_results": [
        {"priority": 1, "file": "Football-Data API", "reason": "Fresh standings/results when key is available."},
        {"priority": 2, "file": "data/games.csv", "reason": "Local historical match data fallback."},
        {"priority": 3, "file": "data/club_games.csv", "reason": "Club-side match aggregates fallback."},
    ],
    "world_cup_schedule": [
        {"priority": 1, "file": "data/fwc26_match_schedule_agent.csv", "reason": "Local World Cup 2026 schedule."}
    ],
}


LEAGUE_LABELS = {
    "GB1": "Premier League",
    "L1": "Bundesliga",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
    "PO1": "Primeira Liga",
    "NL1": "Eredivisie",
    "BRA1": "Brasileirao",
    "ARG1": "Argentina Primera",
    "MLS1": "MLS",
}

CURRENT_STATS_LEAGUES = {
    "eng premier league": "Premier League",
    "es la liga": "La Liga",
    "it serie a": "Serie A",
    "de bundesliga": "Bundesliga",
    "fr ligue 1": "Ligue 1",
}

LEAGUE_TO_CURRENT_STATS = {
    "Premier League": "eng Premier League",
    "La Liga": "es La Liga",
    "Serie A": "it Serie A",
    "Bundesliga": "de Bundesliga",
    "Ligue 1": "fr Ligue 1",
    "GB1": "eng Premier League",
    "ES1": "es La Liga",
    "IT1": "it Serie A",
    "L1": "de Bundesliga",
    "FR1": "fr Ligue 1",
}


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    nf = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in nf if not unicodedata.combining(ch))
    return " ".join(text.lower().strip().split())


def normalize_league(value: Any) -> str:
    text = str(value or "").strip()
    if text in LEAGUE_LABELS:
        return LEAGUE_LABELS[text]
    norm = normalize_text(text)
    if norm in CURRENT_STATS_LEAGUES:
        return CURRENT_STATS_LEAGUES[norm]
    for label in set(LEAGUE_LABELS.values()) | set(CURRENT_STATS_LEAGUES.values()):
        if normalize_text(label) == norm:
            return label
    return text


def normalize_team_name(value: Any) -> str:
    text = normalize_text(value)
    replacements = {
        "munchen": "munich",
        "muenchen": "munich",
        "utd": "united",
        "man city": "manchester city",
        "man utd": "manchester united",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    for token in ["football club", "club", "fc", "cf", "afc", "association football"]:
        text = re.sub(rf"\b{re.escape(token)}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def current_stats_league_key(value: Any) -> str:
    label = normalize_league(value)
    return LEAGUE_TO_CURRENT_STATS.get(label, LEAGUE_TO_CURRENT_STATS.get(str(value), str(value)))


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _norm_series(series: pd.Series) -> pd.Series:
    max_val = series.max()
    return series / max_val if max_val and max_val > 0 else series * 0


def _position_group(value: Any) -> str:
    text = normalize_text(value)
    tokens = set(re.findall(r"[a-z]+", text))
    if "gk" in tokens or "goalkeeper" in tokens:
        return "Goalkeeper"
    if "mf" in tokens or "midfield" in text or "midfielder" in text:
        return "Midfield"
    if "fw" in tokens or "attack" in text or "forward" in text:
        return "Attack"
    if "df" in tokens or "defender" in text:
        return "Defender"
    return ""


def _read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a simple XLSX worksheet without requiring openpyxl."""
    with zipfile.ZipFile(path) as zf:
        ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {node.attrib["Id"]: node.attrib["Target"] for node in rels_root}
        target = None
        for sheet in wb.findall("a:sheets/a:sheet", ns_main):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rels.get(rel_id)
                break
        if not target:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}")
        sheet_path = "xl/" + target.lstrip("/")

        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns_main):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns_main)))

        def colnum(ref: str) -> int:
            letters = re.match(r"([A-Z]+)", ref).group(1)
            n = 0
            for ch in letters:
                n = n * 26 + ord(ch) - 64
            return n

        rows: list[list[Any]] = []
        root = ET.fromstring(zf.read(sheet_path))
        for row in root.findall("a:sheetData/a:row", ns_main):
            vals: dict[int, Any] = {}
            for cell in row.findall("a:c", ns_main):
                v = cell.find("a:v", ns_main)
                val = "" if v is None else v.text
                if cell.attrib.get("t") == "s" and val != "":
                    val = shared[int(val)]
                vals[colnum(cell.attrib["r"])] = val
            if vals:
                rows.append([vals.get(i, "") for i in range(1, max(vals) + 1)])
        if not rows:
            return pd.DataFrame()
        header = [str(h).strip() for h in rows[0]]
        body = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
        return pd.DataFrame(body, columns=header)


def write_source_map(path: Path = SOURCE_MAP_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_version": DATA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_priority": SOURCE_PRIORITY,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_main_players() -> pd.DataFrame:
    return pd.read_csv(MAIN_PLAYERS_FILE, low_memory=False)


def load_appearances() -> pd.DataFrame:
    return pd.read_csv(APPEARANCES_FILE, low_memory=False)


def load_teams() -> pd.DataFrame:
    return pd.read_csv(CLUBS_FILE, low_memory=False)


def load_matches() -> pd.DataFrame:
    return pd.read_csv(GAMES_FILE, low_memory=False)


def load_current_player_stats() -> pd.DataFrame:
    df = _read_xlsx_sheet(FOOTBALL_WORKBOOK_FILE, CURRENT_STATS_SHEET)
    numeric_cols = [
        "Age", "Matches_Played", "Starts", "Minutes_Played", "90s", "Goals", "Assists",
        "Total_Gls_Ast", "Non_Penalty_Goals", "Shots", "Shots_On_Target", "Interceptions",
        "Tackles_Won", "Blocks", "Fouls_Committed", "Fouls_Drawn", "Crs", "Compl",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["league_name"] = df["League"].map(normalize_league)
    df["position_group"] = df["Position"].map(_position_group)
    df["player_name"] = df["Player_Name"].fillna("").astype(str)
    df["club"] = df["Club"].fillna("").astype(str)
    df["source_file"] = str(FOOTBALL_WORKBOOK_FILE).replace("\\", "/")
    df["source_sheet"] = CURRENT_STATS_SHEET
    return df


def _filter_current_stats(
    league: str | None = None,
    team: str | None = None,
    position_filter: str | None = None,
    min_minutes: int = 300,
) -> pd.DataFrame:
    df = load_current_player_stats()
    if league:
        key = current_stats_league_key(league)
        df = df[df["League"].fillna("").map(normalize_text).eq(normalize_text(key))]
    if team:
        q = normalize_team_name(team)
        clubs_norm = df["Club"].fillna("").map(normalize_team_name)
        df = df[clubs_norm.apply(lambda club: bool(q and (q in club or club in q)))]
    if position_filter:
        pos = normalize_text(position_filter)
        if pos in {"attack", "attacker", "forward", "striker", "fw"}:
            df = df[df["Position"].fillna("").astype(str).str.contains("FW", case=False, na=False)]
        elif pos in {"midfield", "midfielder", "mf"}:
            df = df[df["Position"].fillna("").astype(str).str.contains("MF", case=False, na=False)]
        elif pos in {"defender", "defence", "defense", "df"}:
            df = df[df["Position"].fillna("").astype(str).str.contains("DF", case=False, na=False)]
        elif pos in {"goalkeeper", "keeper", "gk"}:
            df = df[df["Position"].fillna("").astype(str).str.contains("GK", case=False, na=False)]
    if "Minutes_Played" in df.columns:
        df = df[_numeric(df, "Minutes_Played") >= min_minutes]
    return df.copy()


def rank_current_players(
    league: str | None = None,
    team: str | None = None,
    position_filter: str | None = None,
    ranking_type: str = "best_player",
    limit: int = 10,
) -> pd.DataFrame:
    df = _filter_current_stats(league=league, team=team, position_filter=position_filter)
    if df.empty:
        return df
    goals = _numeric(df, "Goals")
    assists = _numeric(df, "Assists")
    sot = _numeric(df, "Shots_On_Target")
    shots = _numeric(df, "Shots")
    minutes = _numeric(df, "Minutes_Played")
    tackles = _numeric(df, "Tackles_Won")
    interceptions = _numeric(df, "Interceptions")
    crosses = _numeric(df, "Crs")
    completions = _numeric(df, "Compl")

    if ranking_type == "top_scorer":
        score = (
            0.55 * _norm_series(goals)
            + 0.15 * _norm_series(sot)
            + 0.12 * _norm_series(shots)
            + 0.10 * _norm_series(minutes)
            + 0.08 * _norm_series(assists)
        )
    elif ranking_type in {"best_attacking_player", "best_striker"}:
        score = (
            0.36 * _norm_series(goals)
            + 0.20 * _norm_series(assists)
            + 0.18 * _norm_series(sot)
            + 0.12 * _norm_series(shots)
            + 0.14 * _norm_series(minutes)
        )
    elif ranking_type == "creative_midfielder":
        score = (
            0.32 * _norm_series(assists)
            + 0.20 * _norm_series(completions)
            + 0.16 * _norm_series(crosses)
            + 0.14 * _norm_series(minutes)
            + 0.10 * _norm_series(goals)
            + 0.08 * _norm_series(sot)
        )
    elif ranking_type == "defensive_midfielder":
        score = (
            0.30 * _norm_series(tackles)
            + 0.25 * _norm_series(interceptions)
            + 0.18 * _norm_series(completions)
            + 0.17 * _norm_series(minutes)
            + 0.10 * _norm_series(assists)
        )
    else:
        score = (
            0.25 * _norm_series(goals)
            + 0.22 * _norm_series(assists)
            + 0.18 * _norm_series(sot)
            + 0.15 * _norm_series(minutes)
            + 0.10 * _norm_series(tackles + interceptions)
            + 0.10 * _norm_series(completions)
        )
    return df.assign(current_rank_score=score).sort_values("current_rank_score", ascending=False).head(limit)


def _appearance_aggregates() -> pd.DataFrame:
    apps = load_appearances()
    return apps.groupby("player_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        appearances=("appearance_id", "count"),
    ).reset_index()


def _current_stats_profile_scores() -> pd.DataFrame:
    try:
        stats = load_current_player_stats()
    except Exception as exc:
        print(f"[data_manager] current stats unavailable for profile scoring: {exc}", flush=True)
        return pd.DataFrame()
    if stats.empty:
        return pd.DataFrame()
    stats["_player_key"] = stats["Player_Name"].map(normalize_text)
    grouped = stats.groupby("_player_key").agg(
        current_goals=("Goals", "sum"),
        current_assists=("Assists", "sum"),
        current_minutes=("Minutes_Played", "sum"),
        current_shots=("Shots", "sum"),
        current_shots_on_target=("Shots_On_Target", "sum"),
        current_tackles=("Tackles_Won", "sum"),
        current_interceptions=("Interceptions", "sum"),
        current_crosses=("Crs", "sum"),
        current_completions=("Compl", "sum"),
        current_matches=("Matches_Played", "sum"),
    ).reset_index()

    safe_minutes = grouped["current_minutes"].where(grouped["current_minutes"] >= 90, np.nan)
    for col in [
        "current_goals", "current_assists", "current_shots", "current_shots_on_target",
        "current_tackles", "current_interceptions", "current_crosses", "current_completions",
    ]:
        grouped[f"{col}_per90"] = (grouped[col] / (safe_minutes / 90)).fillna(0)

    grouped["finishing_score"] = (
        0.55 * _norm_series(grouped["current_goals_per90"])
        + 0.30 * _norm_series(grouped["current_shots_on_target_per90"])
        + 0.15 * _norm_series(grouped["current_shots_per90"])
    )
    grouped["creativity_score"] = (
        0.55 * _norm_series(grouped["current_assists_per90"])
        + 0.25 * _norm_series(grouped["current_crosses_per90"])
        + 0.20 * _norm_series(grouped["current_shots_per90"])
    )
    grouped["passing_score"] = (
        0.75 * _norm_series(grouped["current_completions_per90"])
        + 0.25 * _norm_series(grouped["current_minutes"])
    )
    grouped["defensive_score"] = (
        0.55 * _norm_series(grouped["current_tackles_per90"])
        + 0.45 * _norm_series(grouped["current_interceptions_per90"])
    )
    grouped["possession_score"] = (
        0.60 * grouped["passing_score"]
        + 0.25 * _norm_series(grouped["current_minutes"])
        + 0.15 * grouped["creativity_score"]
    )
    grouped["physical_score"] = _norm_series(grouped["current_minutes"]) + 0.15 * _norm_series(grouped["current_matches"])
    grouped["potential_score"] = 0.0
    grouped["overall_score"] = (
        0.24 * grouped["finishing_score"]
        + 0.22 * grouped["creativity_score"]
        + 0.20 * grouped["passing_score"]
        + 0.16 * grouped["defensive_score"]
        + 0.18 * grouped["possession_score"]
    )
    return grouped


def _assign_position_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "finishing_score", "creativity_score", "passing_score", "defensive_score",
        "possession_score", "physical_score", "potential_score", "overall_score",
    ]
    for col in score_cols:
        if col not in profile.columns:
            profile[col] = 0.0
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0)
    profile["position_cluster"] = -1
    profile["position_archetype"] = ""
    profile["position_group"] = profile["position"].map(_position_group)

    def label(row: pd.Series) -> str:
        pos = row.get("position_group") or row.get("position")
        finish = row["finishing_score"]
        creative = row["creativity_score"]
        passing = row["passing_score"]
        defensive = row["defensive_score"]
        possession = row["possession_score"]
        if pos == "Attack":
            if finish >= creative and finish >= possession:
                return "Finisher / Goalscorer"
            if creative >= finish:
                return "Creative forward / Winger"
            return "Possession forward"
        if pos == "Midfield":
            if defensive >= max(creative, passing) * 0.9:
                return "Defensive midfielder / Ball winner"
            if creative >= passing:
                return "Creative midfielder / Playmaker"
            return "Central midfielder / Possession passer"
        if pos == "Defender":
            if passing >= defensive:
                return "Ball-playing defender"
            return "Defensive stopper"
        if pos == "Goalkeeper":
            return "Goalkeeper"
        return "General profile"

    for pos, idx in profile.groupby("position_group").groups.items():
        idx = list(idx)
        if not idx:
            continue
        sub = profile.loc[idx, score_cols]
        if len(sub) >= 4 and sub.to_numpy().std() > 0:
            k = min(4, len(sub))
            X = StandardScaler().fit_transform(sub)
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            profile.loc[idx, "position_cluster"] = labels
        else:
            profile.loc[idx, "position_cluster"] = 0
    profile["position_archetype"] = profile.apply(label, axis=1)
    return profile


def regenerate_player_profiles(output_path: Path = PLAYER_PROFILES_FILE) -> pd.DataFrame:
    """Rebuild current player profiles from the selected main players source."""
    players = load_main_players()
    clean = pd.read_csv(PLAYERS_CLEAN_FILE, low_memory=False)
    apps = _appearance_aggregates()

    profile = clean.drop(columns=[
        c for c in ["club", "league", "market_value_in_eur", "nationality", "international_caps"]
        if c in clean.columns
    ]).copy()

    current_cols = players[[
        "player_id",
        "name",
        "country_of_citizenship",
        "current_club_name",
        "current_club_domestic_competition_id",
        "market_value_in_eur",
        "international_caps",
        "last_season",
        "current_club_id",
    ]].rename(columns={
        "country_of_citizenship": "nationality",
        "current_club_name": "club",
        "current_club_domestic_competition_id": "league",
    })

    profile = profile.merge(current_cols, on="player_id", how="left", suffixes=("", "_current"))
    if "name_current" in profile.columns:
        profile["name"] = profile["name_current"].fillna(profile.get("name"))
        profile = profile.drop(columns=["name_current"])

    profile = profile.drop(columns=[
        c for c in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards", "appearances"]
        if c in profile.columns
    ])
    profile = profile.merge(apps, on="player_id", how="left")
    for col in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards", "appearances"]:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0)

    safe = profile["minutes_played"].where(profile["minutes_played"] >= 90, np.nan)
    profile["goals_per90"] = (profile["goals"] / (safe / 90)).fillna(0)
    profile["assists_per90"] = (profile["assists"] / (safe / 90)).fillna(0)
    profile["ga_per90"] = profile["goals_per90"] + profile["assists_per90"]
    profile["cards_per90"] = ((profile["yellow_cards"] + profile["red_cards"]) / (safe / 90)).fillna(0)
    profile["league_name"] = profile["league"].map(LEAGUE_LABELS).fillna(profile["league"])
    profile["player_name"] = profile["name"].fillna(profile.get("player_name", "")).astype(str)
    profile["_player_key"] = profile["player_name"].map(normalize_text)

    current_scores = _current_stats_profile_scores()
    if not current_scores.empty:
        profile = profile.merge(current_scores, on="_player_key", how="left")
    score_cols = [
        "current_goals", "current_assists", "current_minutes", "current_shots", "current_shots_on_target",
        "current_tackles", "current_interceptions", "current_crosses", "current_completions",
        "finishing_score", "creativity_score", "passing_score", "defensive_score", "possession_score",
        "physical_score", "potential_score", "overall_score",
    ]
    for col in score_cols:
        if col not in profile.columns:
            profile[col] = 0.0
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0)
    profile["potential_score"] = np.maximum(
        profile["potential_score"],
        (1 - ((pd.to_numeric(profile["age"], errors="coerce").fillna(30).clip(16, 36) - 16) / 20)).clip(0, 1)
        * (0.4 + 0.6 * profile["overall_score"])
    )
    profile = _assign_position_profiles(profile)
    profile = profile.drop(columns=["_player_key"], errors="ignore")

    generated_at = datetime.now(timezone.utc).isoformat()
    profile["source_file"] = str(MAIN_PLAYERS_FILE).replace("\\", "/")
    profile["generated_at"] = generated_at
    profile["data_version"] = DATA_VERSION
    profile["source_priority"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    write_source_map()
    return profile


def load_player_profiles(regenerate_if_missing: bool = True) -> pd.DataFrame:
    if not PLAYER_PROFILES_FILE.exists() and regenerate_if_missing:
        try:
            return regenerate_player_profiles()
        except Exception as exc:
            print(f"[data_manager] could not regenerate player_profiles.csv, falling back to players_clean.csv: {exc}", flush=True)
            return pd.read_csv(PLAYERS_CLEAN_FILE, low_memory=False)
    if PLAYER_PROFILES_FILE.exists():
        return pd.read_csv(PLAYER_PROFILES_FILE, low_memory=False)
    return pd.read_csv(PLAYERS_CLEAN_FILE, low_memory=False)


def get_player_current_info(player_name: str) -> pd.Series | None:
    df = load_player_profiles()
    q = normalize_text(player_name)
    names = df["player_name"].fillna("").map(normalize_text)
    exact = df[names == q]
    if not exact.empty:
        return exact.sort_values("market_value_in_eur", ascending=False).iloc[0]
    contains = df[names.str.contains(q, regex=False, na=False)]
    if not contains.empty:
        return contains.sort_values("market_value_in_eur", ascending=False).iloc[0]
    return None


def get_players_by_league(league: str) -> pd.DataFrame:
    df = load_player_profiles()
    q = normalize_text(league)
    if league in set(df["league"].dropna().astype(str)):
        return df[df["league"] == league]
    return df[df["league_name"].fillna("").map(normalize_text).eq(q)]


def get_players_by_team(team: str) -> pd.DataFrame:
    df = load_player_profiles()
    q = normalize_text(team)
    clubs = df["club"].fillna("").map(normalize_text)
    return df[clubs.str.contains(q, regex=False, na=False)]


def get_best_players_by_league(league: str, position_filter: str | None = None, limit: int = 10) -> pd.DataFrame:
    current = rank_current_players(
        league=league,
        position_filter=position_filter,
        ranking_type="best_attacking_player" if position_filter and normalize_text(position_filter) in {"attack", "fw"} else "best_player",
        limit=limit,
    )
    if not current.empty:
        return current
    df = get_players_by_league(league)
    df = df[pd.to_numeric(df.get("last_season", 0), errors="coerce").fillna(0) >= 2025]
    if position_filter:
        df = df[df["position"].fillna("").str.contains(position_filter, case=False, na=False)]
    if df.empty:
        return df
    score = (
        0.35 * df["overall_score"].fillna(0)
        + 0.20 * df["finishing_score"].fillna(0)
        + 0.20 * df["creativity_score"].fillna(0)
        + 0.15 * df["possession_score"].fillna(0)
        + 0.10 * np.log1p(df["market_value_in_eur"].fillna(0))
    )
    return df.assign(data_manager_score=score).sort_values("data_manager_score", ascending=False).head(limit)


def find_similar_players(player_name: str, engine, limit: int = 5) -> pd.DataFrame:
    df = load_player_profiles()
    info = get_player_current_info(player_name)
    if info is None:
        return pd.DataFrame()
    idx = int(df.index[df["player_id"] == info["player_id"]][0])
    mask = (df["position"] == info["position"]) & (df.index != idx)
    ilocs = np.where(mask.values)[0]
    sims = cosine_similarity(engine.X[idx].reshape(1, -1), engine.X[ilocs])[0]
    target_traits = set(str(info.get(c, "")) for c in ["position", "sub_position", "nationality", "league", "archetype"])
    rows = []
    for pos, iloc in enumerate(ilocs):
        row = df.iloc[int(iloc)]
        traits = set(str(row.get(c, "")) for c in ["position", "sub_position", "nationality", "league", "archetype"])
        rows.append((float(sims[pos]) + 0.2 * jaccard(target_traits, traits), int(iloc)))
    top = [iloc for _, iloc in sorted(rows, reverse=True)[:limit]]
    return df.iloc[top].copy()


def recommend_replacements(player_name: str, target_team: str | None = None, engine=None, limit: int = 5) -> pd.DataFrame:
    if engine is not None:
        return find_similar_players(player_name, engine, limit)
    return pd.DataFrame()
