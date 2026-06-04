"""
data_prep.py
Prepare ScoutAI data and build a normalized numeric feature matrix for each player.

Data Science steps:
  - Feature engineering: per-90 stats plus profile features.
  - Missing values: fill by position-group median, not zero.
  - Normalization: StandardScaler z-scores for meaningful similarity.
  - K-Means: player archetype clusters, with k selected by the elbow method.

Input:  data/players.csv, data/appearances.csv
Output: data/players_clean.csv, data/player_features.npy, data/feature_meta.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PLAYERS_CSV     = "data/players.csv"
APPEARANCES_CSV = "data/appearances.csv"
FIFA_CSV        = "data/fifa_players.csv"
OUTPUT_CSV      = "data/players_clean.csv"
FEATURES_NPY    = "data/player_features.npy"
META_JSON       = "data/feature_meta.json"

# Numeric features that make up the player performance vector.
FEATURE_COLS = [
    "goals_per90", "assists_per90", "ga_per90", "cards_per90",
    "minutes_played_log", "appearances_log", "age", "height_in_cm",
    "market_value_log", "international_caps_log",
]
# Skewed features transformed with log1p before scaling.
LOG_COLS = {
    "minutes_played": "minutes_played_log",
    "appearances":    "appearances_log",
    "market_value_in_eur": "market_value_log",
    "international_caps":   "international_caps_log",
}


def log(msg: str) -> None:
    print(f"[data_prep] {msg}", flush=True)


def print_schema(path: str, name: str):
    df = pd.read_csv(path, nrows=3, low_memory=False)
    log(f"Schema for {name} ({path}):")
    log(f"  Columns: {list(df.columns)}")


def load_players(path: str) -> pd.DataFrame:
    log("Loading players...")
    cols = ["player_id", "name", "first_name", "last_name", "country_of_citizenship",
            "sub_position", "position", "foot", "height_in_cm", "market_value_in_eur",
            "current_club_name", "current_club_domestic_competition_id",
            "date_of_birth", "international_caps"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df = df.rename(columns={
        "country_of_citizenship": "nationality",
        "current_club_name": "club",
        "current_club_domestic_competition_id": "league",
    })
    df["player_name"] = df["name"].fillna(
        df.get("first_name", "").fillna("") + " " + df.get("last_name", "").fillna("")
    ).astype(str).str.strip()
    df["age"] = pd.to_datetime(df["date_of_birth"], errors="coerce").apply(
        lambda d: (pd.Timestamp.now() - d).days // 365 if pd.notna(d) else np.nan
    )
    log(f"  {len(df)} players")
    return df


def load_appearances(path: str) -> pd.DataFrame:
    log("Loading appearances and aggregating by player...")
    need = ["player_id", "goals", "assists", "minutes_played", "yellow_cards", "red_cards"]
    df = pd.read_csv(path, usecols=lambda c: c in need, low_memory=False)
    agg = df.groupby("player_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        appearances=("goals", "size"),
    ).reset_index()
    log(f"  {len(agg)} players with appearance data")
    return agg


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    log("Engineering features (per-90 stats + log transforms)...")

    for c in ["goals", "assists", "minutes_played", "yellow_cards",
              "red_cards", "appearances", "international_caps"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    mins = df["minutes_played"].clip(lower=0)
    safe = mins.where(mins >= 90, np.nan)
    df["goals_per90"]   = (df["goals"]   / (safe / 90)).fillna(0)
    df["assists_per90"] = (df["assists"] / (safe / 90)).fillna(0)
    df["ga_per90"]      = df["goals_per90"] + df["assists_per90"]
    df["cards_per90"]   = ((df["yellow_cards"] + df["red_cards"]) / (safe / 90)).fillna(0)

    # Winsorize per-90 values so tiny samples cannot dominate.
    for c in ["goals_per90", "assists_per90", "ga_per90", "cards_per90"]:
        cap = df[c].quantile(0.99)
        df[c] = df[c].clip(upper=cap)

    # Log transforms for skewed features.
    for raw, logged in LOG_COLS.items():
        df[logged] = np.log1p(pd.to_numeric(df[raw], errors="coerce").fillna(0).clip(lower=0))

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_in_cm"] = pd.to_numeric(df["height_in_cm"], errors="coerce")
    return df


def fill_by_position_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values by position-group median, not zero."""
    log("Filling missing values by position median...")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        med_by_pos = df.groupby("position")[col].transform("median")
        df[col] = df[col].fillna(med_by_pos)
        df[col] = df[col].fillna(df[col].median())
    return df


def choose_k_elbow(X: np.ndarray, k_range=range(2, 11)) -> int:
    """Choose k by the elbow method using distance from the endpoint line."""
    log("Running elbow method to choose k...")
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        log(f"  k={k}: inertia={km.inertia_:,.0f}")

    ks = np.array(list(k_range))
    iner = np.array(inertias)
    # Distance from the line connecting the endpoints; the max distance is the elbow.
    p1, p2 = np.array([ks[0], iner[0]]), np.array([ks[-1], iner[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for i in range(len(ks)):
        p = np.array([ks[i], iner[i]]) - p1
        proj = p - np.dot(p, line) * line
        dists.append(np.linalg.norm(proj))
    best_k = int(ks[int(np.argmax(dists))])
    log(f"  Selected k={best_k} by elbow method")
    return best_k


def build_archetypes(centroids: np.ndarray, feature_names: list[str]) -> dict:
    """
    Assign a readable archetype name to each cluster based on its strongest
    centroid feature deviations.
    """
    idx = {n: i for i, n in enumerate(feature_names)}
    desc = {
        "goals_per90":       ("Goalscorer / Poacher", None),
        "assists_per90":     ("Creator / Playmaker", None),
        "market_value_log":  ("Elite high-value", None),
        "minutes_played_log":("High-minutes regular", "Fringe / limited minutes"),
        "international_caps_log": ("Established international", None),
        "age":               ("Veteran", "Young prospect"),
        "cards_per90":       ("Aggressive / physical", None),
        "height_in_cm":      ("Tall / aerial", None),
    }
    labels, used = {}, set()
    for ci, c in enumerate(centroids):
        ranked = sorted(desc.keys(), key=lambda f: abs(c[idx[f]]), reverse=True)
        chosen = None
        for f in ranked:
            hi, lo = desc[f]
            name = hi if c[idx[f]] >= 0 else lo
            if name and name not in used:
                chosen = name
                break
        if chosen is None:
            chosen = f"Archetype {ci}"
        used.add(chosen)
        labels[int(ci)] = chosen
    return labels


def main():
    os.makedirs("data", exist_ok=True)
    for p in (PLAYERS_CSV, APPEARANCES_CSV):
        if not os.path.exists(p):
            log(f"ERROR: {p} was not found.")
            sys.exit(1)

    # Schema validation: print real columns before building features.
    print_schema(PLAYERS_CSV, "players")
    print_schema(APPEARANCES_CSV, "appearances")
    if os.path.exists(FIFA_CSV):
        print_schema(FIFA_CSV, "fifa_players")

    players = load_players(PLAYERS_CSV)
    apps    = load_appearances(APPEARANCES_CSV)

    df = players.merge(apps, on="player_id", how="left")
    log(f"Merged dataset: {len(df)} players")

    df = engineer(df)
    df = fill_by_position_median(df)

    log("Scaling all features with StandardScaler (z-score)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)

    k = choose_k_elbow(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    log(f"Cluster distribution: {df['cluster'].value_counts().sort_index().to_dict()}")

    archetypes = build_archetypes(km.cluster_centers_, FEATURE_COLS)
    df["archetype"] = df["cluster"].map(archetypes)
    log(f"Archetypes: {archetypes}")

    # Helper columns for Jaccard and display.
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 21, 25, 29, 33, 99],
                              labels=["U21", "21-25", "26-29", "30-33", "33+"]).astype(str)
    df["value_tier"] = pd.cut(df["market_value_in_eur"],
                              bins=[-1, 1e6, 5e6, 20e6, 50e6, 1e12],
                              labels=["low", "modest", "mid", "high", "elite"]).astype(str)

    df.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(df)} players -> {OUTPUT_CSV}")

    np.save(FEATURES_NPY, X)
    log(f"Saved normalized feature matrix {X.shape} -> {FEATURES_NPY}")

    meta = {
        "feature_names": FEATURE_COLS,
        "k": k,
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "centroids":    km.cluster_centers_.tolist(),
        "archetypes":   archetypes,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"Saved metadata -> {META_JSON}")
    log("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
