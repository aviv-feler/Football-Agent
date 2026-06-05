"""
data_prep.py
Prepare ScoutAI data and build a normalized numeric feature matrix for each player.

Data Science steps:
  - Feature engineering: per-90 stats plus profile features.
  - Missing values: fill by position-group median, not zero.
  - Normalization: StandardScaler z-scores for meaningful similarity.
  - K-Means: player archetype clusters, with k selected by the elbow method.

Input:  data/players.csv (identity + market value), data/appearances.csv (career stats),
        data/players_data-2025_2026.csv (FBref current-season overlay, via data_manager),
        data/FC26_20250921.csv (EA FC26 playing-style attributes).
Output: data/players_clean.csv, data/player_features.npy, data/feature_meta.json
"""

import os
import sys
import json
import unicodedata
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _norm_name(s) -> str:
    """Accent-insensitive lowercase key for joining across data sources."""
    s = "" if pd.isna(s) else str(s)
    nf = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nf if not unicodedata.combining(c)).lower().strip()

PLAYERS_CSV     = "data/players.csv"
APPEARANCES_CSV = "data/appearances.csv"
FC26_CSV        = "data/FC26_20250921.csv"
OUTPUT_CSV      = "data/players_clean.csv"
FEATURES_NPY    = "data/player_features.npy"
META_JSON       = "data/feature_meta.json"

# Numeric performance features (per-90 stats + profile descriptors).
PERF_FEATURE_COLS = [
    "goals_per90", "assists_per90", "ga_per90", "cards_per90",
    "minutes_played_log", "appearances_log", "age", "height_in_cm",
    "market_value_log", "international_caps_log",
]
# EA FC26 playing-style attributes added to the similarity vector (0-100 scale).
FC26_STYLE_COLS = [
    "fc_pace", "fc_shooting", "fc_passing", "fc_dribbling", "fc_defending", "fc_physic",
]
# FC26 quality/potential columns kept on the player table but NOT in the style vector
# (overall quality is already represented by market value).
FC26_EXTRA_COLS = ["fc_overall", "fc_potential"]
# FC26 source column -> our column name.
FC26_SOURCE_MAP = {
    "pace": "fc_pace", "shooting": "fc_shooting", "passing": "fc_passing",
    "dribbling": "fc_dribbling", "defending": "fc_defending", "physic": "fc_physic",
    "overall": "fc_overall", "potential": "fc_potential",
}

# Full feature set that makes up the normalized player vector.
FEATURE_COLS = PERF_FEATURE_COLS + FC26_STYLE_COLS
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


def apply_current_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay current-season (2025-26 FBref) stats on top of career aggregates.

    For every player matched to the FBref top-5 dataset by name, the goals/assists/
    minutes/cards/appearances used for the feature vector come from 2025-26, so
    similarity and archetypes reflect current form. Unmatched players keep their
    career aggregates. A `stats_source` column records which was used.
    """
    df["stats_source"] = "career"
    try:
        import data_manager as dm
        cur = dm.load_current_player_stats()
    except Exception as exc:
        log(f"WARNING: current FBref stats unavailable, using career only: {exc}")
        return df

    cur = cur.copy()
    cur["_key"] = cur["player_name"].map(_norm_name)
    cur = cur[cur["_key"] != ""].drop_duplicates("_key")

    # Resolve name -> player_id via the highest-value player for each name (collisions).
    base = df[["player_id", "player_name", "market_value_in_eur"]].copy()
    base["_key"] = base["player_name"].map(_norm_name)
    base = (base.sort_values("market_value_in_eur", ascending=False)
                .drop_duplicates("_key"))
    link = base.merge(cur, on="_key", how="inner")

    cur_by_id = {
        int(r.player_id): r for r in link.itertuples(index=False)
    }
    matched = 0
    df = df.set_index("player_id")
    for pid, r in cur_by_id.items():
        if pid not in df.index:
            continue
        df.at[pid, "goals"]          = float(r.Goals)
        df.at[pid, "assists"]        = float(r.Assists)
        df.at[pid, "minutes_played"] = float(r.Minutes_Played)
        df.at[pid, "yellow_cards"]   = float(getattr(r, "Yellow_Cards", 0) or 0)
        df.at[pid, "red_cards"]      = float(getattr(r, "Red_Cards", 0) or 0)
        df.at[pid, "appearances"]    = float(r.Matches_Played)
        df.at[pid, "stats_source"]   = "fbref_2025_26"
        matched += 1
    df = df.reset_index()
    log(f"  Overlaid current 2025-26 stats on {matched} players "
        f"({len(cur)} FBref players, {len(link)} name-linked).")
    return df


def load_fc26(players_df: pd.DataFrame) -> pd.DataFrame:
    """Attach EA FC26 (Sep 2025) playing-style attributes to players by player_id.

    FC26's `long_name` is a legal name (e.g. 'Jude Victor William Bellingham'), so the
    primary match key is date-of-birth + last name, with a full-name fallback. Returns
    player_id + fc_* columns; unmatched players get NaN (filled later by position median).
    """
    empty = pd.DataFrame(columns=["player_id"] + list(FC26_SOURCE_MAP.values()))
    if not os.path.exists(FC26_CSV):
        log(f"WARNING: {FC26_CSV} not found; skipping FC26 attributes.")
        return empty

    fc = pd.read_csv(FC26_CSV, low_memory=False)
    for src in FC26_SOURCE_MAP:
        fc[src] = pd.to_numeric(fc.get(src), errors="coerce")
    fc = fc.rename(columns=FC26_SOURCE_MAP)
    dst = list(FC26_SOURCE_MAP.values())

    def tokens(name) -> set:
        # FC26 legal names append extra surnames ("Mbappé Lottin") and spell some names
        # differently ("Håland"), so match on token overlap rather than exact surname.
        return {t for t in _norm_name(name).split() if len(t) >= 2}

    def fuzzy_overlap(ptok: set, ftok: set) -> int:
        # Count shared name tokens, treating a shared 4-char prefix as a match so
        # nicknames align with legal names ("pedri"~"pedro", "rodri"~"rodrigo").
        n = 0
        for a in ptok:
            for b in ftok:
                if a == b or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]):
                    n += 1
                    break
        return n

    bad = {"", "nan", "nat", "none"}
    # Index FC26 rows by date of birth -> list of (token set, attribute row).
    fc["dobkey"] = fc["dob"].astype(str).str[:10]
    fc["fullkey"] = fc["long_name"].map(_norm_name)
    fc = fc.sort_values("fc_overall", ascending=False, na_position="last")
    dob_index: dict[str, list] = {}
    full_index: dict[str, object] = {}
    for r in fc.itertuples(index=False):
        rec = (tokens(getattr(r, "long_name")), r)
        if r.dobkey not in bad:
            dob_index.setdefault(r.dobkey, []).append(rec)
        if r.fullkey not in bad and r.fullkey not in full_index:
            full_index[r.fullkey] = r

    fields = ["fc_pace", "fc_shooting", "fc_passing", "fc_dribbling",
              "fc_defending", "fc_physic", "fc_overall", "fc_potential"]
    rows = []
    matched = 0
    for p in players_df[["player_id", "player_name", "date_of_birth"]].itertuples(index=False):
        dob = str(p.date_of_birth)[:10]
        ptok = tokens(p.player_name)
        chosen = None
        cands = dob_index.get(dob)
        if cands and ptok:
            # Best name-token overlap among players born the same day (≥1 shared token).
            best_overlap, best = 0, None
            for ftok, frow in cands:
                ov = fuzzy_overlap(ptok, ftok)
                if ov > best_overlap:
                    best_overlap, best = ov, frow
            if best_overlap >= 1:
                chosen = best
        if chosen is None:                       # fallback: exact full-name match
            chosen = full_index.get(_norm_name(p.player_name))
        rec = {"player_id": p.player_id}
        if chosen is not None:
            matched += 1
            for f in fields:
                rec[f] = getattr(chosen, f)
        rows.append(rec)

    out = pd.DataFrame(rows)
    for f in fields:
        if f not in out.columns:
            out[f] = np.nan
    log(f"  FC26 attributes matched to {matched} players (dob + name-token overlap).")
    return out


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


# Position-aware archetype axes: within each position group we run K-Means and name each
# cluster by the feature it scores highest on (relative to that group). One feature -> one
# football-meaningful label; with k = number of axes we get a clean 1:1 naming.
ARCHETYPE_AXES = {
    "Attack": [
        ("fc_shooting",        "Finisher / Poacher"),
        ("fc_pace",            "Pace & dribbling winger"),
        ("assists_per90",      "Creative forward"),
        ("market_value_log",   "Elite all-round forward"),
    ],
    "Midfield": [
        ("fc_defending",       "Defensive midfielder / Ball-winner"),
        ("fc_passing",         "Playmaker / Creator"),
        ("goals_per90",        "Goal-scoring midfielder"),
        ("minutes_played_log", "Box-to-box midfielder"),
    ],
    "Defender": [
        ("fc_passing",         "Ball-playing defender"),
        ("fc_defending",       "Defensive stopper"),
        ("fc_pace",            "Pace-reliant defender"),
        ("height_in_cm",       "Aerial / physical defender"),
    ],
}


def _name_group_clusters(sub_X, axes, fidx):
    """K-Means within one position group; map each cluster to a distinct axis label."""
    k = min(len(axes), len(sub_X))
    if k < 1:
        return ["General profile"] * len(sub_X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cl = km.fit_predict(sub_X)
    # Mean (global z-scored) value per axis feature, per cluster.
    cluster_axis = {
        c: {feat: float(sub_X[cl == c][:, fidx[feat]].mean()) for feat, _ in axes}
        for c in range(k)
    }
    # Greedy 1:1 assignment: each axis label goes to its strongest unclaimed cluster.
    assigned, used = {}, set()
    for feat, name in axes:
        best_c, best_v = None, -1e18
        for c in range(k):
            if c in used:
                continue
            if cluster_axis[c][feat] > best_v:
                best_v, best_c = cluster_axis[c][feat], c
        if best_c is not None:
            assigned[best_c] = name
            used.add(best_c)
    for c in range(k):
        assigned.setdefault(c, axes[-1][1])
    return [assigned[c] for c in cl]


def assign_position_archetypes(df: pd.DataFrame, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Position-aware K-Means archetype label for every player (row-aligned to X)."""
    log("Assigning position-aware K-Means archetypes...")
    fidx = {f: i for i, f in enumerate(feature_names)}
    labels = np.array(["General profile"] * len(df), dtype=object)
    pos = df["position"].astype(str).values
    for group, axes in ARCHETYPE_AXES.items():
        idx = np.where(pos == group)[0]
        if len(idx) == 0:
            continue
        group_labels = _name_group_clusters(X[idx], axes, fidx)
        for j, i in enumerate(idx):
            labels[i] = group_labels[j]
    labels[pos == "Goalkeeper"] = "Goalkeeper"
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
    if os.path.exists(FC26_CSV):
        print_schema(FC26_CSV, "FC26")

    players = load_players(PLAYERS_CSV)
    apps    = load_appearances(APPEARANCES_CSV)

    df = players.merge(apps, on="player_id", how="left")
    log(f"Merged dataset: {len(df)} players")

    df = apply_current_stats(df)

    log("Attaching EA FC26 playing-style attributes...")
    df = df.merge(load_fc26(players), on="player_id", how="left")

    df = engineer(df)
    df = fill_by_position_median(df)

    log("Scaling all features with StandardScaler (z-score)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)

    k = choose_k_elbow(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    log(f"Cluster distribution: {df['cluster'].value_counts().sort_index().to_dict()}")

    # Global cluster is kept for anomaly detection (z-score vs cluster centroid).
    archetypes = build_archetypes(km.cluster_centers_, FEATURE_COLS)
    # Player-facing archetype is position-aware (far more meaningful than the global label).
    df["position_group"] = df["position"]
    df["archetype"] = assign_position_archetypes(df, X, FEATURE_COLS)
    log(f"Position-aware archetype counts: {pd.Series(df['archetype']).value_counts().to_dict()}")

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
