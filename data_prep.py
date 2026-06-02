"""
data_prep.py
הכנת נתונים ל-ScoutAI — בניית מטריצת פיצ'רים מספרית מנורמלת לכל שחקן.

שיטות Data Science:
  • Feature engineering – סטטיסטיקות per-90 + פרופיל (גיל, גובה, שווי, קאפים)
  • טיפול בחוסרים – מילוי לפי חציון קבוצת-העמדה (לא 0)
  • נרמול – StandardScaler (z-score) — קריטי לדמיון תקין
  • K-Means – אשכולות "ארכיטיפ/תפקיד" עם בחירת k בשיטת המרפק (elbow)

קלט:  data/players.csv, data/appearances.csv  (Kaggle)
פלט:  data/players_clean.csv, data/player_features.npy, data/feature_meta.json
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

# הפיצ'רים המספריים שמרכיבים את וקטור הביצועים (כולם זמינים כמעט לכל השחקנים)
FEATURE_COLS = [
    "goals_per90", "assists_per90", "ga_per90", "cards_per90",
    "minutes_played_log", "appearances_log", "age", "height_in_cm",
    "market_value_log", "international_caps_log",
]
# פיצ'רים שעוברים log1p לפני נרמול (מוטים מאוד)
LOG_COLS = {
    "minutes_played": "minutes_played_log",
    "appearances":    "appearances_log",
    "market_value_in_eur": "market_value_log",
    "international_caps":   "international_caps_log",
}


def log(msg: str) -> None:
    print(f"[data_prep] {msg}", flush=True)


# ─── טעינה ───────────────────────────────────────────────────────────────────

def print_schema(path: str, name: str):
    df = pd.read_csv(path, nrows=3, low_memory=False)
    log(f"סכמת {name} ({path}):")
    log(f"  עמודות: {list(df.columns)}")


def load_players(path: str) -> pd.DataFrame:
    log("טוען players (Transfermarkt)...")
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
    log(f"  {len(df)} שחקנים")
    return df


def load_appearances(path: str) -> pd.DataFrame:
    log("טוען appearances ומאגרג לכל שחקן...")
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
    log(f"  {len(agg)} שחקנים עם נתוני משחקים")
    return agg


# ─── הנדסת פיצ'רים ───────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    log("מנדס פיצ'רים (per-90 + לוג טרנספורמציות)...")

    for c in ["goals", "assists", "minutes_played", "yellow_cards",
              "red_cards", "appearances", "international_caps"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # סטטיסטיקות per-90 (מנורמלות לזמן משחק). דורש מינימום דקות כדי להימנע מרעש.
    mins = df["minutes_played"].clip(lower=0)
    safe = mins.where(mins >= 90, np.nan)  # פחות מ-90 דקות → לא אמין
    df["goals_per90"]   = (df["goals"]   / (safe / 90)).fillna(0)
    df["assists_per90"] = (df["assists"] / (safe / 90)).fillna(0)
    df["ga_per90"]      = df["goals_per90"] + df["assists_per90"]
    df["cards_per90"]   = ((df["yellow_cards"] + df["red_cards"]) / (safe / 90)).fillna(0)

    # winsorize per-90 כדי שמדגמים זעירים לא ישתלטו (חיתוך באחוזון 99)
    for c in ["goals_per90", "assists_per90", "ga_per90", "cards_per90"]:
        cap = df[c].quantile(0.99)
        df[c] = df[c].clip(upper=cap)

    # לוג טרנספורמציות לפיצ'רים מוטים
    for raw, logged in LOG_COLS.items():
        df[logged] = np.log1p(pd.to_numeric(df[raw], errors="coerce").fillna(0).clip(lower=0))

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["height_in_cm"] = pd.to_numeric(df["height_in_cm"], errors="coerce")
    return df


def fill_by_position_median(df: pd.DataFrame) -> pd.DataFrame:
    """מילוי חוסרים לפי חציון קבוצת-העמדה (ולא 0)."""
    log("ממלא חוסרים לפי חציון קבוצת-העמדה...")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        med_by_pos = df.groupby("position")[col].transform("median")
        df[col] = df[col].fillna(med_by_pos)
        df[col] = df[col].fillna(df[col].median())  # גיבוי גלובלי
    return df


# ─── אשכולות + elbow ─────────────────────────────────────────────────────────

def choose_k_elbow(X: np.ndarray, k_range=range(2, 11)) -> int:
    """בוחר k בשיטת המרפק: הנקודה עם המרחק המקסימלי מהקו ישר inertia(k)."""
    log("מריץ שיטת המרפק (elbow) לבחירת k...")
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        log(f"  k={k}: inertia={km.inertia_:,.0f}")

    ks = np.array(list(k_range))
    iner = np.array(inertias)
    # מרחק כל נקודה מהקו המחבר את הקצוות → המרפק הוא המרחק המקסימלי
    p1, p2 = np.array([ks[0], iner[0]]), np.array([ks[-1], iner[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for i in range(len(ks)):
        p = np.array([ks[i], iner[i]]) - p1
        proj = p - np.dot(p, line) * line
        dists.append(np.linalg.norm(proj))
    best_k = int(ks[int(np.argmax(dists))])
    log(f"  ✓ נבחר k={best_k} (מרפק)")
    return best_k


def build_archetypes(centroids: np.ndarray, feature_names: list[str]) -> dict:
    """
    נותן שם ארכיטיפ ייחודי לכל אשכול לפי הפיצ'ר המבדיל ביותר במרכז (ב-z-score).
    מבטיח שמות מובחנים בין האשכולות.
    """
    idx = {n: i for i, n in enumerate(feature_names)}
    # פיצ'ר → (שם כשהוא גבוה, שם כשהוא נמוך)
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
        # מדרגים פיצ'רים לפי |z| יורד ובוחרים את הראשון שנותן תווית חדשה
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)
    for p in (PLAYERS_CSV, APPEARANCES_CSV):
        if not os.path.exists(p):
            log(f"ERROR: {p} לא נמצא.")
            sys.exit(1)

    # אימות סכמה — מדפיסים את העמודות האמיתיות לפני בנייה
    print_schema(PLAYERS_CSV, "players")
    print_schema(APPEARANCES_CSV, "appearances")
    if os.path.exists(FIFA_CSV):
        print_schema(FIFA_CSV, "fifa_players")

    players = load_players(PLAYERS_CSV)
    apps    = load_appearances(APPEARANCES_CSV)

    df = players.merge(apps, on="player_id", how="left")
    log(f"מוזג: {len(df)} שחקנים")

    df = engineer(df)
    df = fill_by_position_median(df)

    # ── נרמול (z-score) ──
    log("מנרמל את כל הפיצ'רים עם StandardScaler (z-score)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)

    # ── K-Means עם elbow ──
    k = choose_k_elbow(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    log(f"התפלגות אשכולות: {df['cluster'].value_counts().sort_index().to_dict()}")

    # ── תיוג ארכיטיפים (ייחודי לכל אשכול) ──
    archetypes = build_archetypes(km.cluster_centers_, FEATURE_COLS)
    df["archetype"] = df["cluster"].map(archetypes)
    log(f"ארכיטיפים: {archetypes}")

    # ── עמודות עזר ל-Jaccard / תצוגה ──
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 21, 25, 29, 33, 99],
                              labels=["U21", "21-25", "26-29", "30-33", "33+"]).astype(str)
    df["value_tier"] = pd.cut(df["market_value_in_eur"],
                              bins=[-1, 1e6, 5e6, 20e6, 50e6, 1e12],
                              labels=["low", "modest", "mid", "high", "elite"]).astype(str)

    # ── שמירה ──
    df.to_csv(OUTPUT_CSV, index=False)
    log(f"שמר {len(df)} שחקנים → {OUTPUT_CSV}")

    np.save(FEATURES_NPY, X)
    log(f"שמר מטריצת פיצ'רים מנורמלת {X.shape} → {FEATURES_NPY}")

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
    log(f"שמר metadata → {META_JSON}")
    log("הכנת נתונים הושלמה בהצלחה!")


if __name__ == "__main__":
    main()
