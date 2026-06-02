"""
data_prep.py
הכנת נתונים ל-ScoutAI - מנקה, ממזג ומחשב embeddings לשחקנים.

קבצי קלט:
  data/fifa_players.csv  - נתוני FIFA (גיל, דירוג, עמדה, לאומיות, שווי שוק)
  data/players.csv       - נתוני Transfermarkt (player_id, שווי שוק, מועדון)
  data/appearances.csv   - ביצועים לפי משחק (גולים, בישולים, דקות)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

# ─── קבועים ───────────────────────────────────────────────────────────────────
FIFA_CSV        = "data/fifa_players.csv"
PLAYERS_CSV     = "data/players.csv"
APPEARANCES_CSV = "data/appearances.csv"
OUTPUT_CSV      = "data/players_clean.csv"
EMBEDDINGS_FILE = "data/embeddings.npy"
N_CLUSTERS      = 5
MODEL_NAME      = "all-MiniLM-L6-v2"

NUMERIC_FEATURES = [
    "age", "goals", "assists", "minutes_played",
    "market_value_in_eur", "overall_rating",
    "yellow_cards", "red_cards",
]


def log(msg: str) -> None:
    print(f"[data_prep] {msg}", flush=True)


# ─── טעינת FIFA players ───────────────────────────────────────────────────────

def load_fifa(path: str) -> pd.DataFrame:
    log(f"טוען fifa_players מ-{path}…")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # שמות עמודות ב-fifa_players.csv
    rename = {}
    if "full_name"  in df.columns: rename["full_name"]  = "player_name"
    elif "name"     in df.columns: rename["name"]        = "player_name"
    if "positions"  in df.columns: rename["positions"]   = "position"
    if "nationality"in df.columns: pass  # כבר נכון
    if "value_euro" in df.columns: rename["value_euro"]  = "market_value_in_eur"
    if "overall_rating" not in df.columns and "overall" in df.columns:
        rename["overall"] = "overall_rating"

    df = df.rename(columns=rename)

    # חישוב גיל מ-birth_date אם 'age' חסר
    if "age" not in df.columns and "birth_date" in df.columns:
        df["age"] = pd.to_datetime(df["birth_date"], errors="coerce").apply(
            lambda d: (pd.Timestamp.now() - d).days // 365 if pd.notna(d) else np.nan
        )

    log(f"  {len(df)} שחקנים, עמודות: {list(df.columns[:12])}")
    return df


# ─── טעינת Appearances (ביצועים) ─────────────────────────────────────────────

def load_appearances(path: str) -> pd.DataFrame:
    log(f"טוען appearances מ-{path} (עשוי לקחת כמה שניות)…")
    # קוראים רק את העמודות הנחוצות כדי לחסוך זיכרון
    needed = ["player_id", "player_name", "goals", "assists",
              "minutes_played", "yellow_cards", "red_cards"]
    df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False)

    # אגרגציה לפי שחקן - סכום עבור כמויות
    agg = {c: "sum" for c in ["goals","assists","minutes_played","yellow_cards","red_cards"] if c in df.columns}
    group_col = "player_id" if "player_id" in df.columns else "player_name"
    df = df.groupby(group_col, as_index=False).agg(agg)

    log(f"  {len(df)} שחקנים ייחודיים לאחר אגרגציה")
    return df


# ─── טעינת Players (Transfermarkt) ───────────────────────────────────────────

def load_players_tm(path: str) -> pd.DataFrame:
    log(f"טוען players (Transfermarkt) מ-{path}…")
    needed = ["player_id","first_name","last_name","name",
              "country_of_citizenship","sub_position","position",
              "market_value_in_eur","current_club_name","date_of_birth","height_in_cm"]
    df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False)

    rename = {}
    if "country_of_citizenship" in df.columns:
        rename["country_of_citizenship"] = "nationality"
    if "current_club_name" in df.columns:
        rename["current_club_name"] = "club"
    df = df.rename(columns=rename)

    # שם מלא
    if "name" in df.columns:
        df["player_name"] = df["name"]
    elif "first_name" in df.columns and "last_name" in df.columns:
        df["player_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
        df["player_name"] = df["player_name"].str.strip()

    # גיל מ-date_of_birth
    if "date_of_birth" in df.columns:
        df["age"] = pd.to_datetime(df["date_of_birth"], errors="coerce").apply(
            lambda d: (pd.Timestamp.now() - d).days // 365 if pd.notna(d) else np.nan
        )

    log(f"  {len(df)} שחקנים, עמודות: {list(df.columns[:10])}")
    return df


# ─── מיזוג ───────────────────────────────────────────────────────────────────

def merge_all(tm: pd.DataFrame, appearances: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    log("ממזג את כל מקורות הנתונים…")

    # מיזוג 1: Transfermarkt + Appearances על player_id
    if "player_id" in tm.columns and "player_id" in appearances.columns:
        df = tm.merge(appearances, on="player_id", how="left", suffixes=("", "_app"))
        for c in ["goals","assists","minutes_played","yellow_cards","red_cards"]:
            c_app = c + "_app"
            if c_app in df.columns:
                df[c] = df[c].fillna(df[c_app])
                df.drop(columns=[c_app], inplace=True)
    else:
        df = tm.copy()
        for c in ["goals","assists","minutes_played","yellow_cards","red_cards"]:
            if c not in df.columns:
                df[c] = 0.0

    # מיזוג 2: הוספת overall_rating מ-FIFA על ידי התאמת שם
    if "overall_rating" in fifa.columns and "player_name" in fifa.columns:
        fifa_slim = fifa[["player_name","overall_rating","potential"]].copy()
        # נרמול שמות למיזוג
        df["_name_key"]       = df["player_name"].str.lower().str.strip()
        fifa_slim["_name_key"]= fifa_slim["player_name"].str.lower().str.strip()
        df = df.merge(
            fifa_slim[["_name_key","overall_rating","potential"]],
            on="_name_key", how="left"
        )
        df.drop(columns=["_name_key"], inplace=True)

    log(f"  {len(df)} שחקנים לאחר מיזוג מלא")
    return df


# ─── הנדסת תכונות ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("מנדס תכונות ומנרמל…")

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = imputer.fit_transform(df[NUMERIC_FEATURES])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[NUMERIC_FEATURES])
    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"{c}_scaled" for c in NUMERIC_FEATURES],
        index=df.index,
    )
    df = pd.concat([df, scaled_df], axis=1)
    return df


# ─── K-Means clustering ───────────────────────────────────────────────────────

def cluster_players(df: pd.DataFrame) -> pd.DataFrame:
    log(f"מריץ K-Means עם k={N_CLUSTERS}…")
    scaled_cols = [f"{c}_scaled" for c in NUMERIC_FEATURES]
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[scaled_cols])
    log(f"  התפלגות: {df['cluster'].value_counts().sort_index().to_dict()}")
    return df


# ─── פרופיל טקסט לכל שחקן ───────────────────────────────────────────────────
# הטקסט מתאר סגנון ורמה (ללא שם) כדי שה-embedding יתפוס דמיון בסגנון משחק
from profile_utils import build_profile_text  # noqa: E402


# ─── Sentence Embeddings ──────────────────────────────────────────────────────

def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    log(f"מחשב sentence embeddings עם {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    texts = df["profile_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    log(f"  צורת embeddings: {embeddings.shape}")
    return embeddings


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs("data", exist_ok=True)

    # ── טעינה ──
    missing = [f for f in [FIFA_CSV, PLAYERS_CSV, APPEARANCES_CSV] if not os.path.exists(f)]
    if missing:
        log(f"ERROR: קבצים חסרים: {missing}")
        sys.exit(1)

    tm          = load_players_tm(PLAYERS_CSV)
    appearances = load_appearances(APPEARANCES_CSV)
    fifa        = load_fifa(FIFA_CSV)

    # ── מיזוג ──
    df = merge_all(tm, appearances, fifa)

    # ── עיבוד ──
    df = engineer_features(df)
    df = cluster_players(df)

    # ── פרופיל טקסט ──
    log("בונה פרופילי טקסט…")
    df["profile_text"] = df.apply(build_profile_text, axis=1)

    # ── שמירה ──
    df.to_csv(OUTPUT_CSV, index=False)
    log(f"שמר {len(df)} שחקנים ל-{OUTPUT_CSV}")

    embeddings = compute_embeddings(df)
    np.save(EMBEDDINGS_FILE, embeddings)
    log(f"שמר embeddings ל-{EMBEDDINGS_FILE}")

    log("הכנת נתונים הושלמה בהצלחה!")


if __name__ == "__main__":
    main()
