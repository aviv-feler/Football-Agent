"""
ds_engine.py
מנוע ה-Data Science של ScoutAI — מבוסס וקטורי ביצועים מספריים מנורמלים.

שיטות:
  • Cosine similarity / Euclidean – דמיון בין שחקנים כווקטורים מספריים
  • K-Means archetypes           – אשכול תפקיד/ארכיטיפ לכל שחקן
  • Z-score from cluster centroid – זיהוי חריגות
  • Jaccard                       – דמיון בין קבוצות תכונות קטגוריאליות
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV     = "data/players_clean.csv"
FEATURES_NPY = "data/player_features.npy"
META_JSON    = "data/feature_meta.json"


def jaccard(a: set, b: set) -> float:
    """דמיון Jaccard: |חיתוך| / |איחוד|."""
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class DSEngine:
    """מחזיק את כל מבני ה-DS המחושבים מראש ומספק חישובי דמיון/חריגות."""

    def __init__(self, df: pd.DataFrame, X: np.ndarray, meta: dict):
        self.df = df.reset_index(drop=True)
        self.X = X                                  # מטריצת פיצ'רים מנורמלת (N×F)
        self.feature_names = meta["feature_names"]
        self.k = meta["k"]
        self.centroids = np.array(meta["centroids"])
        self.archetypes = {int(k): v for k, v in meta["archetypes"].items()}

        # סטטיסטיקות per-cluster לחישוב Z-score (ממוצע וסטיית תקן של כל פיצ'ר באשכול)
        self.cluster_mean, self.cluster_std = {}, {}
        clusters = self.df["cluster"].values
        for c in np.unique(clusters):
            rows = self.X[clusters == c]
            self.cluster_mean[int(c)] = rows.mean(axis=0)
            self.cluster_std[int(c)]  = rows.std(axis=0) + 1e-9

        # מיפוי שם מנורמל → אינדקס (לאיתור שחקן מטרה)
        import unicodedata
        def norm(s):
            if not isinstance(s, str): return ""
            nf = unicodedata.normalize("NFKD", s)
            return "".join(ch for ch in nf if not unicodedata.combining(ch)).lower().strip()
        self._norm = norm
        self.names_norm = self.df["player_name"].fillna("").map(norm)

    # ── איתור שחקן לפי שם (השם רק לאיתור, לא לדמיון) ──
    def find_index(self, query: str):
        q = self._norm(query)
        exact = self.df.index[self.names_norm == q]
        if len(exact):
            return self._most_prominent(exact)
        contains = self.df.index[self.names_norm.str.contains(q, na=False, regex=False)]
        if len(contains):
            return self._most_prominent(contains)
        for part in q.split():
            if len(part) < 3:
                continue
            m = self.df.index[self.names_norm.str.contains(part, na=False, regex=False)]
            if len(m):
                return self._most_prominent(m)
        return None

    def _most_prominent(self, idxs):
        """מבין כמה התאמות בוחר את בעל שווי השוק הגבוה ביותר (המוכר ביותר)."""
        sub = self.df.loc[idxs]
        return int(sub["market_value_in_eur"].fillna(0).idxmax())

    # ── Cosine similarity בין מטרה למועמדים ──
    def cosine(self, target_iloc: int, cand_ilocs: np.ndarray) -> np.ndarray:
        return cosine_similarity(self.X[target_iloc].reshape(1, -1), self.X[cand_ilocs])[0]

    # ── Euclidean distance ──
    def euclidean(self, target_iloc: int, cand_ilocs: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.X[cand_ilocs] - self.X[target_iloc], axis=1)

    # ── Cosine מול וקטור-מטרה שרירותי (לסקאוט – content-based) ──
    def cosine_to_vector(self, vec: np.ndarray, cand_ilocs: np.ndarray) -> np.ndarray:
        return cosine_similarity(vec.reshape(1, -1), self.X[cand_ilocs])[0]

    # ── Z-score של שחקן מול מרכז האשכול שלו ──
    def zscores(self, iloc: int) -> dict:
        c = int(self.df.iloc[iloc]["cluster"])
        z = (self.X[iloc] - self.cluster_mean[c]) / self.cluster_std[c]
        return {self.feature_names[i]: float(z[i]) for i in range(len(self.feature_names))}

    # ── קבוצת תכונות קטגוריאליות ל-Jaccard ──
    def trait_set(self, iloc: int) -> set:
        r = self.df.iloc[iloc]
        traits = set()
        for col, prefix in [("position", "pos"), ("sub_position", "subpos"),
                            ("nationality", "nat"), ("foot", "foot"),
                            ("league", "league"), ("age_bucket", "age"),
                            ("value_tier", "val"), ("archetype", "arch")]:
            v = r.get(col)
            if isinstance(v, str) and v and v.lower() != "nan":
                traits.add(f"{prefix}:{v}")
        return traits

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)


def load_engine() -> DSEngine:
    df = pd.read_csv(DATA_CSV, low_memory=False)
    X  = np.load(FEATURES_NPY)
    with open(META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[ds_engine] DSEngine: {X.shape[0]} שחקנים × {X.shape[1]} פיצ'רים, "
          f"k={meta['k']} אשכולות", flush=True)
    return DSEngine(df, X, meta)


# ─── חוזק נבחרות (למונדיאל) — מבוסס שווי שוק + עומק סגל ──────────────────────

NATION_MAP = {
    "usa": "United States", "korea republic": "Korea, South", "ir iran": "Iran",
    "türkiye": "Turkey", "turkiye": "Turkey", "czechia": "Czech Republic",
    "côte d’ivoire": "Cote d'Ivoire", "cote d'ivoire": "Cote d'Ivoire",
    "congo dr": "DR Congo", "cabo verde": "Cape Verde", "curaçao": "Curacao",
    "bosnia & herzegovina": "Bosnia-Herzegovina",
}


def normalize_nation(name: str, known: set):
    if not name:
        return None
    raw, low = name.strip(), name.strip().lower()
    if raw in known:
        return raw
    if low in NATION_MAP and NATION_MAP[low] in known:
        return NATION_MAP[low]
    for n in known:
        if n.lower() == low:
            return n
    for n in known:
        if low in n.lower() or n.lower() in low:
            return n
    return None


def build_national_strength(df: pd.DataFrame, squad: int = 23) -> pd.DataFrame:
    """חוזק נבחרת = צבירת מיטב הסגל לפי שווי שוק (עדכני, אמין)."""
    print("[ds_engine] בונה טבלת חוזק נבחרות (לפי שווי שוק)...", flush=True)
    rows = []
    for nation, grp in df.groupby("nationality"):
        if not isinstance(nation, str):
            continue
        top = grp.nlargest(squad, "market_value_in_eur")
        if len(top) < 5:
            continue
        rows.append({
            "nationality": nation,
            "squad_value_mean": top["market_value_in_eur"].mean(),
            "squad_value_sum":  top["market_value_in_eur"].sum(),
            "top_scorers":      top.nlargest(5, "goals")["goals"].mean(),
            "depth":            len(grp),
        })
    nat = pd.DataFrame(rows).set_index("nationality")
    vmean = np.log1p(nat["squad_value_mean"]) / np.log1p(nat["squad_value_mean"].max())
    vsum  = np.log1p(nat["squad_value_sum"])  / np.log1p(nat["squad_value_sum"].max())
    depth = np.log1p(nat["depth"]) / np.log1p(nat["depth"].max())
    nat["strength"] = 0.6 * vmean + 0.25 * vsum + 0.15 * depth
    print(f"[ds_engine] חוזק חושב ל-{len(nat)} נבחרות", flush=True)
    return nat
