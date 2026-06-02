"""
ds_engine.py
מנוע ה-Data Science המרכזי של ScoutAI.

מממש את שיטות ה-DS שנדרשות בקורס:
  • K-Means clustering  – שחקנים דומים נמצאים באותו אשכול (כבר מחושב ב-data_prep)
  • TF-IDF              – דמיון טקסטואלי על "מסמך" קטגוריאלי לכל שחקן (cosine)
  • Jaccard similarity  – דמיון בין קבוצות התגיות (tags) של שחקנים
  • National strength   – חוזק נבחרות נגזר מצבירת נתוני השחקנים (למונדיאל)

החיפוש אינו לפי שם אלא לפי קריטריונים/פרמטרים דומים.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from profile_utils import (
    _age_band, _rating_tier, _goal_tier, _assist_tier, _value_tier,
)


# ─── בניית tokens קטגוריאליים לכל שחקן ───────────────────────────────────────

def _slug(text: str) -> str:
    """ממיר תיאור חופשי ל-token יחיד (רווחים → קו תחתון)."""
    return re.sub(r"\s+", "_", str(text).strip().lower())


def build_tokens(row: pd.Series) -> list[str]:
    """
    בונה רשימת tokens קטגוריאליים המתארת את פרופיל השחקן.
    משמש גם ל-TF-IDF (כמסמך) וגם ל-Jaccard (כקבוצה).
    """
    position = str(row.get("position", "unknown"))
    sub_pos  = str(row.get("sub_position", "")) or "unknown"
    nat      = str(row.get("nationality", "unknown"))
    age      = int(row.get("age", 0) or 0)
    goals    = int(row.get("goals", 0) or 0)
    assists  = int(row.get("assists", 0) or 0)
    rating   = float(row.get("overall_rating", 0) or 0)
    mv       = int(row.get("market_value_in_eur", 0) or 0)

    return [
        f"pos_{_slug(position)}",
        f"subpos_{_slug(sub_pos)}",
        f"nat_{_slug(nat)}",
        f"age_{_slug(_age_band(age))}",
        f"skill_{_slug(_rating_tier(rating))}",
        f"goals_{_slug(_goal_tier(goals, position))}",
        f"assists_{_slug(_assist_tier(assists))}",
        f"value_{_slug(_value_tier(mv))}",
    ]


# ─── אובייקט תכונות מוכן (נבנה פעם אחת ב-startup) ──────────────────────────────

class PlayerFeatures:
    """מחזיק את כל מבני ה-DS המחושבים מראש לחיפוש דמיון."""

    def __init__(self, df: pd.DataFrame):
        print("[ds_engine] בונה tokens, TF-IDF ו-tag sets...", flush=True)
        token_lists       = df.apply(build_tokens, axis=1).tolist()
        self.token_docs   = [" ".join(toks) for toks in token_lists]
        self.tagsets      = [frozenset(toks) for toks in token_lists]

        # TF-IDF: כל token הוא "מילה". token_pattern מתאים ל-token עם קו תחתון.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf      = self.vectorizer.fit_transform(self.token_docs)  # sparse (N×V)
        print(f"[ds_engine] TF-IDF: {self.tfidf.shape}, אוצר מילים: "
              f"{len(self.vectorizer.vocabulary_)} tokens", flush=True)

    # ── דמיון TF-IDF בין אינדקס מטרה לקבוצת מועמדים ──
    def tfidf_sim(self, target_idx: int, cand_positions: np.ndarray) -> np.ndarray:
        return cosine_similarity(self.tfidf[target_idx], self.tfidf[cand_positions])[0]

    # ── דמיון TF-IDF בין שאילתת טקסט חופשי לקבוצת מועמדים ──
    def tfidf_sim_query(self, query_tokens: list[str], cand_positions: np.ndarray) -> np.ndarray:
        q_vec = self.vectorizer.transform([" ".join(query_tokens)])
        return cosine_similarity(q_vec, self.tfidf[cand_positions])[0]

    # ── Jaccard בין קבוצת מטרה לקבוצות מועמדים ──
    def jaccard_to(self, target_set: frozenset, cand_positions: np.ndarray) -> np.ndarray:
        out = np.empty(len(cand_positions))
        for i, pos in enumerate(cand_positions):
            out[i] = jaccard(target_set, self.tagsets[pos])
        return out


def jaccard(a: frozenset, b: frozenset) -> float:
    """דמיון Jaccard בין שתי קבוצות: |חיתוך| / |איחוד|."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ─── חוזק נבחרות לאומיות (למונדיאל) ───────────────────────────────────────────

# מיפוי שמות נבחרות בלוח המונדיאל → ערכי nationality בנתוני השחקנים
NATION_MAP = {
    "usa": "United States", "korea republic": "Korea, South",
    "ir iran": "Iran", "türkiye": "Turkey", "turkiye": "Turkey",
    "czechia": "Czech Republic", "côte d’ivoire": "Cote d'Ivoire",
    "cote d'ivoire": "Cote d'Ivoire", "congo dr": "DR Congo",
    "cabo verde": "Cape Verde", "curaçao": "Curacao",
    "bosnia & herzegovina": "Bosnia-Herzegovina",
}


def normalize_nation(name: str, known_nations: set) -> str | None:
    """ממפה שם נבחרת לערך ה-nationality המתאים בנתונים."""
    if not name:
        return None
    raw = name.strip()
    low = raw.lower()
    if raw in known_nations:
        return raw
    if low in NATION_MAP and NATION_MAP[low] in known_nations:
        return NATION_MAP[low]
    # התאמה case-insensitive
    for n in known_nations:
        if n.lower() == low:
            return n
    # התאמה חלקית (contains)
    for n in known_nations:
        if low in n.lower() or n.lower() in low:
            return n
    return None


def build_national_strength(df: pd.DataFrame, squad_size: int = 23) -> pd.DataFrame:
    """
    בונה טבלת חוזק לכל נבחרת מתוך צבירת נתוני השחקנים:
    לוקח את מיטב הסגל (לפי דירוג ושווי) ומחשב חוזק כולל.
    """
    print("[ds_engine] בונה טבלת חוזק נבחרות...", flush=True)
    rows = []
    for nation, grp in df.groupby("nationality"):
        if not isinstance(nation, str):
            continue
        top = grp.sort_values(
            ["overall_rating", "market_value_in_eur"], ascending=False
        ).head(squad_size)
        if len(top) < 5:
            continue
        rows.append({
            "nationality": nation,
            "squad_rating": top["overall_rating"].mean(),
            "squad_value":  top["market_value_in_eur"].sum(),
            "attack":       top.nlargest(5, "goals")["goals"].mean(),
            "depth":        len(grp),
        })
    nat_df = pd.DataFrame(rows).set_index("nationality")

    # ניקוד חוזק מנורמל (0..1): שילוב דירוג סגל + שווי (log) + עומק
    r = nat_df["squad_rating"] / nat_df["squad_rating"].max()
    v = np.log1p(nat_df["squad_value"]) / np.log1p(nat_df["squad_value"].max())
    d = np.log1p(nat_df["depth"]) / np.log1p(nat_df["depth"].max())
    nat_df["strength"] = (0.55 * r + 0.35 * v + 0.10 * d)
    print(f"[ds_engine] חוזק חושב ל-{len(nat_df)} נבחרות", flush=True)
    return nat_df
