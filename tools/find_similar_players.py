"""
tools/find_similar_players.py
מציאת שחקנים דומים לפי שיטות Data Science (לא לפי שם):
  • K-Means cluster  – מצמצם למועמדים באותו אשכול התנהגותי
  • Jaccard          – דמיון בין קבוצות התגיות הקטגוריאליות
  • TF-IDF cosine    – דמיון על המסמך הקטגוריאלי
  • Numeric cosine   – קרבה על הפיצ'רים המספריים המנורמלים

השם משמש רק לאיתור שחקן-המוצא; הדירוג עצמו מבוסס פרמטרים.
"""

import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import tool

from ds_engine import jaccard

_FEAT_COLS = [
    "age_scaled", "goals_scaled", "assists_scaled",
    "minutes_played_scaled", "market_value_in_eur_scaled",
    "overall_rating_scaled",
]


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _pick_most_prominent(subset: pd.DataFrame) -> int:
    s = subset.copy()
    s["_rank"] = (s.get("overall_rating", 0).fillna(0) * 1_000_000
                  + s.get("market_value_in_eur", 0).fillna(0))
    return s["_rank"].idxmax()


def _find_player_index(df: pd.DataFrame, names_norm: pd.Series, query: str):
    q = _strip_accents(query)
    exact = df[names_norm == q]
    if not exact.empty:
        return _pick_most_prominent(exact)
    contains = df[names_norm.str.contains(q, na=False, regex=False)]
    if not contains.empty:
        return _pick_most_prominent(contains)
    for part in q.split():
        if len(part) < 3:
            continue
        m = df[names_norm.str.contains(part, na=False, regex=False)]
        if not m.empty:
            return _pick_most_prominent(m)
    return None


def make_find_similar_players_tool(df: pd.DataFrame, features):
    """Factory – features הוא אובייקט PlayerFeatures (TF-IDF + tag sets)."""

    names_norm  = df["player_name"].fillna("").map(_strip_accents)
    has_feats   = all(c in df.columns for c in _FEAT_COLS)
    feat_matrix = df[_FEAT_COLS].fillna(0).values if has_feats else None
    pos_to_iloc = {idx: i for i, idx in enumerate(df.index)}  # label → מיקום מספרי

    @tool
    def find_similar_players(player_name: str) -> str:
        """
        Find the 5 most similar players to a given player using clustering, Jaccard and
        TF-IDF similarity on their statistical/categorical profile (NOT by name).
        Use when the user asks for players similar to / like a specific player.
        Input is a player name.
        """
        target_idx = _find_player_index(df, names_norm, player_name)
        if target_idx is None:
            return (f"לא נמצא שחקן בשם '{player_name}'. "
                    "בדוק איות או נסה שם מלא (למשל 'Kylian Mbappé').")

        target      = df.loc[target_idx]
        target_pos  = str(target.get("position", ""))
        target_clu  = target.get("cluster", -1)
        tgt_iloc    = pos_to_iloc[target_idx]

        # ── שלב 1: צמצום לפי אשכול K-Means + אותה עמדה ──
        mask = (df["cluster"] == target_clu) & (df["position"] == target_pos) & (df.index != target_idx)
        cand = df[mask]
        if len(cand) < 5:  # נפילה אחורה: אותה עמדה בלבד
            mask = (df["position"] == target_pos) & (df.index != target_idx)
            cand = df[mask]
        cand_iloc = np.array([pos_to_iloc[i] for i in cand.index])

        # ── שלב 2: שלוש מדדי דמיון ──
        sim_tfidf = features.tfidf_sim(tgt_iloc, cand_iloc)               # TF-IDF
        sim_jac   = features.jaccard_to(features.tagsets[tgt_iloc], cand_iloc)  # Jaccard
        if feat_matrix is not None:
            dists    = np.linalg.norm(feat_matrix[cand_iloc] - feat_matrix[tgt_iloc], axis=1)
            sim_num  = 1.0 / (1.0 + dists)
        else:
            sim_num = np.zeros(len(cand_iloc))

        # ── שלב 3: שקלול ──
        score = 0.40 * sim_jac + 0.35 * sim_tfidf + 0.25 * sim_num
        order = np.argsort(score)[::-1][:5]

        lines = [
            f"**5 השחקנים הדומים ביותר ל-{target['player_name']}**",
            f"_(שיטה: אשכול K-Means #{int(target_clu)} + Jaccard + TF-IDF | "
            f"עמדה: {target_pos})_\n",
        ]
        for rank, o in enumerate(order, 1):
            row = cand.iloc[o]
            lines.append(
                f"{rank}. **{row['player_name']}** "
                f"({row.get('sub_position', row.get('position','?'))} | {row.get('club','?')} | "
                f"{row.get('nationality','?')} | גיל {int(row.get('age',0))}) — "
                f"דמיון: {round(float(score[o])*100,1)}% "
                f"(Jaccard {round(float(sim_jac[o])*100)}% · TF-IDF {round(float(sim_tfidf[o])*100)}%) | "
                f"דירוג: {int(row.get('overall_rating',0))} | גולים: {int(row.get('goals',0))} | "
                f"בישולים: {int(row.get('assists',0))} | שווי: €{int(row.get('market_value_in_eur',0)):,}"
            )
        return "\n".join(lines)

    return find_similar_players
