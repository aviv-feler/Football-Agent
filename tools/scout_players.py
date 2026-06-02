"""
tools/scout_players.py
סקאוטינג שחקנים לפי קריטריונים בשפה טבעית — מבוסס שיטות Data Science:
  • פירוק הקריטריונים ל-tokens (עמדה, גיל, רמה, אזור)
  • סינון קשיח (עמדה/גיל/יבשת)
  • דירוג: TF-IDF cosine + Jaccard מול ה-tokens של השאילתה + איכות (דירוג+שווי)
"""

import re
import numpy as np
import pandas as pd
from langchain.tools import tool

from ds_engine import jaccard

_POSITION_KEYWORDS = {
    "Attack":     ["striker", "forward", "winger", "attacker", "cf", "st", "חלוץ", "קיצוני", "התקפ"],
    "Midfield":   ["midfield", "midfielder", "playmaker", "cm", "cdm", "cam", "קשר", "אמצע"],
    "Defender":   ["defender", "defence", "defense", "back", "cb", "fullback", "מגן", "בלם", "הגנה"],
    "Goalkeeper": ["goalkeeper", "keeper", "gk", "שוער"],
}

_CONTINENT_NATIONS = {
    "south america": ["Brazil", "Argentina", "Colombia", "Chile", "Uruguay",
                      "Peru", "Ecuador", "Venezuela", "Paraguay", "Bolivia"],
    "africa": ["Nigeria", "Senegal", "Ghana", "Cote d'Ivoire", "Cameroon",
               "Morocco", "Egypt", "Algeria", "Tunisia", "Mali", "Guinea"],
    "europe": ["France", "Spain", "Germany", "Italy", "England", "Portugal",
               "Netherlands", "Belgium", "Croatia", "Denmark", "Norway", "Sweden"],
    "asia": ["Japan", "Korea, South", "Saudi Arabia", "Iran", "Australia", "Qatar", "Iraq"],
    "north america": ["United States", "Mexico", "Canada", "Costa Rica", "Jamaica"],
}


def make_scout_players_tool(df: pd.DataFrame, features):
    """Factory – features הוא אובייקט PlayerFeatures (TF-IDF + tag sets)."""

    rating_max  = float(df["overall_rating"].max()) or 100.0
    pos_to_iloc = {idx: i for i, idx in enumerate(df.index)}

    @tool
    def scout_players(criteria: str) -> str:
        """
        Find top players matching natural-language scouting criteria using TF-IDF and
        Jaccard similarity over player profiles plus hard filters. Handles position,
        age ('under 23', 'young'), region ('from South America'), and quality ('best','top').
        Examples: 'fast striker under 23 from South America', 'best young midfielders'.
        Input is the criteria as plain text.
        """
        crit  = criteria.lower()
        cand  = df.copy()
        applied   = []
        q_tokens  = []   # tokens שמתארים את השאילתה (ל-TF-IDF ול-Jaccard)

        # ── עמדה ──
        for pos_value, keywords in _POSITION_KEYWORDS.items():
            if any(k in crit for k in keywords):
                m = cand["position"] == pos_value
                if m.any():
                    cand = cand[m]; applied.append(f"עמדה={pos_value}")
                    q_tokens.append(f"pos_{pos_value.lower()}")
                break

        # ── גיל ──
        nums = re.findall(r"\b(1[5-9]|[2-3]\d|4[0-5])\b", crit)
        if re.search(r"under|below|younger than|מתחת|פחות מ", crit) and nums:
            lim = int(nums[0]); cand = cand[cand["age"] < lim]; applied.append(f"גיל<{lim}")
            q_tokens += ["age_young_prospect", "age_developing_player_in_his_early_prime"]
        elif re.search(r"over|above|older than|מעל|יותר מ", crit) and nums:
            lim = int(nums[0]); cand = cand[cand["age"] > lim]; applied.append(f"גיל>{lim}")
            q_tokens += ["age_experienced_player", "age_veteran_player"]
        elif re.search(r"\byoung\b|youngster|prospect|teenage|צעיר", crit):
            cand = cand[(cand["age"] > 0) & (cand["age"] <= 23)]; applied.append("צעיר≤23")
            q_tokens += ["age_young_prospect", "age_developing_player_in_his_early_prime"]
        elif re.search(r"veteran|experienced|ותיק|מנוסה", crit):
            cand = cand[cand["age"] >= 30]; applied.append("מנוסה≥30")
            q_tokens += ["age_experienced_player", "age_veteran_player"]

        # ── איכות ──
        wants_best = any(w in crit for w in ["best", "top", "טוב", "מצטיין", "הכי", "elite"])
        if wants_best:
            q_tokens += ["skill_world-class_elite", "skill_top-tier_excellent"]

        # ── סגנון התקפי ──
        if any(w in crit for w in ["goalscorer", "scorer", "goals", "מבקיע", "כובש"]):
            q_tokens += ["goals_prolific_goalscorer", "goals_high-scoring"]
        if any(w in crit for w in ["playmaker", "assist", "creative", "מבשל", "יצירתי"]):
            q_tokens += ["assists_elite_playmaker", "assists_strong_creator"]

        # ── אזור/לאום ──
        for continent, nations in _CONTINENT_NATIONS.items():
            if continent in crit:
                m = cand["nationality"].isin(nations)
                if m.any():
                    cand = cand[m]; applied.append(f"אזור={continent}")
                break
        else:
            for nat in df["nationality"].dropna().unique():
                if isinstance(nat, str) and len(nat) > 3 and nat.lower() in crit:
                    cand = cand[cand["nationality"] == nat]; applied.append(f"לאום={nat}")
                    q_tokens.append(f"nat_{nat.lower().replace(' ', '_')}")
                    break

        if cand.empty:
            return (f"לא נמצאו שחקנים שתואמים את '{criteria}'. נסה לרכך את הקריטריונים.")

        cand_iloc = np.array([pos_to_iloc[i] for i in cand.index])

        # ── דמיון DS לשאילתה ──
        if not q_tokens:
            q_tokens = [t for t in re.findall(r"[a-zA-Z]+", crit) if len(t) > 2]
        sim_tfidf = features.tfidf_sim_query(q_tokens, cand_iloc)
        q_set     = frozenset(q_tokens)
        sim_jac   = np.array([jaccard(q_set, features.tagsets[p]) for p in cand_iloc])

        # ── איכות (דירוג + שווי log) ──
        rating_norm = cand["overall_rating"].fillna(0).values / rating_max
        mv = cand["market_value_in_eur"].fillna(0).clip(lower=0).values
        mv_norm = np.log1p(mv) / np.log1p(mv.max() if mv.max() > 0 else 1)
        quality = 0.5 * rating_norm + 0.5 * mv_norm

        if wants_best:
            score = 0.20 * sim_tfidf + 0.20 * sim_jac + 0.60 * quality
        else:
            score = 0.35 * sim_tfidf + 0.30 * sim_jac + 0.35 * quality

        cand = cand.assign(_score=score)
        top  = cand.nlargest(5, "_score")

        header = f"**5 השחקנים המתאימים ל-'{criteria}'**"
        if applied:
            header += f"  _(סינון: {', '.join(applied)} | דירוג DS: TF-IDF+Jaccard+איכות)_"
        lines = [header + ":\n"]
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"{rank}. **{row['player_name']}** "
                f"({row.get('sub_position', row.get('position','?'))} | {row.get('club','?')} | "
                f"{row.get('nationality','?')} | גיל {int(row.get('age',0))}) — "
                f"דירוג: {int(row.get('overall_rating',0))} | גולים: {int(row.get('goals',0))} | "
                f"בישולים: {int(row.get('assists',0))} | שווי: €{int(row.get('market_value_in_eur',0)):,}"
            )
        return "\n".join(lines)

    return scout_players
