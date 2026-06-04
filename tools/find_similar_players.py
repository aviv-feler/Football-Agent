"""
tools/find_similar_players.py
TOOL 1 — מציאת שחקנים דומים.
שיטה: Cosine similarity על וקטורי ביצועים מספריים מנורמלים (z-score).
השם משמש רק לאיתור שחקן-המוצא; הדמיון מחושב על הסטטיסטיקות.
"""

import numpy as np
from langchain.tools import tool


def make_find_similar_players_tool(engine):
    df = engine.df

    @tool
    def find_similar_players(player_name: str) -> str:
        """
        Find the 5 most similar players to a given player using COSINE SIMILARITY on
        their normalized numeric performance vectors (goals/90, assists/90, minutes,
        market value, age, etc.). Restricted to the same position group.
        Use when the user asks for players similar to / like a specific player.
        Input is a player name.
        """
        idx = engine.find_index(player_name)
        if idx is None:
            return (
                f"לא נמצא שחקן בשם '{player_name}'. בדוק איות או נסה שם מלא."
                f"\n\n🔍 Method: Cosine similarity on {len(engine.feature_names)} "
                "normalized performance features (same position group)."
            )

        target = df.iloc[idx]
        pos = target["position"]

        # מועמדים: אותה עמדה, לא השחקן עצמו
        cand_mask = (df["position"] == pos) & (df.index != idx)
        cand_ilocs = np.where(cand_mask.values)[0]
        if len(cand_ilocs) < 5:
            cand_ilocs = np.where((df.index != idx).values)[0]

        sims = engine.cosine(idx, cand_ilocs)
        order = np.argsort(sims)[::-1][:5]
        top_ilocs = cand_ilocs[order]

        lines = [
            f"**5 השחקנים הדומים ביותר ל-{target['player_name']}** "
            f"(עמדה: {pos} | ארכיטיפ: {target.get('archetype','?')}):\n"
        ]
        for rank, il in enumerate(top_ilocs, 1):
            r = df.iloc[il]
            sim = round(float(sims[cand_ilocs.tolist().index(il)]) * 100, 1)
            lines.append(
                f"{rank}. **{r['player_name']}** "
                f"({r.get('sub_position', r.get('position','?'))} | {r.get('club','?')} | "
                f"{r.get('nationality','?')} | גיל {int(r.get('age',0) or 0)}) — "
                f"דמיון: {sim}% | גולים: {int(r.get('goals',0))} | בישולים: {int(r.get('assists',0))} | "
                f"דקות: {int(r.get('minutes_played',0)):,} | שווי: €{int(r.get('market_value_in_eur',0)):,}"
            )
        lines.append(f"\n🔍 Method: Cosine similarity on {len(engine.feature_names)} "
                     f"normalized performance features (same position group).")
        return "\n".join(lines)

    return find_similar_players
