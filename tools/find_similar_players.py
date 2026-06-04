"""
TOOL 1 - Find similar players.
Method: Cosine similarity on normalized numeric performance vectors.
The player name is used only for lookup; similarity is computed from stats.
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
            return f"Player '{player_name}' was not found. Check spelling or try the full name."

        target = df.iloc[idx]
        pos = target["position"]

        # Candidates: same broad position, excluding the target player.
        cand_mask = (df["position"] == pos) & (df.index != idx)
        cand_ilocs = np.where(cand_mask.values)[0]
        if len(cand_ilocs) < 5:
            cand_ilocs = np.where((df.index != idx).values)[0]

        sims = engine.cosine(idx, cand_ilocs)
        order = np.argsort(sims)[::-1][:5]
        top_ilocs = cand_ilocs[order]

        lines = [
            f"**Top 5 players most similar to {target['player_name']}** "
            f"(position: {pos} | archetype: {target.get('archetype','?')}):\n"
        ]
        for rank, il in enumerate(top_ilocs, 1):
            r = df.iloc[il]
            sim = round(float(sims[cand_ilocs.tolist().index(il)]) * 100, 1)
            lines.append(
                f"{rank}. **{r['player_name']}** "
                f"({r.get('sub_position', r.get('position','?'))} | {r.get('club','?')} | "
                f"{r.get('nationality','?')} | age {int(r.get('age',0) or 0)}) - "
                f"similarity: {sim}% | goals: {int(r.get('goals',0))} | assists: {int(r.get('assists',0))} | "
                f"minutes: {int(r.get('minutes_played',0)):,} | value: EUR {int(r.get('market_value_in_eur',0)):,}"
            )
        lines.append(f"\n🔍 Method: Cosine similarity on {len(engine.feature_names)} "
                     f"normalized performance features (same position group).")
        return "\n".join(lines)

    return find_similar_players
