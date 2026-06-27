"""
TOOL 1 - Find similar players.
Method: Cosine similarity on normalized numeric performance vectors.
The player name is used only for lookup; similarity is computed from stats.
"""

import numpy as np
from langchain.tools import tool


def make_find_similar_players_tool(engine):
    df = engine.df

    def method_note():
        return (
            f"🔍 Method: Cosine similarity on {len(engine.feature_names)} "
            "normalized performance features (same position group)."
        )

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
            return f"Player '{player_name}' was not found. Check spelling or try the full name.\n\n{method_note()}"

        target = df.iloc[idx]
        pos = target["position"]

        # Candidates: same broad position, excluding the target player.
        cand_ilocs = np.where(((df["position"] == pos) & (df.index != idx)).values)[0]
        if len(cand_ilocs) < 5:
            cand_ilocs = np.where((df.index != idx).values)[0]

        sims = engine.cosine(idx, cand_ilocs)
        top = np.argsort(sims)[::-1][:5]

        header = (
            f"**Top 5 players most similar to {target['player_name']}** "
            f"(position: {pos} | archetype: {target.get('archetype', '?')}):\n"
        )
        rows = [_format_row(rank, df.iloc[cand_ilocs[i]], sims[i])
                for rank, i in enumerate(top, 1)]
        return "\n".join([header, *rows, "\n" + method_note()])

    return find_similar_players


def _format_row(rank, r, sim):
    return (
        f"{rank}. **{r['player_name']}** "
        f"({r.get('sub_position', r.get('position', '?'))} | {r.get('club', '?')} | "
        f"{r.get('nationality', '?')} | age {int(r.get('age', 0) or 0)}) - "
        f"similarity: {round(float(sim) * 100, 1)}% | goals: {int(r.get('goals', 0))} | "
        f"assists: {int(r.get('assists', 0))} | minutes: {int(r.get('minutes_played', 0)):,} | "
        f"value: EUR {int(r.get('market_value_in_eur', 0)):,}"
    )
