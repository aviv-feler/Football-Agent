"""
TOOL 3 - Show the player's position-aware K-Means archetype.
Method: K-Means run within the player's position group on normalized feature vectors;
the player's standout traits are read from the z-scored feature vector.
"""

from langchain.tools import tool


def make_get_player_archetype_tool(engine):
    df = engine.df

    @tool
    def get_player_archetype(player_name: str) -> str:
        """
        Show a player's archetype/role (from position-aware K-Means), the traits that
        define them, and other notable players of the same archetype.
        Use when asked about a player's role/archetype/profile or 'what type of player'.
        Input is a player name.
        """
        feats = engine.feature_names
        idx = engine.find_index(player_name)
        if idx is None:
            return (
                f"Player '{player_name}' was not found."
                f"\n\n🔍 Method: Position-aware K-Means clustering on {len(feats)} normalized features."
            )

        r = df.iloc[idx]
        role = r.get("archetype") or "General profile"
        position = r.get("position", "?")

        # Defining traits: the player's strongest normalized (z-scored) features.
        z = engine.X[idx]
        ranked = sorted(range(len(feats)), key=lambda i: abs(z[i]), reverse=True)[:4]
        traits = [
            f"{feats[i]} ({'high' if z[i] > 0 else 'low'}, z={z[i]:+.2f})"
            for i in ranked
        ]

        # Other notable players sharing this archetype within the same position group.
        same = df[(df["archetype"] == role) & (df["position"] == position) & (df.index != df.index[idx])]
        notable = same.nlargest(6, "market_value_in_eur")["player_name"].tolist()

        lines = [
            f"**{r['player_name']} — archetype analysis**\n",
            f"- Position: {position} ({r.get('sub_position', '?')})",
            f"- K-Means archetype: **{role}**",
            f"- Defining traits (vs all players): {', '.join(traits)}",
            f"- Other notable {role}: {', '.join(notable[:5]) if notable else 'n/a'}",
            f"\n🔍 Method: Position-aware K-Means clustering on {len(feats)} normalized features.",
        ]
        return "\n".join(lines)

    return get_player_archetype
