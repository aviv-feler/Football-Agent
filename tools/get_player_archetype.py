"""
TOOL 3 - Show the player's K-Means archetype.
Method: K-Means clustering, with k selected by the elbow method, on normalized stats.
"""

import numpy as np
from langchain.tools import tool


def make_get_player_archetype_tool(engine):
    df = engine.df

    @tool
    def get_player_archetype(player_name: str) -> str:
        """
        Show which K-Means cluster (player archetype/role) a player belongs to, the
        cluster's defining characteristics, and other notable players in the same cluster.
        Use when asked about a player's role/archetype/profile or 'what type of player'.
        Input is a player name.
        """
        idx = engine.find_index(player_name)
        if idx is None:
            return (
                f"Player '{player_name}' was not found."
                f"\n\n🔍 Method: K-Means clustering (k={engine.k}) on "
                f"{len(engine.feature_names)} normalized features."
            )

        r = df.iloc[idx]
        c = int(r["cluster"])
        arch = engine.archetypes.get(c, f"Cluster {c}")

        # Defining centroid traits: strongest z-score deviations.
        centroid = engine.centroids[c]
        feats = engine.feature_names
        ranked = sorted(range(len(feats)), key=lambda i: abs(centroid[i]), reverse=True)[:4]
        traits = []
        for i in ranked:
            direction = "high" if centroid[i] > 0 else "low"
            traits.append(f"{feats[i]} ({direction}, z={centroid[i]:+.2f})")

        # Other notable players in the same cluster, sorted by market value.
        same = df[(df["cluster"] == c) & (df.index != df.index[idx])]
        notable = same.nlargest(6, "market_value_in_eur")["player_name"].tolist()

        lines = [
            f"**{r['player_name']} - archetype analysis**\n",
            f"- K-Means cluster: #{c} - **{arch}**",
            f"- Position: {r.get('position','?')} ({r.get('sub_position','?')})",
            f"- Defining cluster traits: {', '.join(traits)}",
            f"- Other notable players in this archetype: {', '.join(notable[:5])}",
            f"\n🔍 Method: K-Means clustering (k={engine.k}, selected by the elbow method) "
            f"on {len(feats)} normalized features.",
        ]
        return "\n".join(lines)

    return get_player_archetype
