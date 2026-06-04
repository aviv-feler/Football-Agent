"""
tools/get_player_archetype.py
TOOL 3 — הצגת ה-K-Means clustering: לאיזה ארכיטיפ/תפקיד שייך שחקן.
שיטה: K-Means clustering (k נבחר ב-elbow) על וקטורי ביצועים מנורמלים.
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
                f"לא נמצא שחקן בשם '{player_name}'."
                f"\n\n🔍 Method: K-Means clustering (k={engine.k}) on "
                f"{len(engine.feature_names)} normalized features."
            )

        r = df.iloc[idx]
        c = int(r["cluster"])
        arch = engine.archetypes.get(c, f"Cluster {c}")

        # מאפייני מרכז האשכול: הפיצ'רים הבולטים ביותר (z מול הממוצע הכללי)
        centroid = engine.centroids[c]
        feats = engine.feature_names
        ranked = sorted(range(len(feats)), key=lambda i: abs(centroid[i]), reverse=True)[:4]
        traits = []
        for i in ranked:
            direction = "גבוה" if centroid[i] > 0 else "נמוך"
            traits.append(f"{feats[i]} ({direction}, z={centroid[i]:+.2f})")

        # שחקנים בולטים נוספים באותו אשכול (לפי שווי שוק)
        same = df[(df["cluster"] == c) & (df.index != df.index[idx])]
        notable = same.nlargest(6, "market_value_in_eur")["player_name"].tolist()

        lines = [
            f"**{r['player_name']} — ניתוח ארכיטיפ**\n",
            f"• אשכול K-Means: #{c} — **{arch}**",
            f"• עמדה: {r.get('position','?')} ({r.get('sub_position','?')})",
            f"• מאפייני האשכול המבדילים: {', '.join(traits)}",
            f"• שחקנים בולטים נוספים בארכיטיפ זה: {', '.join(notable[:5])}",
            f"\n🔍 Method: K-Means clustering (k={engine.k}, נבחר בשיטת המרפק) "
            f"על {len(feats)} פיצ'רים מנורמלים.",
        ]
        return "\n".join(lines)

    return get_player_archetype
