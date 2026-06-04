"""
TOOL 5 - Compare two players with Jaccard similarity.
Method: Jaccard similarity on categorical trait sets.
"""

from langchain.tools import tool
from ds_engine import jaccard


def make_compare_players_jaccard_tool(engine):
    df = engine.df

    @tool
    def compare_players_jaccard(players: str) -> str:
        """
        Compare two players by JACCARD SIMILARITY on their categorical trait sets
        (position, sub-position, nationality, foot, league, age bucket, value tier,
        archetype). Input: the two player names separated by 'vs', 'and', or a comma.
        Example: 'Messi vs Ronaldo'.
        """
        parts = [p.strip() for p in __import__("re").split(r"\s+vs\s+|,|\s+and\s+|\s+ו-?", players) if p.strip()]
        if len(parts) < 2:
            return (
                "Please provide two players, for example: 'Mbappé vs Haaland'."
                "\n\n🔍 Method: Jaccard similarity on categorical trait sets."
            )

        i1, i2 = engine.find_index(parts[0]), engine.find_index(parts[1])
        if i1 is None:
            return (
                f"Player not found: '{parts[0]}'."
                "\n\n🔍 Method: Jaccard similarity on categorical trait sets."
            )
        if i2 is None:
            return (
                f"Player not found: '{parts[1]}'."
                "\n\n🔍 Method: Jaccard similarity on categorical trait sets."
            )

        s1, s2 = engine.trait_set(i1), engine.trait_set(i2)
        j = jaccard(s1, s2)
        shared = sorted(s1 & s2)
        only1  = sorted(s1 - s2)
        only2  = sorted(s2 - s1)

        r1, r2 = df.iloc[i1], df.iloc[i2]
        lines = [
            f"**Jaccard comparison: {r1['player_name']} vs {r2['player_name']}**\n",
            f"Jaccard similarity: **{round(j*100,1)}%** "
            f"(|intersection|={len(s1 & s2)} / |union|={len(s1 | s2)})",
            f"- Shared traits: {', '.join(shared) if shared else 'none'}",
            f"- Unique to {r1['player_name']}: {', '.join(only1) if only1 else 'none'}",
            f"- Unique to {r2['player_name']}: {', '.join(only2) if only2 else 'none'}",
            f"\n🔍 Method: Jaccard similarity on categorical trait sets.",
        ]
        return "\n".join(lines)

    return compare_players_jaccard
