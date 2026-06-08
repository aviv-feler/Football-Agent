"""
TOOL 3 - Show the player's position-aware K-Means archetype.
Method: K-Means run within the player's position group on normalized feature vectors;
the player's standout traits are read from the z-scored feature vector.
"""

from langchain.tools import tool

from viz import embed_viz

_PROFILE_ATTRS = [("Pace", "fc_pace"), ("Shooting", "fc_shooting"), ("Passing", "fc_passing"),
                  ("Dribbling", "fc_dribbling"), ("Defending", "fc_defending"), ("Physical", "fc_physic")]

# Human-readable labels for the raw feature columns, so the narration reads naturally
# (and translates cleanly to Hebrew, e.g. "goals per 90" → "שערים ל-90 דקות") instead of
# leaking machine names like "goals_per90".
_FEATURE_LABEL = {
    "goals_per90": "goals per 90", "assists_per90": "assists per 90",
    "ga_per90": "goals + assists per 90", "cards_per90": "cards per 90",
    "minutes_played_log": "minutes played", "appearances_log": "appearances",
    "age": "age", "height_in_cm": "height", "market_value_log": "market value",
    "international_caps_log": "international caps", "fc_pace": "pace", "fc_shooting": "shooting",
    "fc_passing": "passing", "fc_dribbling": "dribbling", "fc_defending": "defending",
    "fc_physic": "physical",
}


def _to_int(v):
    try:
        f = float(v)
        return int(round(f)) if f == f else None  # f==f filters NaN
    except (TypeError, ValueError):
        return None


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
            f"{_FEATURE_LABEL.get(feats[i], feats[i].replace('_', ' '))} "
            f"({'high' if z[i] > 0 else 'low'}, z={z[i]:+.2f})"
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
        attrs = [{"k": label, "v": _to_int(r.get(col))}
                 for label, col in _PROFILE_ATTRS if _to_int(r.get(col)) is not None]
        viz = {
            "type": "profile",
            "name": r.get("player_name", player_name),
            "pos": r.get("sub_position") or position,
            "nat": r.get("nationality") or "",
            "club": r.get("club") or "",
            "archetype": role,
            "attrs": attrs,
        }
        return embed_viz("\n".join(lines), viz if attrs else None)

    return get_player_archetype
