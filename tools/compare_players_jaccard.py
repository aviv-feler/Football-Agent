"""
TOOL 5 - Compare two players with a football-readable side-by-side breakdown.
Primary: real performance attributes + season stats. Secondary: Jaccard playing-style
overlap score. The Jaccard computation is kept but the raw internal trait codes are not
shown to users. The detailed comparison is returned as a structured visual card; the
text reply stays short so it reads cleanly in the chat bubble.
"""

import pandas as pd
from langchain.tools import tool
from ds_engine import jaccard
from viz import embed_viz


def _num(value, default=0):
    try:
        return default if value is None or pd.isna(value) else value
    except Exception:
        return default


_READABLE_TRAIT = {
    "arch":   None,          # skip — shown as archetype line
    "pos":    None,          # skip — too generic
    "subpos": None,          # skip — shown as position
    "league": None,          # skip — internal code
    "nat":    "nationality",
    "foot":   "preferred foot",
    "age":    None,          # skip — shown in stats
    "val":    "market tier",
}

_METHOD = "🔍 Method: Jaccard similarity on categorical traits + attribute comparison."


def make_compare_players_jaccard_tool(engine):
    df = engine.df

    @tool
    def compare_players_jaccard(players: str) -> str:
        """
        Compare two players side by side: performance attributes, real stats (goals,
        assists, minutes), playing-style overlap, and key differences. Use for
        "compare X and Y", "X vs Y", "who is better — X or Y?".
        Input: two player names separated by 'vs', 'and', or a comma.
        Example: 'Mbappé vs Vinicius Jr'.
        """
        import re as _re
        parts = [p.strip() for p in _re.split(r"\s+vs\.?\s+|,\s*|\s+and\s+|\s+ו-?", players) if p.strip()]
        if len(parts) < 2:
            return f"Provide two players, e.g. 'Mbappé vs Vinicius Jr'.\n\n{_METHOD}"

        i1, i2 = engine.find_index(parts[0]), engine.find_index(parts[1])
        if i1 is None:
            return f"Player not found: '{parts[0]}'.\n\n{_METHOD}"
        if i2 is None:
            return f"Player not found: '{parts[1]}'.\n\n{_METHOD}"

        r1, r2 = df.iloc[i1], df.iloc[i2]
        n1, n2 = str(r1["player_name"]), str(r2["player_name"])

        # Jaccard for style overlap — computed but shown as a single clean score.
        s1, s2 = engine.trait_set(i1), engine.trait_set(i2)
        j = jaccard(s1, s2)
        overlap = round(j * 100, 1)

        # Shared readable traits (exclude internal codes).
        readable_shared = []
        for trait in sorted(s1 & s2):
            prefix = trait.split(":")[0] if ":" in trait else ""
            label  = _READABLE_TRAIT.get(prefix, "skip")
            if label is None:
                continue
            value  = trait.split(":", 1)[1] if ":" in trait else trait
            readable_shared.append(f"{label}: {value}" if label != "skip" else None)
        readable_shared = [x for x in readable_shared if x][:4]

        # Numeric attribute bars (real performance attributes, 0-99 scale).
        ATTRS = [
            ("Pace",      "fc_pace"),
            ("Shooting",  "fc_shooting"),
            ("Passing",   "fc_passing"),
            ("Dribbling", "fc_dribbling"),
            ("Defending", "fc_defending"),
            ("Physical",  "fc_physic"),
        ]
        attr_viz = []
        for label, col in ATTRS:
            v1, v2 = int(_num(r1.get(col))), int(_num(r2.get(col)))
            if not v1 and not v2:
                continue
            attr_viz.append({"k": label, "a": v1, "b": v2})

        # Who leads each area (>5 pts), grouped per player for a clean insight line.
        edges = []
        for label, col in ATTRS:
            v1, v2 = int(_num(r1.get(col))), int(_num(r2.get(col)))
            if v1 > v2 + 5:
                edges.append(f"{n1} leads in {label.lower()}")
            elif v2 > v1 + 5:
                edges.append(f"{n2} leads in {label.lower()}")
        edges = edges[:4]

        ovr1, ovr2 = int(_num(r1.get("fc_overall"))), int(_num(r2.get("fc_overall")))
        pot1, pot2 = int(_num(r1.get("fc_potential"))), int(_num(r2.get("fc_potential")))
        g1, g2 = int(_num(r1.get("goals"))), int(_num(r2.get("goals")))
        a1, a2 = int(_num(r1.get("assists"))), int(_num(r2.get("assists")))
        m1, m2 = int(_num(r1.get("minutes_played"))), int(_num(r2.get("minutes_played")))

        stats = [{"k": "OVR / POT", "a": f"{ovr1} / {pot1}", "b": f"{ovr2} / {pot2}"}]
        if g1 or g2:
            stats.append({"k": "Goals", "a": g1, "b": g2})
        if a1 or a2:
            stats.append({"k": "Assists", "a": a1, "b": a2})
        if m1 or m2:
            stats.append({"k": "Minutes", "a": f"{m1:,}", "b": f"{m2:,}"})

        def _sub(row):
            pos = str(row.get("sub_position", "") or "").strip()
            club = str(row.get("club", "") or "").strip()
            return " · ".join([p for p in (pos, club) if p and p != "?"])

        viz = {
            "type": "compare",
            "a": n1, "b": n2,
            "a_sub": _sub(r1), "b_sub": _sub(r2),
            "arch_a": str(r1.get("archetype", "") or "").strip(),
            "arch_b": str(r2.get("archetype", "") or "").strip(),
            "overlap": overlap,
            "stats": stats,
            "attrs": attr_viz,
            "both": readable_shared,
            "edges": edges,
        }

        # Short, clean text reply — the card carries the detailed numbers, so we don't
        # re-list them here (and there are no ASCII bars that misalign in the bubble).
        text_lines = [f"**{n1} vs {n2}**", ""]
        summary = f"{overlap}% playing-style overlap."
        if readable_shared:
            summary += f" Shared: {', '.join(readable_shared)}."
        text_lines.append(summary)
        if edges:
            text_lines.append("Key edges: " + " · ".join(edges) + ".")
        text_lines += ["", _METHOD]

        return embed_viz("\n".join(text_lines), viz)

    return compare_players_jaccard
