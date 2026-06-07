"""
TOOL 5 - Compare two players with a football-readable side-by-side breakdown.
Primary: FC26 attributes + real stats. Secondary: Jaccard playing-style overlap score.
The Jaccard computation is kept but the raw internal trait codes are not shown to users.
"""

import pandas as pd
from langchain.tools import tool
from ds_engine import jaccard


def _num(value, default=0):
    try:
        return default if value is None or pd.isna(value) else value
    except Exception:
        return default


def _bar(v, mx, width=10):
    filled = round(max(0, min(v, mx)) / mx * width) if mx else 0
    return "█" * filled + "░" * (width - filled)


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


def make_compare_players_jaccard_tool(engine):
    df = engine.df

    @tool
    def compare_players_jaccard(players: str) -> str:
        """
        Compare two players side by side: FC26 attributes, real stats (goals, assists,
        minutes), playing-style overlap, and key differences. Use for "compare X and Y",
        "X vs Y", "who is better — X or Y?".
        Input: two player names separated by 'vs', 'and', or a comma.
        Example: 'Mbappé vs Vinicius Jr'.
        """
        import re as _re
        parts = [p.strip() for p in _re.split(r"\s+vs\.?\s+|,\s*|\s+and\s+|\s+ו-?", players) if p.strip()]
        if len(parts) < 2:
            return (
                "Provide two players, e.g. 'Mbappé vs Vinicius Jr'."
                "\n\n🔍 Method: Jaccard similarity on categorical traits + FC26 attribute comparison."
            )

        i1, i2 = engine.find_index(parts[0]), engine.find_index(parts[1])
        if i1 is None:
            return f"Player not found: '{parts[0]}'.\n\n🔍 Method: Jaccard similarity on categorical traits + FC26 attribute comparison."
        if i2 is None:
            return f"Player not found: '{parts[1]}'.\n\n🔍 Method: Jaccard similarity on categorical traits + FC26 attribute comparison."

        r1, r2 = df.iloc[i1], df.iloc[i2]
        n1, n2 = r1["player_name"], r2["player_name"]

        # Jaccard for style overlap — computed but shown as a single clean score
        s1, s2 = engine.trait_set(i1), engine.trait_set(i2)
        j = jaccard(s1, s2)

        # Shared readable traits (exclude internal codes)
        readable_shared = []
        for trait in sorted(s1 & s2):
            prefix = trait.split(":")[0] if ":" in trait else ""
            label  = _READABLE_TRAIT.get(prefix, "skip")
            if label is None:
                continue
            value  = trait.split(":", 1)[1] if ":" in trait else trait
            readable_shared.append(f"{label}: {value}" if label != "skip" else None)
        readable_shared = [x for x in readable_shared if x]

        # FC26 attributes
        ATTRS = [
            ("Pace",      "fc_pace"),
            ("Shooting",  "fc_shooting"),
            ("Passing",   "fc_passing"),
            ("Dribbling", "fc_dribbling"),
            ("Defending", "fc_defending"),
            ("Physical",  "fc_physic"),
        ]
        attr_lines = []
        for label, col in ATTRS:
            v1, v2 = int(_num(r1.get(col))), int(_num(r2.get(col)))
            if not v1 and not v2:
                continue
            winner = f" ← {n1}" if v1 > v2 + 3 else (f" ← {n2}" if v2 > v1 + 3 else "")
            attr_lines.append(f"  {label:<12} {v1:>3}  {_bar(v1,99)}  {_bar(v2,99)}  {v2:<3}{winner}")

        ovr1, ovr2 = int(_num(r1.get("fc_overall"))), int(_num(r2.get("fc_overall")))
        pot1, pot2 = int(_num(r1.get("fc_potential"))), int(_num(r2.get("fc_potential")))

        lines = [
            f"**{n1}  vs  {n2}**\n",
            f"{'':14} {n1[:14]:<14}  {n2[:14]:<14}",
            f"{'Position':<14} {r1.get('sub_position','?')[:14]:<14}  {r2.get('sub_position','?')[:14]:<14}",
            f"{'Club':<14} {str(r1.get('club','?'))[:14]:<14}  {str(r2.get('club','?'))[:14]:<14}",
            f"{'Archetype':<14} {str(r1.get('archetype','?'))[:14]:<14}  {str(r2.get('archetype','?'))[:14]:<14}",
            f"{'OVR / POT':<14} {ovr1} / {pot1:<9}  {ovr2} / {pot2}",
            f"{'Goals':<14} {int(_num(r1.get('goals'))):<14}  {int(_num(r2.get('goals')))}",
            f"{'Assists':<14} {int(_num(r1.get('assists'))):<14}  {int(_num(r2.get('assists')))}",
            f"{'Minutes':<14} {int(_num(r1.get('minutes_played'))):,<14}  {int(_num(r2.get('minutes_played'))):,}",
            "\n**FC26 Attributes:**",
            f"  {'Attribute':<12} {'':>3}  {n1[:10]:<12}  {n2[:10]:<12}",
        ] + attr_lines + [
            f"\n**Playing style overlap: {round(j*100,1)}%**",
        ]
        if readable_shared:
            lines.append(f"Both: {', '.join(readable_shared[:4])}")

        # Who wins each area
        area_wins = []
        for label, col in ATTRS:
            v1, v2 = int(_num(r1.get(col))), int(_num(r2.get(col)))
            if v1 > v2 + 5:
                area_wins.append(f"{n1} leads in {label.lower()}")
            elif v2 > v1 + 5:
                area_wins.append(f"{n2} leads in {label.lower()}")
        if area_wins:
            lines.append("\nKey edges: " + " · ".join(area_wins[:4]))

        lines.append("\n🔍 Method: Jaccard similarity on categorical traits + FC26 attribute comparison.")
        return "\n".join(lines)

    return compare_players_jaccard
