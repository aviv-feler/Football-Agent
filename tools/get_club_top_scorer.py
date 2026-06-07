"""
Top goal scorers for a specific club using the player dataset.

Active-player filter: require fc_overall IS NOT NULL (historical/retired FC "icons" have
no current FC26 rating) AND age <= 38. This removes e.g. Drogba (48, NaN) and Ramires
(39, NaN) while keeping current squad members.

Uses the `goals` column directly (real season stats) rather than the per-90 estimate.
"""

import pandas as pd
from langchain.tools import tool


def make_get_club_top_scorer_tool(engine):
    df = engine.df

    @tool
    def get_club_top_scorer(club: str) -> str:
        """
        Find the top goal scorers for a specific club this season.
        Use for: "who scored the most goals for Chelsea?", "Chelsea top scorer last season",
        "who scored most for Barcelona?". Uses real goals data from the player database.
        Input is the club name (English), e.g. "Chelsea", "Arsenal", "Real Madrid".
        """
        club_lower = club.strip().lower()
        mask = df["club"].fillna("").str.lower().str.contains(club_lower, na=False, regex=False)
        club_df = df[mask].copy()

        if club_df.empty:
            for w in [w for w in club_lower.split() if len(w) >= 4]:
                m2 = df["club"].fillna("").str.lower().str.contains(w, na=False, regex=False)
                if m2.any():
                    club_df = df[m2].copy()
                    break

        if club_df.empty:
            return (
                f"No players found for club '{club}'. "
                "Try the full name, e.g. 'Chelsea FC', 'Arsenal', 'Real Madrid'."
                "\n\n🔍 Method: Local player dataset (current-season goals)."
            )

        # Keep only active players: fc_overall not null (excludes historical FC icons/legends
        # whose fc_overall is NaN) and age <= 38 (additional safety for edge cases).
        age_col = pd.to_numeric(club_df["age"], errors="coerce")
        active = club_df[
            club_df["fc_overall"].notna() &
            (age_col <= 38)
        ].copy()

        # Fall back to the full club roster if the active filter leaves nothing
        if active.empty:
            active = club_df.copy()

        active["goals_num"] = pd.to_numeric(active["goals"], errors="coerce").fillna(0)
        active["apps_num"]  = pd.to_numeric(active["appearances"], errors="coerce").fillna(0)
        active["mins_num"]  = pd.to_numeric(active["minutes_played"], errors="coerce").fillna(0)

        # Require at least 1 appearance so bench-warmers with 0 stats don't pad the list
        scored = active[active["apps_num"] >= 1].copy()
        if scored.empty:
            scored = active.copy()

        top = scored.nlargest(5, "goals_num")[
            ["player_name", "sub_position", "goals_num", "apps_num", "mins_num"]
        ]

        actual_club = active["club"].iloc[0] if not active.empty else club
        lines = [f"**Top scorers for {actual_club} (current season):**\n"]
        for i, (_, row) in enumerate(top.iterrows(), 1):
            goals = int(row["goals_num"])
            apps  = int(row["apps_num"])
            mins  = int(row["mins_num"])
            pos   = row.get("sub_position") or "?"
            lines.append(
                f"{i}. {row['player_name']} ({pos}) — {goals} goals in {apps} apps ({mins} min)"
            )
        lines.append(
            "\n🔍 Method: Local player dataset (current-season goals). "
            "Figures may differ slightly from official records."
        )
        return "\n".join(lines)

    return get_club_top_scorer
