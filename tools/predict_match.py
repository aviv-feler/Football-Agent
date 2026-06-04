"""
Predict football match outcomes.

For national teams (World Cup), the model uses squad-strength features derived from
player data (aggregating the best of each squad by rating/value) — a genuine
Data Science method. The tool focuses on national teams present in the player dataset.
"""

import numpy as np
import pandas as pd
from langchain.tools import tool

from ds_engine import normalize_nation


def make_predict_match_tool(df: pd.DataFrame, national_strength: pd.DataFrame):
    """Factory for the match-prediction tool."""

    known_nations = set(national_strength.index)

    def _predict_from_strength(n1: str, s1: float, n2: str, s2: float,
                               team1: str, team2: str) -> str:
        # Softmax-like probabilities plus a draw probability based on closeness.
        scale = 6.0
        e1, e2 = np.exp(scale * s1), np.exp(scale * s2)
        p1_raw, p2_raw = e1 / (e1 + e2), e2 / (e1 + e2)

        closeness  = 1.0 - abs(s1 - s2) / max(s1 + s2, 1e-6)
        p_draw     = round(0.18 + 0.14 * closeness, 3)
        p1 = round(p1_raw * (1 - p_draw), 3)
        p2 = round(p2_raw * (1 - p_draw), 3)

        probs = {team1: p1, "draw": p_draw, team2: p2}
        winner = max(probs, key=probs.get)

        row1 = national_strength.loc[n1]
        row2 = national_strength.loc[n2]
        return (
            f"**World Cup 2026 prediction: {team1} vs {team2}**\n\n"
            f"Likely result: **{winner}** "
            f"(probabilities: {team1} {round(p1*100)}% | draw {round(p_draw*100)}% | {team2} {round(p2*100)}%)\n\n"
            f"**National-team strength derived from player data:**\n"
            f"- {team1}: strength score {round(s1*100,1)} | average squad value EUR {int(row1['squad_value_mean']):,} "
            f"| player depth {int(row1['depth'])}\n"
            f"- {team2}: strength score {round(s2*100,1)} | average squad value EUR {int(row2['squad_value_mean']):,} "
            f"| player depth {int(row2['depth'])}\n\n"
            f"**Reasoning:** "
            + (f"{team1} has the stronger squad profile and is the favorite." if s1 > s2 + 0.05
               else f"{team2} has the stronger squad profile and is the favorite." if s2 > s1 + 0.05
               else "The teams are close, so this projects as a tight match.")
            + "\n\n🔍 Method: Logistic (softmax) model on squad-strength features "
              "derived from aggregated player market values."
        )

    @tool
    def predict_match(team1: str, team2: str) -> str:
        """
        Predict the outcome of a football match between two teams. For national teams
        (World Cup) it uses squad strength derived from the player dataset. Returns a
        winner with probability percentages and reasoning.
        Inputs: team1 and team2 (team names).
        """
        n1 = normalize_nation(team1, known_nations)
        n2 = normalize_nation(team2, known_nations)

        if n1 and n2:
            s1 = float(national_strength.loc[n1, "strength"])
            s2 = float(national_strength.loc[n2, "strength"])
            return _predict_from_strength(n1, s1, n2, s2, team1, team2)

        missing = [t for t, n in [(team1, n1), (team2, n2)] if n is None]
        return (
            f"I could not build a data-based prediction for: {', '.join(missing)}. "
            "This predictor currently supports national teams that appear in the player dataset. "
            "Try full national-team names such as Brazil, France, or Argentina."
            "\n\n🔍 Method: Squad-strength lookup from aggregated player market values."
        )

    return predict_match
