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


def make_predict_match_tool(df: pd.DataFrame, national_strength: pd.DataFrame, club_model=None,
                            match_predictor=None):
    """Factory for the match-prediction tool.

    Routes a matchup to the club Poisson model when both teams are known clubs, and to
    the trained Logistic Regression squad-strength model for national teams (falling
    back to the hybrid strength softmax if the trained model is unavailable).
    """

    known_nations = set(national_strength.index)

    def _predict_club(team1: str, team2: str) -> str:
        res = club_model.predict(team1, team2)
        order = sorted(
            [(res["home"], res["p_home"]), ("draw", res["p_draw"]), (res["away"], res["p_away"])],
            key=lambda kv: kv[1], reverse=True,
        )
        winner = order[0][0]
        return (
            f"**Match prediction: {res['home']} (home) vs {res['away']}**\n\n"
            f"Likely result: **{winner}** "
            f"(probabilities: {res['home']} {round(res['p_home']*100)}% | "
            f"draw {round(res['p_draw']*100)}% | {res['away']} {round(res['p_away']*100)}%)\n\n"
            f"Expected goals: {res['home']} {res['exp_home']:.2f} - "
            f"{res['exp_away']:.2f} {res['away']} "
            f"(most likely score {res['likely_score'][0]}-{res['likely_score'][1]}).\n\n"
            "🔍 Method: Recency-weighted Poisson goals model (team attack/defence factors "
            "+ home advantage) trained on 10 seasons of top-5 league results."
        )

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

        def pedigree(row) -> str:
            if bool(row.get("has_history", False)):
                return (f"historical tournament rating from {int(row['n_wc_matches'])} World Cup matches")
            return "no World Cup history (squad strength only)"

        return (
            f"**World Cup 2026 prediction: {team1} vs {team2}**\n\n"
            f"Likely result: **{winner}** "
            f"(probabilities: {team1} {round(p1*100)}% | draw {round(p_draw*100)}% | {team2} {round(p2*100)}%)\n\n"
            f"**Hybrid national-team strength (current squad value + World Cup pedigree):**\n"
            f"- {team1}: strength score {round(s1*100,1)} | average squad value EUR {int(row1['squad_value_mean']):,} "
            f"| {pedigree(row1)}\n"
            f"- {team2}: strength score {round(s2*100,1)} | average squad value EUR {int(row2['squad_value_mean']):,} "
            f"| {pedigree(row2)}\n\n"
            f"**Reasoning:** "
            + (f"{team1} has the stronger combined profile and is the favorite." if s1 > s2 + 0.05
               else f"{team2} has the stronger combined profile and is the favorite." if s2 > s1 + 0.05
               else "The teams are close, so this projects as a tight match.")
            + "\n\n🔍 Method: Softmax (logistic) probabilities on hybrid national-team strength = current "
              "squad market value blended with Elo from historical World Cup results."
        )

    @tool
    def predict_match(team1: str, team2: str) -> str:
        """
        Predict the outcome of a football match between two teams. Club matchups
        (top-5 leagues) use a Poisson goals model; national-team matchups (World Cup)
        use a hybrid squad-strength + World Cup pedigree model. Returns a winner with
        probability percentages and reasoning.
        Inputs: team1 and team2 (team names).
        """
        # Club matchup: both teams known to the club model.
        if club_model is not None and getattr(club_model, "ok", False):
            if club_model.resolve(team1) and club_model.resolve(team2):
                return _predict_club(team1, team2)

        # National teams: prefer the trained Logistic Regression squad-strength model.
        if match_predictor is not None and match_predictor.has_team(team1) and match_predictor.has_team(team2):
            from match_predictor import format_prediction
            pred = match_predictor.predict(team1, team2)
            if pred is not None:
                return format_prediction(pred)

        n1 = normalize_nation(team1, known_nations)
        n2 = normalize_nation(team2, known_nations)

        if n1 and n2:
            s1 = float(national_strength.loc[n1, "strength"])
            s2 = float(national_strength.loc[n2, "strength"])
            return _predict_from_strength(n1, s1, n2, s2, team1, team2)

        missing = [t for t, n in [(team1, n1), (team2, n2)] if n is None]
        return (
            f"I could not build a data-based prediction for: {', '.join(missing)}. "
            "This predictor covers top-5 league clubs (e.g. Man City, Real Madrid) and "
            "national teams in the player dataset (e.g. Brazil, France). Try full team names."
            "\n\n🔍 Method: Poisson club model / hybrid national squad-strength + WC pedigree."
        )

    return predict_match
