"""
World Cup 2026 prediction tools backed by WCPredictor.
LLM extracts intent + entities → engine simulates → LLM narrates.
"""

from langchain.tools import tool
from wc_predictor import WCPredictor, format_group_prediction, format_wc_winner, format_wc_match, format_wc_top_scorer


def make_wc_prediction_tools(wc: WCPredictor, players_df=None, squads_df=None,
                             match_predictor=None) -> list:

    @tool
    def predict_wc_group(group: str) -> str:
        """
        Predict the GROUP STAGE standings for a FIFA World Cup 2026 group and which
        teams are most likely to qualify. Use for 'Who will qualify from Group C?',
        'Predict Group I', 'Which teams come out of the group with France?'
        group: a single letter A–L (e.g. 'C', 'I', 'L').
        """
        result = wc.predict_group(group.strip().upper())
        return format_group_prediction(result)

    @tool
    def predict_wc_winner() -> str:
        """
        Simulate the full 2026 World Cup tournament (10,000 Monte Carlo simulations)
        and return each team's probability of winning the tournament, reaching the
        semi-finals, and more. Use for 'Who will win the World Cup?',
        'What are the favourites for the World Cup?', 'Who has the best chance?'
        """
        result = wc.predict_wc_winner(n_sims=5_000)
        return format_wc_winner(result)

    @tool
    def predict_wc_match(team_a: str, team_b: str) -> str:
        """
        Predict the outcome and SCORELINE for a specific World Cup 2026 match
        (neutral ground). Use for 'What will be the score in Brazil vs France?',
        'Who will win Netherlands vs Argentina at the World Cup?'
        Pass corrected full national team names.
        """
        # Prefer the trained Logistic Regression squad-strength model.
        if match_predictor is not None and match_predictor.has_team(team_a) and match_predictor.has_team(team_b):
            from match_predictor import format_prediction
            pred = match_predictor.predict(team_a, team_b)
            if pred is not None:
                return format_prediction(pred)
        result = wc.predict_match(team_a, team_b)
        return format_wc_match(result)

    @tool
    def predict_wc_top_scorer(n: int = 10) -> str:
        """
        Predict the top Golden Boot / top scorer candidates for the 2026 World Cup.
        Combines each player's goals_per90 rate with expected games played (based on
        Monte Carlo tournament stage probabilities) and shooting quality.
        Use for: 'Who will be the top scorer at the World Cup?', 'Who wins the Golden Boot?',
        'Best goal scorers in WC 2026?', 'Who will score the most goals at the World Cup?'
        n: number of candidates to return (default 10).
        """
        if players_df is None:
            return "Player data not available for WC top scorer projection."
        result = wc.predict_wc_top_scorer(players_df, squads_df=squads_df, n=n)
        return format_wc_top_scorer(result)

    return [predict_wc_group, predict_wc_winner, predict_wc_match, predict_wc_top_scorer]
