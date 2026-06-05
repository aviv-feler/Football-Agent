"""
Prediction tools backed by PredictionEngine.
The LLM extracts intent + entities and calls the right tool;
the engine builds features and predicts; the LLM narrates.
"""

from langchain.tools import tool
from prediction_engine import PredictionEngine, generate_prediction_response, parse_prediction_query


def make_prediction_tools(engine: PredictionEngine) -> list:

    @tool
    def predict_club_match_score(home_team: str, away_team: str, user_context: str = "") -> str:
        """
        Predict the SCORELINE and result for a club vs club match. Uses a RandomForest
        result model + Gradient Boosting xG model + context-aware Poisson scoreline
        selection trained on 10 seasons of top-5 league data.
        Use for: 'What will be the score between Chelsea and Man Utd?', 'Predict Arsenal vs Liverpool'.
        home_team: the home team (or the first-mentioned if unknown — state the assumption).
        away_team: the away team.
        user_context: the original user sentence (helps the home/away resolver).
        """
        result = engine.predict_club_match(home_team, away_team, user_context)
        return generate_prediction_response(result)

    @tool
    def predict_national_match_score(team_a: str, team_b: str) -> str:
        """
        Predict the result/scoreline for a NATIONAL TEAM match (World Cup / international).
        Uses the hybrid squad-strength + World Cup Elo pedigree model.
        Treats the match as neutral ground by default.
        Use for: 'Brazil vs France World Cup', 'Who has a better chance — Netherlands or Spain?'
        """
        from tools.predict_match import make_predict_match_tool
        return f"[route to existing predict_match: {team_a} vs {team_b}]"

    @tool
    def predict_top_scorer(league: str = "", n: int = 5) -> str:
        """
        Predict the top scorer(s) for next season in a given league or across Europe.
        Uses a RandomForest regressor trained on player attributes (shooting, pace, goals_per90,
        overall, potential, team attack strength) to project goals next season.
        Use for: 'Who will be the PL top scorer?', 'Best goal scorers in Europe next season'.
        league: 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1', or '' for all.
        n: number of candidates to return (default 5).
        """
        result = engine.predict_top_scorer(league=league, n=n)
        return generate_prediction_response(result)

    @tool
    def predict_player_goals(player_name: str) -> str:
        """
        Project how many goals a specific player will score next season.
        Uses the same RandomForest model as the top-scorer tool.
        Use for: 'How many goals will Haaland score?', 'Will Salah score more than 20?'
        player_name: corrected full player name.
        """
        result = engine.predict_player_goals(player_name)
        return generate_prediction_response(result)

    return [predict_club_match_score, predict_top_scorer, predict_player_goals]
