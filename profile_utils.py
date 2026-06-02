"""
profile_utils.py
בניית טקסט פרופיל לשחקן עבור sentence embeddings.

חשוב: הטקסט מתאר את *סגנון המשחק והרמה* של השחקן ולא את שמו.
כך ה-embedding תופס דמיון בסגנון/רמה ולא דמיון בשמות
(הבעיה הקודמת: "Mbappé" החזיר שחקנים אחרים בשם "Kylian").
"""

import pandas as pd


def _age_band(age: int) -> str:
    if age <= 0:        return "unknown age"
    if age < 21:        return "young prospect"
    if age < 26:        return "developing player in his early prime"
    if age < 30:        return "player in his peak prime years"
    if age < 33:        return "experienced player"
    return "veteran player"


def _rating_tier(rating: float) -> str:
    if rating >= 85:    return "world-class elite"
    if rating >= 80:    return "top-tier excellent"
    if rating >= 75:    return "very strong"
    if rating >= 70:    return "good professional"
    if rating >= 65:    return "solid"
    if rating > 0:      return "developing"
    return "unrated"


def _goal_tier(goals: int, position: str) -> str:
    if goals >= 200:    return "prolific goalscorer"
    if goals >= 100:    return "high-scoring"
    if goals >= 50:     return "regular scorer"
    if goals >= 20:     return "occasional scorer"
    if goals >= 5:      return "infrequent scorer"
    return "rarely scores"


def _assist_tier(assists: int) -> str:
    if assists >= 100:  return "elite playmaker"
    if assists >= 50:   return "strong creator"
    if assists >= 20:   return "creative contributor"
    if assists >= 5:    return "occasional provider"
    return "low creative output"


def _value_tier(mv: int) -> str:
    if mv >= 50_000_000:    return "superstar market value"
    if mv >= 20_000_000:    return "very high market value"
    if mv >= 5_000_000:     return "high market value"
    if mv >= 1_000_000:     return "moderate market value"
    if mv > 0:              return "modest market value"
    return "low market value"


def build_profile_text(row: pd.Series) -> str:
    """בונה תיאור סגנון/רמה (ללא שם) ל-embedding."""
    position   = str(row.get("position", "Unknown"))
    sub_pos    = str(row.get("sub_position", "")) if "sub_position" in row.index else ""
    nationality= str(row.get("nationality", "Unknown"))
    age        = int(row.get("age", 0) or 0)
    goals      = int(row.get("goals", 0) or 0)
    assists    = int(row.get("assists", 0) or 0)
    rating     = float(row.get("overall_rating", 0) or 0)
    mv         = int(row.get("market_value_in_eur", 0) or 0)

    role = sub_pos if sub_pos and sub_pos.lower() != "nan" else position

    return (
        f"A {_age_band(age)} playing as {role} ({position}) from {nationality}. "
        f"Skill level: {_rating_tier(rating)}. "
        f"Attacking output: {_goal_tier(goals, position)}, {_assist_tier(assists)}. "
        f"Market profile: {_value_tier(mv)}. "
        f"Career totals: {goals} goals, {assists} assists across {int(row.get('minutes_played',0) or 0)} minutes."
    )
