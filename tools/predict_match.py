"""
tools/predict_match.py
חיזוי תוצאות משחקים.

לנבחרות לאומיות (מונדיאל): משתמש בטבלת חוזק הנבחרות שנגזרה מנתוני השחקנים
(צבירת מיטב הסגל לפי דירוג/שווי) — שיטת Data Science אמיתית.
הכלי מתמקד בנבחרות לאומיות שקיימות בנתוני השחקנים המקומיים.
"""

import numpy as np
import pandas as pd
from langchain.tools import tool

from ds_engine import normalize_nation


def make_predict_match_tool(df: pd.DataFrame, national_strength: pd.DataFrame):
    """Factory – national_strength הוא טבלת החוזק מ-ds_engine.build_national_strength."""

    known_nations = set(national_strength.index)

    def _predict_from_strength(n1: str, s1: float, n2: str, s2: float,
                               team1: str, team2: str) -> str:
        # הסתברויות עם softmax על ההפרש + סיכוי תיקו לפי קרבה
        scale = 6.0
        e1, e2 = np.exp(scale * s1), np.exp(scale * s2)
        p1_raw, p2_raw = e1 / (e1 + e2), e2 / (e1 + e2)

        closeness  = 1.0 - abs(s1 - s2) / max(s1 + s2, 1e-6)
        p_draw     = round(0.18 + 0.14 * closeness, 3)
        p1 = round(p1_raw * (1 - p_draw), 3)
        p2 = round(p2_raw * (1 - p_draw), 3)

        probs = {team1: p1, "תיקו": p_draw, team2: p2}
        winner = max(probs, key=probs.get)

        row1 = national_strength.loc[n1]
        row2 = national_strength.loc[n2]
        return (
            f"**חיזוי מונדיאל 2026: {team1} vs {team2}**\n\n"
            f"🏆 תוצאה סבירה: **{winner}** "
            f"(הסתברויות: {team1} {round(p1*100)}% | תיקו {round(p_draw*100)}% | {team2} {round(p2*100)}%)\n\n"
            f"📊 **חוזק נבחרות (נגזר מנתוני השחקנים):**\n"
            f"• {team1}: ניקוד חוזק {round(s1*100,1)} | שווי סגל ממוצע €{int(row1['squad_value_mean']):,} "
            f"| עומק סגל {int(row1['depth'])} שחקנים\n"
            f"• {team2}: ניקוד חוזק {round(s2*100,1)} | שווי סגל ממוצע €{int(row2['squad_value_mean']):,} "
            f"| עומק סגל {int(row2['depth'])} שחקנים\n\n"
            f"💡 **ניתוח:** "
            + (f"{team1} עם סגל חזק יותר ולכן הפייבוריטית." if s1 > s2 + 0.05
               else f"{team2} עם סגל חזק יותר ולכן הפייבוריטית." if s2 > s1 + 0.05
               else "הנבחרות שקולות — צפוי משחק צמוד.")
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

        # לפחות אחת לא זוהתה כנבחרת — הסבר ידידותי
        missing = [t for t, n in [(team1, n1), (team2, n2)] if n is None]
        return (
            f"לא הצלחתי לבנות חיזוי מבוסס-נתונים עבור: {', '.join(missing)}. "
            "החיזוי מבוסס על נבחרות לאומיות שיש להן שחקנים בנתונים. "
            "נסה שמות נבחרות מלאים (למשל Brazil, France, Argentina)."
            "\n\n🔍 Method: Squad-strength lookup from aggregated player market values."
        )

    return predict_match
