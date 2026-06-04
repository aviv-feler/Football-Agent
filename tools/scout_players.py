"""
tools/scout_players.py
TOOL 2 — סקאוטינג לפי קריטריונים בשפה טבעית (Content-Based Recommendation).
הסוכן (Gemini) הוא שכבת ה-NLP שמעבירה את הקריטריון; כאן מפרקים לפילטרים,
בונים "וקטור-מטרה" אידיאלי, ומדרגים את המועמדים ב-Cosine similarity.

(חלופה אקדמית: Item-Item Collaborative Filtering — לא מומש כי אין נתוני
דירוג משתמשים; Content-Based מתאים יותר לפרופיל פיצ'רים.)
"""

import re
import numpy as np
from langchain.tools import tool

_POSITION_KEYWORDS = {
    "Attack":     ["striker", "forward", "winger", "attacker", "cf", "st", "חלוץ", "קיצוני", "התקפ"],
    "Midfield":   ["midfield", "midfielder", "playmaker", "cm", "cdm", "cam", "קשר", "אמצע"],
    "Defender":   ["defender", "defence", "defense", "centre-back", "fullback", "מגן", "בלם", "הגנה"],
    "Goalkeeper": ["goalkeeper", "keeper", "gk", "שוער"],
}
_CONTINENT_NATIONS = {
    "south america": ["Brazil", "Argentina", "Colombia", "Chile", "Uruguay",
                      "Peru", "Ecuador", "Venezuela", "Paraguay", "Bolivia"],
    "africa": ["Nigeria", "Senegal", "Ghana", "Cote d'Ivoire", "Cameroon",
               "Morocco", "Egypt", "Algeria", "Tunisia", "Mali"],
    "europe": ["France", "Spain", "Germany", "Italy", "England", "Portugal",
               "Netherlands", "Belgium", "Croatia", "Denmark", "Norway", "Sweden"],
    "asia": ["Japan", "Korea, South", "Saudi Arabia", "Iran", "Australia", "Qatar"],
    "north america": ["United States", "Mexico", "Canada", "Costa Rica", "Jamaica"],
}


def make_scout_players_tool(engine, wc_nations: set | None = None):
    df = engine.df
    fidx = {n: i for i, n in enumerate(engine.feature_names)}
    wc_nations = wc_nations or set()

    @tool
    def scout_players(criteria: str) -> str:
        """
        Find top players matching natural-language scouting criteria. Parses the criteria
        into filters (position, age, region, World Cup participation) then ranks the
        filtered set by COSINE SIMILARITY to an ideal target profile (content-based).
        Examples: 'fast striker under 23 from South America', 'best strikers in the World Cup'.
        Input is the criteria as plain text.
        """
        crit = criteria.lower()
        cand = df
        applied, emphasis = [], {}

        # ── פילטר מונדיאל: רק שחקנים מנבחרות שמשתתפות במונדיאל 2026 ──
        if wc_nations and any(w in crit for w in ["world cup", "mundial", "מונדיאל", "wc2026", "fwc"]):
            m = cand["nationality"].isin(wc_nations)
            if m.any():
                cand = cand[m]; applied.append("World Cup 2026 teams")

        # ── פילטר עמדה ──
        for pos_value, kws in _POSITION_KEYWORDS.items():
            if any(k in crit for k in kws):
                m = cand["position"] == pos_value
                if m.any():
                    cand = cand[m]; applied.append(f"position={pos_value}")
                break

        # ── פילטר גיל ──
        nums = re.findall(r"\b(1[5-9]|[2-3]\d|4[0-5])\b", crit)
        if re.search(r"under|below|younger|מתחת|פחות", crit) and nums:
            lim = int(nums[0]); cand = cand[cand["age"] < lim]; applied.append(f"age<{lim}")
            emphasis["age"] = -1.5
        elif re.search(r"over|above|older|מעל|יותר", crit) and nums:
            lim = int(nums[0]); cand = cand[cand["age"] > lim]; applied.append(f"age>{lim}")
            emphasis["age"] = 1.5
        elif re.search(r"\byoung\b|youngster|prospect|צעיר", crit):
            cand = cand[(cand["age"] > 0) & (cand["age"] <= 23)]; applied.append("age<=23")
            emphasis["age"] = -1.5
        elif re.search(r"veteran|experienced|ותיק|מנוסה", crit):
            cand = cand[cand["age"] >= 30]; applied.append("age>=30")
            emphasis["age"] = 1.5

        # ── פילטר אזור/לאום ──
        for cont, nations in _CONTINENT_NATIONS.items():
            if cont in crit:
                m = cand["nationality"].isin(nations)
                if m.any():
                    cand = cand[m]; applied.append(f"region={cont}")
                break
        else:
            for nat in df["nationality"].dropna().unique():
                if isinstance(nat, str) and len(nat) > 3 and nat.lower() in crit:
                    cand = cand[cand["nationality"] == nat]; applied.append(f"nationality={nat}")
                    break

        # ── דגשי פרופיל (לבניית וקטור-המטרה) ──
        if any(w in crit for w in ["goalscorer", "scorer", "goals", "מבקיע", "כובש", "striker", "חלוץ"]):
            emphasis["goals_per90"] = 1.8; emphasis["ga_per90"] = 1.2
        if any(w in crit for w in ["playmaker", "assist", "creative", "מבשל", "יצירתי"]):
            emphasis["assists_per90"] = 1.8; emphasis["ga_per90"] = 1.2
        if any(w in crit for w in ["best", "top", "elite", "טוב", "מצטיין", "הכי"]):
            emphasis["market_value_log"] = 1.8; emphasis["minutes_played_log"] = 1.0
            emphasis.setdefault("ga_per90", 1.0)

        if cand.empty:
            return (
                f"לא נמצאו שחקנים שתואמים את '{criteria}'. נסה לרכך את הקריטריונים."
                "\n\n🔍 Method: Content-based filtering — Cosine similarity to an ideal "
                "profile after hard filters."
            )

        # ── וקטור-מטרה אידיאלי (z-space): 0=ממוצע, דגשים חיוביים/שליליים ──
        target = np.zeros(len(engine.feature_names))
        if not emphasis:  # ברירת מחדל: שחקן איכותי ופעיל
            emphasis = {"market_value_log": 1.2, "minutes_played_log": 1.0, "ga_per90": 0.8}
        for feat, val in emphasis.items():
            if feat in fidx:
                target[fidx[feat]] = val

        cand_ilocs = np.array([df.index.get_loc(i) for i in cand.index])
        # דמיון סגנון (cosine, מתעלם מעוצמה) משולב עם איכות (שווי שוק) כדי
        # שלא נחזיר שחקנים אלמונים שרק "מצביעים לכיוון" הנכון.
        sims = engine.cosine_to_vector(target, cand_ilocs)
        sims_n = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
        mv = cand["market_value_in_eur"].fillna(0).clip(lower=0).values
        mv_q = np.log1p(mv) / (np.log1p(mv.max()) if mv.max() > 0 else 1)
        score = 0.5 * sims_n + 0.5 * mv_q
        order = np.argsort(score)[::-1][:5]
        top_ilocs = cand_ilocs[order]

        header = f"**5 שחקנים מובילים ל-'{criteria}'**"
        if applied:
            header += f"  _(פילטרים: {', '.join(applied)})_"
        lines = [header + ":\n"]
        for rank, il in enumerate(top_ilocs, 1):
            r = df.iloc[il]
            lines.append(
                f"{rank}. **{r['player_name']}** "
                f"({r.get('sub_position', r.get('position','?'))} | {r.get('club','?')} | "
                f"{r.get('nationality','?')} | גיל {int(r.get('age',0) or 0)}) — "
                f"ארכיטיפ: {r.get('archetype','?')} | גולים: {int(r.get('goals',0))} | "
                f"בישולים: {int(r.get('assists',0))} | שווי: €{int(r.get('market_value_in_eur',0)):,}"
            )
        emphasized = ", ".join(k for k in emphasis)
        lines.append(f"\n🔍 Method: Content-based filtering — Cosine similarity to an ideal "
                     f"profile (emphasis: {emphasized}) after hard filters.")
        return "\n".join(lines)

    return scout_players
