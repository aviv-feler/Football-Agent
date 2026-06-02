"""
test_tools.py
בדיקות ידניות לכל כלי ב-ScoutAI (מבוסס שיטות Data Science).
הרץ: python test_tools.py    (אינו צורך מכסת LLM — בודק את הכלים ישירות)
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_CSV = "data/players_clean.csv"
WC_CSV   = "data/fwc26_match_schedule_agent.csv"


def separator(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    if not os.path.exists(DATA_CSV):
        print("ERROR: הרץ python data_prep.py תחילה.")
        sys.exit(1)

    print("ScoutAI – בדיקת כלים (DS: K-Means + TF-IDF + Jaccard)")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"נטענו {len(df)} שחקנים.")

    # בניית מבני ה-DS
    from ds_engine import PlayerFeatures, build_national_strength
    features = PlayerFeatures(df)
    national_strength = build_national_strength(df)
    schedule = pd.read_csv(WC_CSV) if os.path.exists(WC_CSV) else pd.DataFrame()

    from tools.find_similar_players import make_find_similar_players_tool
    from tools.scout_players import make_scout_players_tool
    from tools.detect_anomalies import make_detect_anomalies_tool
    from tools.predict_match import make_predict_match_tool
    from tools.get_live_standings import make_get_live_standings_tool
    from tools.world_cup import make_world_cup_tool

    fs = make_find_similar_players_tool(df, features)
    sc = make_scout_players_tool(df, features)
    an = make_detect_anomalies_tool(df)
    pm = make_predict_match_tool(df, national_strength)
    gs = make_get_live_standings_tool()
    wc = make_world_cup_tool(schedule)

    # ── Tool 1: find_similar_players (cluster + Jaccard + TF-IDF) ──
    separator("TOOL 1: find_similar_players")
    print(fs.invoke("Mbappé"))
    print(fs.invoke("XYZ_Unknown"))  # מקרה קצה: שחקן לא קיים

    # ── Tool 2: scout_players ──
    separator("TOOL 2: scout_players")
    for q in ["best young strikers", "top defenders under 23 from South America",
              "experienced playmakers from Europe"]:
        print(f"\n>>> {q}")
        print(sc.invoke(q))

    # ── Tool 3: detect_anomalies ──
    separator("TOOL 3: detect_anomalies")
    print(an.invoke(""))

    # ── Tool 4: predict_match (national strength) ──
    separator("TOOL 4: predict_match")
    for t1, t2 in [("Brazil", "France"), ("Argentina", "Spain")]:
        print(f"\n>>> {t1} vs {t2}")
        print(pm.invoke({"team1": t1, "team2": t2}))

    # ── Tool 5: get_live_standings ──
    separator("TOOL 5: get_live_standings")
    print(gs.invoke("Premier League"))

    # ── Tool 6: world_cup_info ──
    separator("TOOL 6: world_cup_info")
    for q in ["Brazil matches", "Group D", "World Cup final"]:
        print(f"\n>>> {q}")
        print(wc.invoke(q))

    print("\n" + "="*60 + "\n  כל הבדיקות הושלמו!\n" + "="*60)


if __name__ == "__main__":
    main()
