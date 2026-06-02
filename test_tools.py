"""
test_tools.py
בדיקת כל הכלים ישירות (ללא צריכת מכסת LLM) — מאמת את שיטות ה-Data Science.
הרץ: python test_tools.py
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

WC_CSV = "data/fwc26_match_schedule_agent.csv"


def sep(t): print(f"\n{'='*64}\n  {t}\n{'='*64}")


def main():
    if not os.path.exists("data/players_clean.csv") or not os.path.exists("data/player_features.npy"):
        print("ERROR: הרץ python data_prep.py תחילה.")
        sys.exit(1)

    from ds_engine import load_engine, build_national_strength
    engine = load_engine()
    national_strength = build_national_strength(engine.df)
    schedule = pd.read_csv(WC_CSV) if os.path.exists(WC_CSV) else pd.DataFrame()

    from tools.find_similar_players import make_find_similar_players_tool
    from tools.scout_players import make_scout_players_tool
    from tools.get_player_archetype import make_get_player_archetype_tool
    from tools.detect_anomalies import make_detect_anomalies_tool
    from tools.compare_players_jaccard import make_compare_players_jaccard_tool
    from tools.predict_match import make_predict_match_tool
    from tools.get_live_standings import make_get_live_standings_tool
    from tools.world_cup import make_world_cup_tool

    fs = make_find_similar_players_tool(engine)
    sc = make_scout_players_tool(engine)
    ar = make_get_player_archetype_tool(engine)
    an = make_detect_anomalies_tool(engine)
    jc = make_compare_players_jaccard_tool(engine)
    pm = make_predict_match_tool(engine.df, national_strength)
    gs = make_get_live_standings_tool()
    wc = make_world_cup_tool(schedule)

    sep("TOOL 1: find_similar_players — Cosine on numeric vectors")
    print(fs.invoke("Mbappé"))

    sep("TOOL 2: scout_players — Content-based (cosine)")
    print(sc.invoke("best young strikers"))
    print()
    print(sc.invoke("creative midfielders from Europe"))

    sep("TOOL 3: get_player_archetype — K-Means")
    print(ar.invoke("Rodri"))

    sep("TOOL 4: detect_anomalies — Z-score from cluster centroid")
    print(an.invoke("Attack"))

    sep("TOOL 5: compare_players_jaccard — Jaccard on trait sets")
    print(jc.invoke("Mbappé vs Haaland"))

    sep("TOOL 6: predict_match — squad strength")
    print(pm.invoke({"team1": "Brazil", "team2": "France"}))

    sep("TOOL 7: get_live_standings")
    print(gs.invoke("Premier League")[:400])

    sep("TOOL 8: world_cup_info")
    print(wc.invoke("Group D"))

    print("\n" + "="*64 + "\n  כל הבדיקות הושלמו!\n" + "="*64)


if __name__ == "__main__":
    main()
