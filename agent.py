"""
agent.py
ScoutAI agent - בנוי עם tool calling ישיר (LangChain 1.x / langchain-google-genai 4.x).
החיפוש מבוסס שיטות Data Science: K-Means clustering, TF-IDF, Jaccard.
"""

import os
import time
import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from ds_engine import PlayerFeatures, build_national_strength
from tools.find_similar_players import make_find_similar_players_tool
from tools.scout_players import make_scout_players_tool
from tools.detect_anomalies import make_detect_anomalies_tool
from tools.predict_match import make_predict_match_tool
from tools.get_live_standings import make_get_live_standings_tool
from tools.world_cup import make_world_cup_tool

DATA_CSV     = "data/players_clean.csv"
WC_CSV       = "data/fwc26_match_schedule_agent.csv"

SYSTEM_PROMPT = """You are ScoutAI, an expert football analyst and scout assistant built for a
Data Science course. You work on a database of ~48,000 players and the FIFA World Cup 2026 schedule.

CRITICAL RULES — follow strictly:
1. NEVER answer about players, similar players, scouting lists, standings, predictions, or
   World Cup fixtures from your own memory. Your training knowledge is outdated. You MUST call
   the appropriate tool and base your answer ONLY on its returned data.
2. Tool selection:
   - "similar to X" / "like X"                 -> find_similar_players
   - "find / best / top players ..." criteria  -> scout_players
   - "anomalies / over- or under-performers"    -> detect_anomalies
   - "predict / who wins X vs Y"                -> predict_match
   - "standings / league table"                 -> get_live_standings
   - "World Cup schedule / group / fixtures / when does X play" -> world_cup_info
3. The similarity/scouting tools use clustering, TF-IDF and Jaccard — explain results by
   the data (shared profile, cluster, stats), never by player name.
4. Present the tool's data clearly. Do not invent players, numbers, or rankings.
5. Always reply in the SAME language the user used (Hebrew or English). Be concise and specific."""


def load_resources():
    """טוען נתונים ובונה את מבני ה-DS פעם אחת ב-startup."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} לא נמצא. הרץ python data_prep.py תחילה.")

    print("[agent] טוען נתוני שחקנים...", flush=True)
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"[agent] {len(df)} שחקנים נטענו.", flush=True)

    # מבני Data Science
    features = PlayerFeatures(df)
    national_strength = build_national_strength(df)

    # לוח מונדיאל 2026
    if os.path.exists(WC_CSV):
        schedule = pd.read_csv(WC_CSV)
        print(f"[agent] לוח מונדיאל: {len(schedule)} משחקים נטענו.", flush=True)
    else:
        schedule = pd.DataFrame()
        print(f"[agent] אזהרה: {WC_CSV} לא נמצא.", flush=True)

    return df, features, national_strength, schedule


# רשימת מודלים לסבב. לכל מודל מכסת free-tier יומית נפרדת (20/יום בפרויקט חדש),
# כך שסבב על כמה מודלים מגדיל את הקיבולת היומית הכוללת.
MODEL_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


class ScoutAgent:
    """Agent עם לולאת tool-calling ידנית + סבב מודלים על מיצוי מכסה (429)."""

    MAX_ITERATIONS = 5

    def __init__(self, llms_with_tools: list, tools: list):
        # רשימת מודלים (כל אחד כבר עם bind_tools) — מסודרים לפי עדיפות
        self.llms = llms_with_tools
        self.model_idx = 0
        self.tool_map = {t.name: t for t in tools}
        self.system_msg = SystemMessage(content=SYSTEM_PROMPT)

    @staticmethod
    def _extract_text(content) -> str:
        """gemini לפעמים מחזיר רשימת בלוקים במקום מחרוזת — מחלץ את הטקסט."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and b.get("text"):
                    parts.append(b["text"])
                elif isinstance(b, str):
                    parts.append(b)
            return "\n".join(parts).strip()
        return str(content)

    def _call_llm(self, messages):
        """
        קריאה ל-LLM. על 429 (מיצוי מכסה) עוברים למודל הבא בשרשרת.
        כך מנצלים את המכסה היומית הנפרדת של כל מודל.
        """
        last_err = None
        # מתחילים מהמודל הנוכחי ועוברים על כל השאר
        for offset in range(len(self.llms)):
            idx = (self.model_idx + offset) % len(self.llms)
            try:
                resp = self.llms[idx].invoke(messages)
                self.model_idx = idx  # נשארים על המודל שעבד
                return resp
            except Exception as e:
                last_err = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[agent] מודל {MODEL_CHAIN[idx]} מיצה מכסה, עובר לבא...", flush=True)
                    continue
                raise
        raise last_err

    def invoke(self, user_input: str) -> str:
        messages = [self.system_msg, HumanMessage(content=user_input)]
        for _ in range(self.MAX_ITERATIONS):
            try:
                response = self._call_llm(messages)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    return ("מיצינו את מכסת ה-API החינמית של Gemini להיום (20 בקשות ליום לכל מודל "
                            "בפרויקט חדש). המכסה מתאפסת מדי יום. נסה שוב מאוחר יותר.")
                raise
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                return self._extract_text(response.content) or "אין תגובה."

            for tc in response.tool_calls:
                name, args = tc["name"], tc["args"]
                tool_id    = tc.get("id", name)
                fn = self.tool_map.get(name)
                if fn is None:
                    result = f"כלי '{name}' לא נמצא."
                else:
                    try:
                        if len(args) == 1:
                            result = fn.invoke(next(iter(args.values())))
                        else:
                            result = fn.invoke(args)
                    except Exception as e:
                        result = f"שגיאה בהרצת {name}: {e}"
                print(f"[agent] tool={name} | args={args}", flush=True)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return "הגעתי למגבלת האיטרציות. נסה לשאול שאלה ממוקדת יותר."


def build_agent(df, features, national_strength, schedule):
    """בונה ומחזיר את ה-ScoutAgent עם כל הכלים."""
    tools = [
        make_find_similar_players_tool(df, features),
        make_scout_players_tool(df, features),
        make_detect_anomalies_tool(df),
        make_predict_match_tool(df, national_strength),
        make_get_live_standings_tool(),
        make_world_cup_tool(schedule),
    ]

    # בונים מופע לכל מודל בשרשרת (כל אחד עם bind_tools) לצורך סבב על מיצוי מכסה
    api_key = os.getenv("GEMINI_API_KEY")
    llms_with_tools = []
    for model_name in MODEL_CHAIN:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
        llms_with_tools.append(llm.bind_tools(tools))

    agent = ScoutAgent(llms_with_tools=llms_with_tools, tools=tools)
    print(f"[agent] Agent מוכן ({len(MODEL_CHAIN)} מודלים בסבב).", flush=True)
    return agent
