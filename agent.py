"""
agent.py
ScoutAI agent with direct tool-calling support (LangChain 1.x / langchain-google-genai 4.x).
Search is based on Data Science methods: K-Means clustering, Cosine similarity, Jaccard.
Internal prompts and tool context are kept in English; final answers are written in
the user's language.
"""

import os
import re
import time
import threading
import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from data_manager import PLAYER_PROFILES_FILE, load_player_profiles, write_source_map
from ds_engine import load_engine, build_national_strength, normalize_nation
from tools.find_similar_players import make_find_similar_players_tool
from tools.scout_players import make_scout_players_tool
from tools.get_player_archetype import make_get_player_archetype_tool
from tools.detect_anomalies import make_detect_anomalies_tool
from tools.compare_players_jaccard import make_compare_players_jaccard_tool
from tools.predict_match import make_predict_match_tool
from tools.get_live_standings import make_get_live_standings_tool
from tools.world_cup import make_world_cup_tool
from football_qa import FootballQAPipeline

DATA_CSV     = "data/players_clean.csv"
WC_CSV       = "data/fwc26_match_schedule_agent.csv"

SYSTEM_PROMPT = """You are ScoutAI, an expert football analyst and scout assistant built for a
Data Science course. You work on a database of ~48,000 players and the FIFA World Cup 2026 schedule.

CRITICAL RULES — follow strictly:
1. NEVER answer about players, similar players, scouting lists, archetypes, anomalies,
   predictions, standings, or World Cup fixtures from your own memory. Your training
   knowledge is outdated. You MUST call the right tool and base your answer ONLY on its data.
2. Tool selection:
   - "similar to X" / "like X"                      -> find_similar_players  (cosine on stats)
   - "find / best / top players ..." criteria       -> scout_players         (content-based)
   - "what type/role/archetype is X" / "profile of X"-> get_player_archetype  (K-Means)
   - "anomalies / over- or under-performers"         -> detect_anomalies      (Z-score)
   - "compare X and Y" / "how similar are X and Y"   -> compare_players_jaccard (Jaccard)
   - "predict / who wins X vs Y"                     -> predict_match
   - "standings / league table"                      -> get_live_standings
   - "World Cup schedule / group / when does X play" -> world_cup_info
3. Each tool ends its output with a "🔍 Method:" line naming the Data Science model used.
   ALWAYS preserve and include that Method line in your answer — it is required for grading.
4. Present the tool's data clearly. Do not invent players, numbers, or rankings.
5. Always reply in the SAME language the user used (Hebrew or English). Be concise and specific."""


def load_resources():
    """Load data and build DS resources once at startup."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} was not found. Run python data_prep.py first.")

    print("[agent] Loading DS engine...", flush=True)
    engine = load_engine()
    write_source_map()
    current_profiles = load_player_profiles()
    national_strength = build_national_strength(current_profiles)
    print(f"[agent] current player source: {PLAYER_PROFILES_FILE}", flush=True)

    if os.path.exists(WC_CSV):
        schedule = pd.read_csv(WC_CSV)
        print(f"[agent] World Cup schedule loaded: {len(schedule)} matches.", flush=True)
    else:
        schedule = pd.DataFrame()
        print(f"[agent] Warning: {WC_CSV} was not found.", flush=True)

    return engine, national_strength, schedule


# Model rotation. Each Gemini model may have a separate free-tier quota.
MODEL_CHAIN = [
    "gemini-2.5-flash",        # ראשי — חכם יותר לניתוב וניסוח
    "gemini-flash-latest",     # גיבוי באותה רמה
    "gemini-2.5-flash-lite",   # גיבוי קל יותר (מכסה נפרדת)
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
]


class ScoutAgent:
    """Agent with a manual tool-calling loop and model rotation on quota errors."""

    MAX_ITERATIONS = 5

    MAX_HISTORY_TURNS = 6

    def __init__(self, llms_with_tools: list, tools: list, qa_pipeline: FootballQAPipeline | None = None):
        self.llms = llms_with_tools
        self.model_idx = 0
        self.tool_map = {t.name: t for t in tools}
        self.system_msg = SystemMessage(content=SYSTEM_PROMPT)
        self.qa_pipeline = qa_pipeline
        # Separate conversation history per session to avoid mixing users.
        self.histories: dict[str, list] = {}
        self._lock = threading.RLock()

    def _get_history(self, session_id: str) -> list:
        with self._lock:
            return list(self.histories.get(session_id, []))

    def _remember(self, session_id: str, user_input: str, answer: str):
        """Store the current conversation turn and trim older history."""
        with self._lock:
            history = self.histories.setdefault(session_id, [])
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))
            max_msgs = self.MAX_HISTORY_TURNS * 2
            if len(history) > max_msgs:
                self.histories[session_id] = history[-max_msgs:]

    def reset(self, session_id: str | None = None):
        """Clear conversation history (new chat)."""
        with self._lock:
            if session_id:
                self.histories.pop(session_id, None)
            else:
                self.histories = {}
        if self.qa_pipeline is not None:
            self.qa_pipeline.reset_context()

    @staticmethod
    def _looks_like_scout_query(text: str) -> bool:
        q = text.lower()
        scout_words = [
            "best", "top", "find", "scout", "player", "players", "striker",
            "forward", "winger", "midfielder", "defender", "goalkeeper",
            "הכי", "טוב", "מוביל", "מצא", "חפש", "שחקן", "שחקנים",
            "חלוץ", "התקפה", "קשר", "בלם", "מגן", "שוער",
        ]
        excluded_words = [
            "similar", "like", "דומה", "דומים", "compare", "השווה",
            "predict", "נבא", "תחזה", "טבלה", "standings", "מונדיאל",
            "world cup", "archetype", "ארכיטיפ", "פרופיל", "חריג",
        ]
        return any(w in q for w in scout_words) and not any(w in q for w in excluded_words)

    @staticmethod
    def _extract_similar_player(text: str) -> str:
        q = text.strip()
        patterns = [
            r"similar to\s+(.+)$",
            r"like\s+(.+)$",
            r"דומים\s+ל(.+)$",
            r"דומה\s+ל(.+)$",
            r"כמו\s+(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" ?.,:;\"'")
        return q

    @staticmethod
    def _extract_after_keywords(text: str, patterns: list[str]) -> str:
        q = text.strip()
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" ?.,:;\"'")
        return q

    @staticmethod
    def _extract_matchup(text: str) -> dict | None:
        q = text.strip()
        cleaned = re.sub(
            r"^\s*(predict|who wins|forecast|נבא|תחזה|מי ינצח)\s*:?\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        parts = re.split(
            r"\s+vs\.?\s+|\s+against\s+|\s+versus\s+|נגד|מול",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(parts) != 2:
            return None
        team1, team2 = [p.strip(" ?.,:;\"'") for p in parts]
        if not team1 or not team2:
            return None
        return {"team1": team1, "team2": team2}

    def _direct_tool_answer(self, user_input: str) -> str | None:
        """Deterministic routing for common graded/demo data-science queries."""
        q = user_input.lower()

        if any(w in q for w in ["similar", "like", "דומה", "דומים", "כמו"]):
            tool = self.tool_map.get("find_similar_players")
            if tool:
                return str(tool.invoke(self._extract_similar_player(user_input)))

        if any(w in q for w in ["archetype", "profile", "type of player", "what type", "ארכיטיפ", "פרופיל"]):
            tool = self.tool_map.get("get_player_archetype")
            if tool:
                player = self._extract_after_keywords(user_input, [
                    r"archetype\s+(?:of|for)\s+(.+)$",
                    r"profile\s+(?:of|for)\s+(.+)$",
                    r"type\s+of\s+player\s+is\s+(.+)$",
                    r"what\s+type\s+of\s+player\s+is\s+(.+)$",
                    r"ארכיטיפ\s+של\s+(.+)$",
                    r"פרופיל\s+של\s+(.+)$",
                ])
                return str(tool.invoke(player))

        if any(w in q for w in ["anomal", "overperform", "underperform", "חריג", "חריגים"]):
            tool = self.tool_map.get("detect_anomalies")
            if tool:
                filter_by = self._extract_after_keywords(user_input, [
                    r"anomal(?:y|ies)\s+(?:in|for)\s+(.+)$",
                    r"overperformers?\s+(?:in|for)\s+(.+)$",
                    r"underperformers?\s+(?:in|for)\s+(.+)$",
                    r"חריגים\s+(?:ב|עבור)\s+(.+)$",
                ])
                if filter_by == user_input:
                    filter_by = ""
                return str(tool.invoke(filter_by))

        if any(w in q for w in ["compare", "jaccard", "השווה", "השוואה"]):
            tool = self.tool_map.get("compare_players_jaccard")
            if tool:
                players = self._extract_after_keywords(user_input, [
                    r"compare\s+(.+)$",
                    r"jaccard\s+(.+)$",
                    r"השווה\s+(.+)$",
                ])
                return str(tool.invoke(players))

        if any(w in q for w in ["predict", "who wins", "forecast", "נבא", "תחזה", "מי ינצח"]):
            tool = self.tool_map.get("predict_match")
            matchup = self._extract_matchup(user_input)
            if tool and matchup:
                return str(tool.invoke(matchup))

        if any(w in q for w in ["standings", "table", "league table", "טבלה"]):
            tool = self.tool_map.get("get_live_standings")
            if tool:
                competition = self._extract_after_keywords(user_input, [
                    r"standings\s+(?:for|of|in)?\s*(.+)$",
                    r"table\s+(?:for|of|in)?\s*(.+)$",
                    r"league\s+table\s+(?:for|of|in)?\s*(.+)$",
                    r"טבלה\s+(?:של|ב)?\s*(.+)$",
                ])
                return str(tool.invoke(competition))

        if any(w in q for w in ["world cup", "mundial", "group ", "fixture", "schedule", "מונדיאל", "בית "]):
            tool = self.tool_map.get("world_cup_info")
            if tool:
                return str(tool.invoke(user_input))

        if self._looks_like_scout_query(user_input):
            tool = self.tool_map.get("scout_players")
            if tool:
                return str(tool.invoke(user_input))

        return None

    @staticmethod
    def _extract_text(content) -> str:
        """Gemini can return block lists; extract plain text."""
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
        """Call the LLM. On quota errors, rotate to the next configured model."""
        last_err = None
        for offset in range(len(self.llms)):
            idx = (self.model_idx + offset) % len(self.llms)
            try:
                resp = self.llms[idx].invoke(messages)
                self.model_idx = idx
                return resp
            except Exception as e:
                last_err = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[agent] Model {MODEL_CHAIN[idx]} hit quota; rotating.", flush=True)
                    continue
                raise
        raise last_err

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect Hebrew by the presence of Hebrew Unicode characters."""
        for ch in text:
            if "֐" <= ch <= "׿":
                return "Hebrew"
        return "English"

    def invoke(self, user_input: str, session_id: str = "default") -> str:
        lang = self._detect_language(user_input)
        direct = self._direct_tool_answer(user_input)
        if direct:
            self._remember(session_id, user_input, direct)
            return direct

        if self.qa_pipeline is not None and os.getenv("SCOUTAI_ENABLE_SMART_QA", "0").lower() in {"1", "true", "yes"}:
            try:
                qa = self.qa_pipeline.answerFootballQuestion(user_input, use_gemini=True)
                if qa.intent.intent != "unknown_question" or qa.intent.combined_score >= 0.16:
                    answer = qa.answer
                    self._remember(session_id, user_input, answer)
                    print(
                        f"[agent] smart_qa intent={qa.intent.intent} "
                        f"score={qa.intent.combined_score:.3f} "
                        f"entities={[e.name for e in qa.entities]}",
                        flush=True,
                    )
                    return answer
            except Exception as e:
                print(f"[agent] smart_qa fallback: {e}", flush=True)

        lang_directive = SystemMessage(content=(
            f"LANGUAGE RULE: The user's message is written in {lang}. "
            f"Write your ENTIRE final answer in {lang}. If tool context is in a "
            f"different language, translate it to {lang}. Keep player names, club names, "
            f"and the '🔍 Method:' line unchanged."
        ))
        # History is prepended before the current turn to preserve context across requests.
        history = self._get_history(session_id)
        messages = [self.system_msg, *history, lang_directive,
                    HumanMessage(content=user_input)]
        for _ in range(self.MAX_ITERATIONS):
            try:
                response = self._call_llm(messages)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if lang == "Hebrew":
                        return ("Gemini's free API quota has been reached for today. "
                                "The quota resets daily; please try again later.")
                    return ("We've hit today's free Gemini API quota. "
                            "It resets daily — please try again later.")
                raise
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                answer = self._extract_text(response.content) or "No response."
                if answer.strip() in {"No response.", "No response"}:
                    fallback = self._direct_tool_answer(user_input)
                    if fallback:
                        answer = fallback
                self._remember(session_id, user_input, answer)
                return answer

            for tc in response.tool_calls:
                name, args = tc["name"], tc["args"]
                tool_id    = tc.get("id", name)
                fn = self.tool_map.get(name)
                if fn is None:
                    result = f"Tool '{name}' was not found."
                else:
                    try:
                        if len(args) == 1:
                            result = fn.invoke(next(iter(args.values())))
                        else:
                            result = fn.invoke(args)
                    except Exception as e:
                        result = f"Error running {name}: {e}"
                print(f"[agent] tool={name} | args={args}", flush=True)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        return "The agent reached its iteration limit. Try asking a more focused question."


def build_agent(engine, national_strength, schedule):
    """Build and return the ScoutAgent with all tools."""
    # World Cup nations mapped to nationality values in the player data.
    wc_nations = set()
    if not schedule.empty:
        known = set(engine.df["nationality"].dropna().unique())
        names = set(schedule["team1_name"].dropna()) | set(schedule["team2_name"].dropna())
        for n in names:
            mapped = normalize_nation(n, known) if isinstance(n, str) else None
            if mapped:
                wc_nations.add(mapped)
        print(f"[agent] World Cup: mapped {len(wc_nations)} teams to data nationalities.", flush=True)

    tools = [
        make_find_similar_players_tool(engine),
        make_scout_players_tool(engine, wc_nations),
        make_get_player_archetype_tool(engine),
        make_detect_anomalies_tool(engine),
        make_compare_players_jaccard_tool(engine),
        make_predict_match_tool(engine.df, national_strength),
        make_get_live_standings_tool(),
        make_world_cup_tool(schedule),
    ]

    # Build one tool-bound LLM per configured model for quota rotation.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env or the hosting environment.")

    plain_llms = []
    llms_with_tools = []
    for model_name in MODEL_CHAIN:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_retries=0,
        )
        plain_llms.append(llm)
        llms_with_tools.append(llm.bind_tools(tools))

    qa_pipeline = FootballQAPipeline(engine, national_strength, schedule, plain_llms)

    agent = ScoutAgent(llms_with_tools=llms_with_tools, tools=tools, qa_pipeline=qa_pipeline)
    print(f"[agent] Agent ready ({len(MODEL_CHAIN)} models in rotation).", flush=True)
    return agent
