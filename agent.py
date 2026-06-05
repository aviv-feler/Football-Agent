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
from scouting import ScoutingEngine, parse_scouting_query
from tools.scouting_tools import make_scouting_tools
from tools.get_player_archetype import make_get_player_archetype_tool
from tools.detect_anomalies import make_detect_anomalies_tool
from tools.compare_players_jaccard import make_compare_players_jaccard_tool
from tools.predict_match import make_predict_match_tool
from tools.predict_score import make_prediction_tools
from tools.wc_prediction_tools import make_wc_prediction_tools
from tools.get_live_standings import make_get_live_standings_tool
from tools.get_top_scorers import make_get_top_scorers_tool
from tools.world_cup import make_world_cup_tool
from club_model import ClubModel
from prediction_engine import PredictionEngine, parse_prediction_query
from wc_predictor import WCPredictor

DATA_CSV     = "data/players_clean.csv"
WC_CSV       = "data/fwc26_match_schedule_agent.csv"

SYSTEM_PROMPT = """You are FOOTBOT, an elite AI football intelligence platform built on real Data Science models.
You have a database of ~48,000 players, 10 seasons of top-5 league results, and FIFA World Cup 2026.

IDENTITY & TONE:
- Respond like a confident, precise football analyst — decisive, data-driven, never vague
- Be specific: cite exact numbers, percentages, scores from the tool data
- Give a clear verdict or recommendation — never just say "it depends"
- Structure longer answers with bold headers or bullet points for readability
- Never say "based on my training data", "I think", or "I believe" — present tool facts directly

LANGUAGE: ALWAYS respond in the SAME language the user wrote in.
If Hebrew → full Hebrew response. If English → full English response.

HARD RULES:
1. NEVER answer player stats, rankings, similarity, predictions, standings or fixtures from memory.
   Call the right tool first. If NO tool data was retrieved and the question is about football facts,
   respond ONLY: "I can only answer questions based on our football database. Try asking about a
   specific player, club, or match."
2. Correct misspellings and expand nicknames BEFORE calling any tool:
   "ronado"→"Cristiano Ronaldo", "mbape"→"Kylian Mbappé", "barca"→"Barcelona",
   "man u"→"Manchester United". If ambiguous, ask briefly.
3. You MAY call multiple tools and combine outputs into one rich answer.
4. Tools:
   PREDICTION (extract teams/player/league first, then call the right tool; engine predicts, you narrate):
   - predict_club_match_score(home_team, away_team, user_context)
       → CLUB vs CLUB scoreline. RF classifier + GB xG model + context-aware Poisson scoreline.
         Always resolve which team is home before calling. Pass the raw user sentence as user_context.
   - predict_match(team1, team2)
       → NATIONAL TEAM result (squad-strength + WC Elo).
   - predict_wc_match(team_a, team_b)
       → Specific WC 2026 match: scoreline + probabilities on neutral ground.
   - predict_wc_group(group)
       → Full Group A–L standings simulation + who qualifies.
   - predict_wc_winner()
       → Full tournament Monte Carlo (10k sims) → win probabilities for all 48 teams.
   - predict_top_scorer(league, n)
       → Who will be top scorer next season? RF regressor on player attributes.
   - predict_player_goals(player_name)
       → Goals projection for a specific player next season. Same RF model.
   SCOUTING (extract intent + entities, then call the right one; the engine ranks, you narrate):
   - find_similar_player(player_name)                          — players similar to a reference
   - find_replacement(player_name, club, max_age)             — replacement options for a player
   - search_by_profile(role, positions, max_age, min_potential, important_features, description)
                                                               — free-text profile search
   - find_wonderkids(role, positions, max_age, min_potential) — young high-potential prospects
   OTHER:
   - get_player_archetype(player)      — K-Means role / archetype
   - detect_anomalies(filter)          — Z-score over/under-performers
   - compare_players_jaccard(a vs b)   — side-by-side stats + Jaccard trait overlap
   - predict_match(team1, team2)       — Poisson (clubs) or hybrid Elo+squad (national/WC)
   - get_live_standings(competition)   — LIVE league table / fixtures (football-data.org API)
   - get_top_scorers(competition)      — LIVE current-season top scorers (football-data.org API)
   - world_cup_info(query)             — World Cup 2026 schedule
5. Prefer the LIVE API tools (get_live_standings, get_top_scorers) whenever the user asks
   about the CURRENT/LIVE league table, standings, or this season's top scorers. Use the
   data-science tools for analytical questions (similar / scout / predict / compare /
   archetype / anomalies).
5. Prefer LIVE API tools (get_live_standings, get_top_scorers) for current standings/scorers.
   Use DS tools for analytical questions (similar/scout/predict/compare/archetype/anomalies).
6. Keep the "🔍 Method:" line from every tool output — it is required for academic grading.
7. Cite real numbers from tools, then add one short decisive insight. Never invent data."""


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

    def __init__(self, llms_with_tools: list, tools: list, qa_pipeline=None):
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
        """Deterministic routing used as a fallback when the LLM is unavailable (quota)."""
        q = user_input.lower()

        if any(w in q for w in ["archetype", "type of player", "what type", "ארכיטיפ"]):
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

        if any(w in q for w in ["top scorer", "topscorer", "scorers", "golden boot", "מלך השערים"]):
            tool = self.tool_map.get("get_top_scorers")
            if tool:
                competition = self._extract_after_keywords(user_input, [
                    r"top scorers?\s+(?:in|for|of)?\s*(.+)$",
                    r"scorers?\s+(?:in|for|of)?\s*(.+)$",
                    r"מלך השערים\s+(?:ב|של)?\s*(.+)$",
                ])
                return str(tool.invoke(competition))

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

        # Scouting: parse intent + entities and route to the matching scouting tool.
        ctx = parse_scouting_query(user_input)
        ref = ctx.get("reference_player")
        if ctx["intent"] == "replacement" and ref:
            tool = self.tool_map.get("find_replacement")
            if tool:
                return str(tool.invoke({"player_name": ref, "club": ctx.get("club", ""),
                                        "max_age": ctx.get("age_max", 0)}))
        if ctx["intent"] == "similar" and ref:
            tool = self.tool_map.get("find_similar_player")
            if tool:
                return str(tool.invoke(ref))
        if ctx["intent"] == "wonderkid":
            tool = self.tool_map.get("find_wonderkids")
            if tool:
                return str(tool.invoke({"role": ctx["role"], "max_age": ctx["age_max"] or 21,
                                        "min_potential": ctx["potential_min"] or 80,
                                        "important_features": ",".join(ctx["important_features"])}))
        if self._looks_like_scout_query(user_input) or ctx["role"] or ctx["important_features"]:
            tool = self.tool_map.get("search_by_profile")
            if tool:
                return str(tool.invoke({"role": ctx["role"], "max_age": ctx["age_max"],
                                        "min_potential": ctx["potential_min"],
                                        "important_features": ",".join(ctx["important_features"]),
                                        "description": user_input}))

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
        # Gemini-first: the model reasons over the conversation and decides which tools to
        # call (including the live football-data.org API tools), then writes a smart answer
        # in the user's language. The deterministic keyword router (_direct_tool_answer) is
        # kept only as a fallback for when the LLM is unavailable (e.g. quota exhausted), so
        # the agent still responds with real model output instead of an error.
        lang = self._detect_language(user_input)
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
                    # Quota fallback: answer deterministically from the tools (no LLM).
                    fallback = self._direct_tool_answer(user_input)
                    if fallback:
                        self._remember(session_id, user_input, fallback)
                        return fallback
                    return ("ScoutAI has reached today's free Gemini quota and this question "
                            "needs the language model. The quota resets daily — please try "
                            "again later, or ask directly for similar players, a scouting "
                            "list, a prediction, standings, or top scorers.")
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
                try:
                    print(f"[agent] tool={name} | args={args}", flush=True)
                except Exception:
                    pass  # never let a debug log line break a user request
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Hit the iteration cap without a final answer — fall back to the deterministic router.
        fallback = self._direct_tool_answer(user_input)
        if fallback:
            self._remember(session_id, user_input, fallback)
            return fallback
        return "I couldn't complete that — try asking a more focused question."


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

    club_model = ClubModel()
    if club_model.ok:
        print(f"[agent] Club model (Poisson) ready: {len(club_model.factors)} clubs.", flush=True)
    else:
        print("[agent] Club model unavailable (data/club_matches.csv missing).", flush=True)

    pred_engine = PredictionEngine()
    wc_pred = WCPredictor(schedule, national_strength)
    print(f"[agent] WC predictor ready: {len(wc_pred.all_teams)} teams, {len(wc_pred.groups)} groups.", flush=True)
    print("[agent] Pre-computing WC tournament simulation (5k sims)...", flush=True)
    wc_pred.warm_up()
    print("[agent] WC simulation cached.", flush=True)
    scout = ScoutingEngine(engine)
    print(f"[agent] Scouting engine ready: {len(scout.pool)} players with real attributes.", flush=True)

    tools = [
        *make_prediction_tools(pred_engine),
        *make_wc_prediction_tools(wc_pred),
        *make_scouting_tools(scout),
        make_get_player_archetype_tool(engine),
        make_detect_anomalies_tool(engine),
        make_compare_players_jaccard_tool(engine),
        make_predict_match_tool(engine.df, national_strength, club_model),
        make_get_live_standings_tool(),
        make_get_top_scorers_tool(),
        make_world_cup_tool(schedule),
    ]

    # Build one tool-bound LLM per configured model for quota rotation.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env or the hosting environment.")

    llms_with_tools = []
    for model_name in MODEL_CHAIN:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_retries=0,
        )
        llms_with_tools.append(llm.bind_tools(tools))

    agent = ScoutAgent(llms_with_tools=llms_with_tools, tools=tools)
    print(f"[agent] Agent ready ({len(MODEL_CHAIN)} models, {len(tools)} tools).", flush=True)
    return agent
