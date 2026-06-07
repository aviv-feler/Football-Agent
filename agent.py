"""
agent.py
ScoutAI agent with direct tool-calling support (LangChain / langchain-openai).
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
from langchain_openai import ChatOpenAI
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
from tools.get_national_squad import make_get_national_squad_tool
from tools.get_live_standings import make_get_live_standings_tool
from tools.get_top_scorers import make_get_top_scorers_tool
from tools.get_club_top_scorer import make_get_club_top_scorer_tool
from tools.world_cup import make_world_cup_tool
from club_model import ClubModel
from prediction_engine import PredictionEngine, parse_prediction_query
from wc_predictor import WCPredictor
from match_predictor import MatchPredictor
from viz import split_viz

DATA_CSV     = "data/players_clean.csv"
WC_CSV       = "data/fwc26_match_schedule_agent.csv"
WC_SQUADS_CSV = "data/world_cup_2026_squads.csv"
PREDICTOR_PKL = "predictor_model.pkl"

SYSTEM_PROMPT = """You are FOOTBOT, an elite AI football intelligence platform backed by real Data Science models.
You have a database of ~48,000 players, 10 seasons of top-5 league results, and a full FIFA World Cup 2026 prediction engine.

YOUR ROLE:
You are a knowledgeable football analyst. You reason over real data — but every factual
claim comes from a tool, never from memory.

HOW TO THINK:
1. Understand what the user actually wants to know.
2. FACTS REQUIRE TOOLS. For ANY factual question — results, standings, who won/champion,
   top scorers, fixtures, player stats, current club, squads, predictions, similar players,
   archetypes — you MUST call the matching tool and base your answer ONLY on its output.
   NEVER answer these from your own training memory: it is outdated and will be wrong.
   This rule is the same in EVERY language (Hebrew included). If unsure which tool, pick the
   closest one and call it — do not guess from memory.
3. NORMALIZE ENTITIES TO CANONICAL ENGLISH before every tool call, whatever language the
   user wrote in — tools only understand English names:
   • "הליגה האנגלית" / "פריימר ליג" / "ליגה אנגלית" / "אליפות אנגליה" → "Premier League"
   • "לה ליגה"→"La Liga", "סדרה א"→"Serie A", "בונדסליגה"/"אליפות גרמניה"→"Bundesliga", "ליגה 1"→"Ligue 1"
   • "ריאל מדריד"→"Real Madrid", "ברצלונה"/"barca"→"Barcelona", "מבאפה"/"mbape"→"Kylian Mbappé"
   • "מי זכתה באליפות הליגה האנגלית?" → call get_live_standings("Premier League") and report the leader.
   • When extracting team names from prediction queries like "Brazil vs England prediction",
     strip trailing words like "prediction/result/match" — the teams are "Brazil" and "England".
   CLUB → LEAGUE shortcuts — never ask the user to confirm these mappings:
   • Arsenal/Chelsea/Liverpool/Man City/Man United/Tottenham/Newcastle → "Premier League"
   • Barcelona/Real Madrid/Atletico Madrid → "La Liga"
   • Bayern Munich/Dortmund/Leverkusen → "Bundesliga"
   • Juventus/Inter/AC Milan/Napoli → "Serie A"
   • PSG/Monaco/Lyon/Marseille → "Ligue 1"
4. Call each tool AT MOST ONCE per question. Do NOT call two overlapping tools for the same
   thing (use predict_wc_match OR predict_match, not both; call get_national_squad once).
5. Use your own football knowledge ONLY for opinion/tactics/history questions
   ("what do you think about X", playing style) — NEVER for live facts, results, or stats.

TONE & LENGTH:
- Confident and decisive — give a clear verdict, no hedging.
- BE CONCISE and FAST. A visual CARD is shown automatically for match predictions,
  top-scorer/rankings, similar-player lists, and player profiles. For those answers, write
  only 1–3 sentences of insight and DO NOT re-list the numbers (score, %, xG, attributes,
  ranks) — the card already shows them. Repeating them makes the reply slow and cluttered.
- For everything else, keep it tight; use short bullets only when they genuinely help.
- Never say "I cannot answer" — if it's factual, call the right tool instead.

LANGUAGE: respond in the SAME language the user wrote in (Hebrew → full Hebrew, English →
full English), but ALWAYS pass canonical ENGLISH entity names to tools.

TOOLS — use them to gather data, then reason:
PREDICTION:
- predict_club_match_score(home_team, away_team, user_context) → club scoreline (RF + xG + Poisson)
- predict_wc_match(team_a, team_b)    → WC 2026 match: probabilities + scoreline (neutral ground)
- predict_wc_group(group)             → Group A–L full standings simulation
- predict_wc_winner()                 → tournament Monte Carlo → win probabilities for all 48 teams
- predict_wc_top_scorer(n)            → Golden Boot candidates: goals_per90 × expected games × shooting
- predict_top_scorer(league, n)       → next-season league top scorer (RF regressor)
- predict_player_goals(player_name)   → goals projection for a specific player next season
- predict_match(team1, team2)         → national/club match result (squad-strength + historical ratings)
SCOUTING:
- find_similar_player(player_name)    → cosine-similarity players (use for "similar to X / plays like X")
- find_replacement(player_name, club, max_age, role) → replacement candidates. ALWAYS pass role= when user specifies a position. Map: left-back/LB→"left_back", right-back/RB→"right_back", CB→"centre_back", CDM→"defensive_midfielder", CAM→"creative_attacker", ST→"striker", GK→"goalkeeper". left_back and right_back return ONLY that specific side — use them instead of "fullback" whenever the side is mentioned.
- search_by_profile(role, positions, max_age, min_potential, important_features, description) → find players matching a description (no reference player). Do NOT use for named players or club stats.
- find_wonderkids(role, positions, max_age, min_potential, max_overall) → young prospects. Pass max_overall=70 for cheap/affordable/hidden-gem players.
ANALYSIS:
- get_player_archetype(player)        → NAMED player's archetype + attribute card. Use ONLY for a specific player ("Mbappe's profile", "tell me about Vinicius"). NOT for league tables or rankings.
- detect_anomalies(filter)            → Z-score over/under-performers
- compare_players_jaccard(a vs b)     → side-by-side stats + trait overlap
SQUADS & LIVE DATA:
- get_national_squad(team)            → OFFICIAL WC 2026 squad / roster / called-up players + each player's CURRENT club
- get_live_standings(competition)     → LIVE league table. Use for "show me the [league] table", "Serie A standings", "who won the league".
- get_top_scorers(competition)        → LIVE current-season league scorers (football-data.org)
- get_club_top_scorer(club)           → top scorers for a SPECIFIC CLUB from the player dataset. Use for "who scored most for Chelsea?", "[club]'s top scorer last season".
- world_cup_info(query)               → WC 2026 schedule and fixtures

TOOL STRATEGY:
- "Show me the [league] table / [league] standings / current [league] table" → get_live_standings. Never route a league table request to get_player_archetype.
- "Who is in X's squad / X's World Cup roster" → get_national_squad(X).
- "who won the league / מי זכתה" → get_live_standings(<league>) and report the leader.
- "[PLAYER]'s profile / tell me about [PLAYER] / [PLAYER]'s stats" → get_player_archetype([PLAYER]). Only when a named real player is mentioned.
- "who scored the most goals for [CLUB]? / [CLUB]'s top scorer / who scored for [CLUB] last season?" → get_club_top_scorer([CLUB]). This is NEVER a profile search — do not use search_by_profile for club scorer questions.
- "top scorers in [league] / golden boot" → get_top_scorers([league]).
- "who is the best [position]? / best goalkeeper / top striker" → search_by_profile(role=..., important_features="overall,potential"). Always answer with data — never refuse.
- "cheap/affordable/budget wonderkids" → find_wonderkids(max_overall=70).
- "replacement for [PLAYER] as [position]" → find_replacement(..., role=<normalized role>). If the user corrects a replacement answer ("but he plays left-back", "he is a right-back"), call find_replacement AGAIN with the corrected role — never switch to search_by_profile for a correction.
- "analyze [PLAYER]" → chain get_player_archetype then find_similar_player.
- Match predictions: a 1-1 score IS a draw. Never call it a win for either side.
- SELF-CHECK: Before responding, confirm your answer matches the intent. League table → standings data. Left-back replacement → fullbacks/left-backs in results. Club top scorer → players from that club.

DATA SOURCES & ATTRIBUTION (be honest — never claim a source you didn't use):
- football-data.org is used ONLY by get_live_standings and get_top_scorers. NEVER cite
  football-data.org as the source for a player's club, transfer, or squad — it was not queried.
- A player's club in the scouting/analysis data is a snapshot and can be outdated. For a WC
  national-team player's CURRENT club, use get_national_squad (official 2026 squad clubs).
- Only state a data source that an actual tool result gave you (its "🔍 Method:" line).

ALWAYS keep the "🔍 Method:" line from tool outputs — required for academic grading.
NEVER invent stats. If you cite a number, it must come from a tool result."""


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


# OpenAI model chain. gpt-5-mini is the primary; gpt-4o-mini is a cheaper fallback used
# only if the primary hits a rate limit.
MODEL_CHAIN = [
    "gpt-5-mini",    # primary
    "gpt-4o-mini",   # fallback on rate-limit
]


class _QuotaExhausted(Exception):
    """Raised when every configured model is quota-exhausted (429)."""


class _RecoverableModelError(Exception):
    """Raised when a model rejects the request for a non-quota reason (e.g. a
    cross-model Gemini 2.5 thought_signature mismatch). The turn is retried from
    a clean message history with the next model."""


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

    # Canonical English league name → its English + Hebrew aliases (for normalization).
    _LEAGUE_ALIASES = {
        "Premier League": ["premier league", "epl", "english league", "הליגה האנגלית",
                           "ליגה אנגלית", "פריימר ליג", "פרמייר ליג", "אליפות אנגליה"],
        "La Liga":        ["la liga", "לה ליגה", "הליגה הספרדית", "ליגה ספרדית"],
        "Serie A":        ["serie a", "סריה א", "סדרה א", "ליגה איטלקית"],
        "Bundesliga":     ["bundesliga", "בונדסליגה", "ליגה גרמנית"],
        "Ligue 1":        ["ligue 1", "ליגה צרפתית"],
        "Champions League": ["champions league", "ליגת האלופות"],
    }

    @classmethod
    def _detect_competition(cls, text: str) -> str:
        """Map a query (any language) to a canonical English league name, or ''."""
        q = text.lower()
        for canon, aliases in cls._LEAGUE_ALIASES.items():
            if any(a in q for a in aliases):
                return canon
        return ""

    @staticmethod
    def _extract_after_keywords(text: str, patterns: list[str]) -> str:
        q = text.strip()
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" ?.,:;\"'")
        return q

    # Words that are intent/meta markers, not part of a team name.
    _INTENT_WORDS = re.compile(
        r"\b(prediction|predictions|result|results|match|game|score|preview|analysis|forecast)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_matchup(cls, text: str) -> dict | None:
        q = text.strip()
        # Strip leading intent words ("predict:", "who wins", …)
        cleaned = re.sub(
            r"^\s*(predict|who wins|forecast|נבא|תחזה|מי ינצח)\s*:?\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        # Strip trailing intent/meta words ("Brazil vs England prediction" → "Brazil vs England")
        cleaned = cls._INTENT_WORDS.sub("", cleaned).strip(" ?.,:;\"'")
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

    # Role synonyms used by the deterministic fallback router.
    _ROLE_MAP: dict[str, str] = {
        "left-back": "left_back",   "left back": "left_back",   "lb": "left_back",
        "lwb": "left_back",         "left wing back": "left_back",
        "right-back": "right_back", "right back": "right_back", "rb": "right_back",
        "rwb": "right_back",        "right wing back": "right_back",
        "wing back": "fullback",    "wing-back": "fullback",    "full back": "fullback",
        "centre-back": "centre_back", "center-back": "centre_back", "centre back": "centre_back",
        "center back": "centre_back", "central defender": "centre_back", "cb": "centre_back",
        "cm": "box_to_box",         "box to box": "box_to_box",
        "cdm": "defensive_midfielder", "holding mid": "defensive_midfielder",
        "cam": "creative_attacker",
        "st": "striker",            "cf": "striker",            "centre forward": "striker",
        "gk": "goalkeeper",         "keeper": "goalkeeper",
        "lw": "winger",             "rw": "winger",
    }

    @classmethod
    def _extract_role_from_text(cls, text: str) -> str:
        """Extract a position/role from phrases like 'as a left-back', 'at striker'."""
        q = text.lower()
        m = re.search(
            r"\b(?:as\s+an?\s+|at\s+(?:the\s+)?|playing\s+(?:as\s+)?(?:an?\s+)?)([a-z][\w\s-]{2,20})",
            q,
        )
        if m:
            candidate = m.group(1).strip()
            for key, role in cls._ROLE_MAP.items():
                if key == candidate or candidate.startswith(key):
                    return role
        return ""

    @staticmethod
    def _extract_champion(standings_text: str) -> str | None:
        """Parse the #1 team name from a formatted standings text block."""
        past_sep = False
        for line in standings_text.splitlines():
            if line.startswith("---") or line.startswith("────"):
                past_sep = True
                continue
            if past_sep:
                stripped = line.strip()
                if stripped.startswith("1 ") or stripped.startswith("1\t"):
                    m = re.match(r"^1\s+(.+?)\s{2,}", stripped)
                    if m:
                        return m.group(1).strip()
        return None

    def _direct_tool_answer(self, user_input: str, history: list = None) -> str | None:
        """Deterministic routing used as a fallback when the LLM is unavailable (quota)."""
        q = user_input.lower()

        # Player profile / archetype — only trigger when the query contains "profile",
        # "archetype", or "tell me about" AND successfully extracts a player name.
        # "show me" and "who is" are intentionally excluded: they are too broad and fire
        # for league table requests, goalkeeper questions, etc.
        _profile_triggers = ["archetype", "type of player", "what type", "ארכיטיפ",
                             "profile", "פרופיל", "tell me about", "ספר לי על"]
        _profile_guards   = ["search", "find", "similar", "דומה", "חפש", "best", "top",
                             "wonderkid", "replacement", "table", "standings", "טבלה",
                             "scorer", "who won", "champion"]
        if (any(w in q for w in _profile_triggers)
                and not any(w in q for w in _profile_guards)):
            tool = self.tool_map.get("get_player_archetype")
            if tool:
                player = self._extract_after_keywords(user_input, [
                    r"archetype\s+(?:of|for)\s+(.+)$",
                    r"profile\s+(?:of|for)\s+(.+)$",
                    r"(.+?)(?:'s)\s+profile",
                    r"show\s+me\s+(.+?)\s+(?:player\s+)?profile",
                    r"tell\s+me\s+about\s+(.+)$",
                    r"type\s+of\s+player\s+is\s+(.+)$",
                    r"what\s+type\s+of\s+player\s+is\s+(.+)$",
                    r"ארכיטיפ\s+של\s+(.+)$",
                    r"פרופיל\s+של\s+(.+)$",
                    r"ספר\s+לי\s+על\s+(.+)$",
                ])
                # Only call the tool if a specific name was actually extracted,
                # not if the full raw input was returned unchanged (no pattern matched).
                if player and player.strip() != user_input.strip():
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

        _pred_triggers = [
            "predict", "who wins", "forecast", "נבא", "תחזה", "מי ינצח",
            "what will be the result", "what will the score be", "result of the match",
            "score of the match", "who will win", "what score", "result between",
            "מה יהיה הניצחון", "מה תהיה התוצאה", "מה יהיה הסקור",
        ]
        if any(w in q for w in _pred_triggers):
            tool = self.tool_map.get("predict_match")
            matchup = self._extract_matchup(user_input)
            if tool and matchup:
                return str(tool.invoke(matchup))

        # Club-specific top scorer — must be checked BEFORE the generic scorer path because
        # "goals for Chelsea" triggers "goals" → shooting in parse_scouting_query, which would
        # otherwise fall through to search_by_profile.
        _club_scorer_triggers = [
            "scored the most", "scored most", "most goals for", "most goals at",
            "top scorer for", "top scorer at", "top scorer of", "who scored for",
            "מי הבקיע", "מלך השערים של",
        ]
        if any(w in q for w in _club_scorer_triggers):
            tool = self.tool_map.get("get_club_top_scorer")
            if tool:
                club = self._extract_after_keywords(user_input, [
                    r"(?:most goals|scored most|top scorer)\s+(?:for|at|of)\s+(.+?)(?:\s+(?:last|this|in)\s+season.*)?$",
                    r"who\s+scored\s+(?:the\s+)?most\s+(?:goals\s+)?(?:for|at)\s+(.+?)(?:\s+(?:last|this|in)\s+season.*)?$",
                    r"מלך\s+השערים\s+של\s+(.+?)(?:\s+בעונה.*)?$",
                ])
                if club and club.strip() != user_input.strip():
                    return str(tool.invoke(club.strip()))

        if any(w in q for w in ["top scorer", "topscorer", "scorers", "golden boot", "מלך השערים"]):
            if any(w in q for w in ["world cup", "wc", "mundial", "מונדיאל", "2026"]):
                tool = self.tool_map.get("predict_wc_top_scorer")
                if tool:
                    return str(tool.invoke({"n": 10}))
            tool = self.tool_map.get("get_top_scorers")
            if tool:
                competition = self._extract_after_keywords(user_input, [
                    r"top scorers?\s+(?:in|for|of)?\s*(.+)$",
                    r"scorers?\s+(?:in|for|of)?\s*(.+)$",
                    r"מלך השערים\s+(?:ב|של)?\s*(.+)$",
                ])
                return str(tool.invoke(competition))

        # Standings / "who won the league / champion" — incl. Hebrew.
        comp = self._detect_competition(user_input)
        # If no competition found in current message, search recent history for context
        # (handles follow-ups like "but who won?" after a table was shown).
        if not comp and history:
            for msg in reversed(history[-6:]):
                content = msg.content if hasattr(msg, "content") else str(msg)
                comp = self._detect_competition(content)
                if comp:
                    break
        champion_words = ["who won", "champion", "winner", "won the", "אליפות", "אלופ", "זכת", "ניצח"]
        is_champion_q  = comp and any(w in q for w in champion_words)
        is_table_q     = any(w in q for w in ["standings", "table", "league table", "טבלה"])
        if is_table_q or is_champion_q:
            tool = self.tool_map.get("get_live_standings")
            if tool:
                competition = comp or self._extract_after_keywords(user_input, [
                    r"standings\s+(?:for|of|in)?\s*(.+)$",
                    r"table\s+(?:for|of|in)?\s*(.+)$",
                    r"league\s+table\s+(?:for|of|in)?\s*(.+)$",
                    r"טבלה\s+(?:של|ב)?\s*(.+)$",
                ])
                result_text = str(tool.invoke(competition))
                # For "who won" questions, prepend a direct champion answer extracted from
                # the first row of the standings rather than making the user scan the table.
                if is_champion_q and not is_table_q:
                    champion = self._extract_champion(result_text)
                    if champion:
                        return f"**{champion}** won {competition}.\n\n{result_text}"
                return result_text

        if any(w in q for w in ["squad", "roster", "called up", "call-up", "call up", "סגל"]):
            tool = self.tool_map.get("get_national_squad")
            if tool:
                team = self._extract_after_keywords(user_input, [
                    r"squad\s+(?:of|for)\s+(.+)$",
                    r"roster\s+(?:of|for)\s+(.+)$",
                    r"(.+?)(?:'s)?\s+(?:squad|roster)\b",
                    r"סגל\s+(?:של\s+)?(.+)$",
                ])
                return str(tool.invoke({"team_name": team}))

        if any(w in q for w in ["world cup", "mundial", "group ", "fixture", "schedule", "מונדיאל", "בית "]):
            tool = self.tool_map.get("world_cup_info")
            if tool:
                return str(tool.invoke(user_input))

        # Follow-up position filter — "but only attackers", "only strikers", "just goalkeepers"
        # Re-runs the last scouting query from history with the added position constraint.
        _filter_re = re.search(
            r"\b(?:only|just|but only|show|filter|רק|בלבד)\b.{0,25}"
            r"\b(attacker|striker|forward|winger|midfielder|defender|goalkeeper|"
            r"חלוץ|חלוצים|קיצוני|קשר|בלם|שוער|left.back|right.back)\b",
            q, re.IGNORECASE,
        )
        if _filter_re and history:
            role_from_filter = self._ROLE_MAP.get(_filter_re.group(1).lower()) or _filter_re.group(1).lower()
            for msg in reversed(history[-6:]):
                content = msg.content if hasattr(msg, "content") else str(msg)
                prev_ctx = parse_scouting_query(content)
                if prev_ctx["intent"] == "wonderkid" or "wonderkid" in content.lower() or "cheap" in content.lower():
                    cheap = any(w in content.lower() for w in ["cheap", "affordable", "budget", "זול"])
                    tool = self.tool_map.get("find_wonderkids")
                    if tool:
                        return str(tool.invoke({
                            "role": role_from_filter,
                            "max_age": prev_ctx.get("age_max") or 21,
                            "min_potential": prev_ctx.get("potential_min") or 80,
                            "max_overall": 70 if cheap else 0,
                        }))
                    break

        # Scouting: parse intent + entities and route to the matching scouting tool.
        ctx = parse_scouting_query(user_input)
        ref = ctx.get("reference_player")
        lim = ctx.get("limit", 5)
        if ctx["intent"] == "replacement" and ref:
            tool = self.tool_map.get("find_replacement")
            if tool:
                role_override = self._extract_role_from_text(user_input)
                return str(tool.invoke({"player_name": ref, "club": ctx.get("club", ""),
                                        "max_age": ctx.get("age_max", 0),
                                        "role": role_override}))
        if ctx["intent"] == "similar" and ref:
            tool = self.tool_map.get("find_similar_player")
            if tool:
                return str(tool.invoke(ref))
        if ctx["intent"] == "wonderkid":
            tool = self.tool_map.get("find_wonderkids")
            if tool:
                cheap = any(w in q for w in ["cheap", "affordable", "budget", "low cost",
                                             "value", "hidden gem", "unknown", "זול", "זולים"])
                return str(tool.invoke({"role": ctx["role"], "max_age": ctx["age_max"] or 21,
                                        "min_potential": ctx["potential_min"] or 80,
                                        "important_features": ",".join(ctx["important_features"]),
                                        "max_overall": 70 if cheap else 0,
                                        "limit": lim}))
        if self._looks_like_scout_query(user_input) or ctx["role"] or ctx["important_features"]:
            tool = self.tool_map.get("search_by_profile")
            if tool:
                return str(tool.invoke({"role": ctx["role"], "max_age": ctx["age_max"],
                                        "min_potential": ctx["potential_min"],
                                        "important_features": ",".join(ctx["important_features"]),
                                        "description": user_input,
                                        "limit": lim}))

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
        """Call the LLM, rotating past quota-exhausted models.

        Raises _QuotaExhausted if every model is rate-limited, or
        _RecoverableModelError if a model rejects the (accumulated) history for a
        non-quota reason — the caller then restarts the turn from a clean history."""
        last_err = None
        for offset in range(len(self.llms)):
            idx = (self.model_idx + offset) % len(self.llms)
            try:
                resp = self.llms[idx].invoke(messages)
                self.model_idx = idx
                return resp
            except Exception as e:
                last_err = e
                msg = str(e)
                low = msg.lower()
                # OpenAI rate-limit (429 / RateLimitError) or transient server errors
                # (502/503/500 / ServiceUnavailableError) → rotate to next model.
                if ("429" in msg or "rate_limit" in low or "rate limit" in low
                        or "503" in msg or "502" in msg or "500" in msg
                        or "unavailable" in low or "overloaded" in low
                        or "high demand" in low or "insufficient_quota" in low):
                    print(f"[agent] Model {MODEL_CHAIN[idx]} unavailable/quota ({msg[:60]}); rotating.", flush=True)
                    continue
                raise
        raise _QuotaExhausted(str(last_err))

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect Hebrew by the presence of Hebrew Unicode characters."""
        for ch in text:
            if "֐" <= ch <= "׿":
                return "Hebrew"
        return "English"

    def _run_tool_loop(self, messages: list, user_input: str, viz_box: list) -> str | None:
        """Run the reason→tool-call→narrate loop. Returns the final answer, or None
        if the iteration cap is hit without one. Collects any structured viz payloads
        emitted by tools into viz_box. May raise _QuotaExhausted / _RecoverableModelError."""
        for _ in range(self.MAX_ITERATIONS):
            response = self._call_llm(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                answer = self._extract_text(response.content) or "No response."
                if answer.strip() in {"No response.", "No response"}:
                    fallback = self._direct_tool_answer(user_input, history=None)
                    if fallback:
                        answer = fallback
                return answer

            for tc in response.tool_calls:
                name, args = tc["name"], tc["args"]
                tool_id    = tc.get("id", name)
                fn = self.tool_map.get(name)
                if fn is None:
                    result = f"Tool '{name}' was not found."
                else:
                    try:
                        # A LangChain structured tool accepts a bare string for a single
                        # string field, but rejects a bare non-string scalar (e.g. an int
                        # `n`) — those must stay wrapped in the args dict.
                        if len(args) == 1 and isinstance(next(iter(args.values())), str):
                            result = fn.invoke(next(iter(args.values())))
                        else:
                            result = fn.invoke(args)
                    except Exception as e:
                        result = f"Error running {name}: {e}"
                # Strip any viz payload so the LLM narrates clean text; keep it for the UI.
                clean, viz = split_viz(str(result))
                if viz:
                    viz_box.append(viz)
                try:
                    print(f"[agent] tool={name} | args={args}", flush=True)
                except Exception:
                    pass  # never let a debug log line break a user request
                messages.append(ToolMessage(content=clean, tool_call_id=tool_id))
        return None

    @staticmethod
    def _pick_viz(viz_box: list):
        return viz_box[-1] if viz_box else None

    def invoke(self, user_input: str, session_id: str = "default") -> tuple[str, dict | None]:
        # Gemini-first: the model reasons over the conversation and decides which tools to
        # call (including the live football-data.org API tools), then writes a smart answer
        # in the user's language. The deterministic keyword router (_direct_tool_answer) is
        # kept only as a fallback for when the LLM is unavailable (e.g. quota exhausted), so
        # the agent still responds with real model output instead of an error.
        # Returns (answer_text, viz_payload_or_None); viz drives an in-chat visual card.
        lang = self._detect_language(user_input)
        lang_directive = SystemMessage(content=(
            f"LANGUAGE RULE: The user's message is written in {lang}. "
            f"Write your ENTIRE final answer in {lang}. If tool context is in a "
            f"different language, translate it to {lang}. Keep player names, club names, "
            f"and the '🔍 Method:' line unchanged."
        ))
        # History is prepended before the current turn to preserve context across requests.
        history = self._get_history(session_id)
        base = [self.system_msg, *history, lang_directive, HumanMessage(content=user_input)]

        quota_hit = False
        viz_box: list = []
        recoverable_errors = 0
        # Each restart begins from a clean copy of `base`, so a model that chokes on
        # another model's tool-call history never poisons the retry.
        for _restart in range(len(self.llms)):
            viz_box = []
            try:
                answer = self._run_tool_loop(list(base), user_input, viz_box)
            except _QuotaExhausted:
                quota_hit = True
                break
            except _RecoverableModelError:
                recoverable_errors += 1
                if recoverable_errors >= len(self.llms):
                    # Every model has rejected the history — treat as quota exhausted
                    # so we fall through to the deterministic router instead of looping
                    # until the gunicorn worker is killed.
                    quota_hit = True
                    break
                continue  # _call_llm already advanced model_idx; retry the turn cleanly
            if answer is not None:
                # The final answer is normally clean LLM prose; split_viz also covers the
                # "No response → deterministic fallback" case where a marker slips through.
                answer, inline_viz = split_viz(answer)
                viz = inline_viz or self._pick_viz(viz_box)
                self._remember(session_id, user_input, answer)
                return answer, viz
            break  # iteration cap reached — drop to the deterministic fallback

        # No clean LLM answer — fall back to the deterministic router.
        fallback = self._direct_tool_answer(user_input, history=self._get_history(session_id))
        if fallback:
            clean, viz = split_viz(fallback)
            self._remember(session_id, user_input, clean)
            return clean, viz
        if quota_hit:
            return ("FOOTBOT's AI is temporarily unavailable (rate limit or quota reached). "
                    "Please try again in a moment, or ask directly for similar players, "
                    "a scouting list, a prediction, standings, or top scorers.", None)
        return "I couldn't complete that — try asking a more focused question.", None


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
    # Official 2026 squad lists → used to restrict top-scorer candidates to called-up players.
    if os.path.exists(WC_SQUADS_CSV):
        wc_squads = pd.read_csv(WC_SQUADS_CSV)
        print(f"[agent] WC 2026 squads loaded: {len(wc_squads)} players across "
              f"{wc_squads['team'].nunique()} teams.", flush=True)
    else:
        wc_squads = None
        print(f"[agent] Warning: {WC_SQUADS_CSV} not found — top-scorer won't filter by squad.", flush=True)
    print(f"[agent] WC predictor ready: {len(wc_pred.all_teams)} teams, {len(wc_pred.groups)} groups.", flush=True)
    print("[agent] Pre-computing WC tournament simulation (5k sims)...", flush=True)
    wc_pred.warm_up()
    print("[agent] WC simulation cached.", flush=True)
    scout = ScoutingEngine(engine)
    print(f"[agent] Scouting engine ready: {len(scout.pool)} players with real attributes.", flush=True)

    # Trained Logistic Regression match predictor (squad-strength features).
    match_predictor = None
    if os.path.exists(PREDICTOR_PKL):
        try:
            match_predictor = MatchPredictor(PREDICTOR_PKL)
            print(f"[agent] Match predictor (Logistic Regression) loaded: "
                  f"{match_predictor.n_train} training matches, {len(match_predictor.table)} nations.", flush=True)
        except Exception as e:
            print(f"[agent] Warning: could not load {PREDICTOR_PKL} ({e}); using strength fallback.", flush=True)
    else:
        print(f"[agent] Warning: {PREDICTOR_PKL} not found — run python train_predictor.py. "
              "Using strength fallback for national predictions.", flush=True)

    tools = [
        *make_prediction_tools(pred_engine),
        *make_wc_prediction_tools(wc_pred, players_df=engine.df, squads_df=wc_squads,
                                  match_predictor=match_predictor),
        *make_scouting_tools(scout),
        make_get_player_archetype_tool(engine),
        make_detect_anomalies_tool(engine),
        make_compare_players_jaccard_tool(engine),
        make_predict_match_tool(engine.df, national_strength, club_model, match_predictor=match_predictor),
        make_get_national_squad_tool(wc_squads),
        make_get_live_standings_tool(),
        make_get_top_scorers_tool(),
        make_get_club_top_scorer_tool(engine),
        make_world_cup_tool(schedule),
    ]

    # Build one tool-bound LLM per configured model for rate-limit rotation.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Add it to .env or the hosting environment.")

    llms_with_tools = []
    for model_name in MODEL_CHAIN:
        kwargs = dict(model=model_name, api_key=api_key, max_retries=0)
        # GPT-5 / o-series are reasoning models: by default they "think" before every
        # reply, which made narration calls take 12-16s. The bot's job (tool routing +
        # summarising a tool result in the user's language) needs almost no reasoning,
        # so force minimal effort — this cut total latency ~15s → ~4s in benchmarks.
        # Reasoning models also only accept the default temperature (1), so don't set it.
        is_reasoning = model_name.startswith(("gpt-5", "o1", "o3", "o4"))
        if is_reasoning:
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0.3
        llm = ChatOpenAI(**kwargs)
        llms_with_tools.append(llm.bind_tools(tools))

    agent = ScoutAgent(llms_with_tools=llms_with_tools, tools=tools)
    print(f"[agent] Agent ready ({len(MODEL_CHAIN)} models, {len(tools)} tools).", flush=True)
    return agent
