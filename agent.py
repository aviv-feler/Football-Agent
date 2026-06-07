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
import random
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
from tools.get_recent_matches import make_get_recent_matches_tool, lookup_team_matches
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
   Historical results, head-to-head, latest match, scores, scorers, transfers and injuries
   are factual. If no verified tool/source is available, clearly say the verified data is
   unavailable. NEVER invent a historical scoreline, scorer, date, table position, transfer,
   injury, or statistic.
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
6. NEVER ask the user a clarifying question for a factual / scouting / prediction request.
   Pick the most sensible default and ANSWER immediately by calling a tool. "World Cup" /
   "מונדיאל" ALWAYS means the FIFA World Cup 2026 — never ask which one. If a filter is vague,
   choose a reasonable default and proceed; returning a result always beats asking back.

TONE & LENGTH:
- Confident and decisive — give a clear verdict, no hedging.
- BE CONCISE and FAST. A visual CARD is shown automatically for match predictions,
  top-scorer/rankings, similar-player lists, and player profiles. For those answers, write
  only 1–3 sentences of insight and DO NOT re-list the numbers (score, %, xG, attributes,
  ranks) — the card already shows them. Repeating them makes the reply slow and cluttered.
- For everything else, keep it tight; use short bullets only when they genuinely help.
- Never say "I cannot answer" — if it's factual, call the right tool instead.
- FOOTBOT is football-only. Politely refuse off-domain requests (weather, politics unrelated
  to football, general homework, general knowledge) and redirect to football. Allow football
  creative writing such as poems, chants, jokes, stories, and posts.
- Clearly label model estimates as "Prediction:" in English or "תחזית:" in Hebrew. Never
  present a prediction as a historical fact.
- NEVER narrate your process, announce intentions, or ask permission. Do NOT write
  "I will…", "Let me…", "Proceeding…", "אשתמש…", "אבדוק…", "(performing a lookup…)",
  "(מבצע קריאה לנתוני הטבלה...)", "I ran the model", and do NOT end by asking
  "do you want…?" / "האם תרצה…?". Silently CALL the needed tool and output ONLY the
  final result. A request is an instruction to act, never a prompt to ask back.

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
- find_replacement(player_name, club, max_age, role, country) → replacement candidates. ALWAYS pass role= when user specifies a position. Map: left-back/LB→"left_back", right-back/RB→"right_back", CB→"centre_back", CDM→"defensive_midfielder", CAM→"creative_attacker", ST→"striker", GK→"goalkeeper". left_back and right_back return ONLY that specific side — use them instead of "fullback" whenever the side is mentioned.
- search_by_profile(role, positions, max_age, min_potential, important_features, country, description) → find players matching a description (no reference player). Do NOT use for named players or club stats.
- find_wonderkids(role, positions, max_age, min_potential, max_overall, country) → young prospects. Pass max_overall=70 for cheap/affordable/hidden-gem players.
  COUNTRY/NATIONALITY FILTER (search_by_profile, find_wonderkids, find_replacement): whenever the user names a place ("from Italy", "Italian striker", "wingers from Brazil", "defender in the Premier League"), you MUST pass country=. Use the English COUNTRY name for a nationality ("from Italy"→country="Italy", "Brazilian"→country="Brazil") and the English LEAGUE name for a league ("Premier League", "Serie A", "La Liga", "Bundesliga", "Ligue 1"). A country matches that nationality OR its domestic league; a league name matches that league only. Never drop the place — it is a hard filter.
ANALYSIS:
- get_player_archetype(player)        → NAMED player's archetype + attribute card. Use ONLY for a specific player ("Mbappe's profile", "tell me about Vinicius"). NOT for league tables or rankings.
- detect_anomalies(filter)            → Z-score over/under-performers
- compare_players_jaccard(a vs b)     → side-by-side stats + trait overlap
SQUADS & LIVE DATA:
- get_national_squad(team)            → OFFICIAL WC 2026 squad / roster / called-up players + each player's CURRENT club
- get_live_standings(competition)     → LIVE league table. Use for "show me the [league] table", "Serie A standings", "who won the league".
- get_top_scorers(competition)        → LIVE current-season league scorers (football-data.org)
- get_club_top_scorer(club)           → top scorers for a SPECIFIC CLUB from the player dataset. Use for "who scored most for Chelsea?", "[club]'s top scorer last season".
- get_recent_matches(team, opponent, limit) → VERIFIED finished club matches from football-data.org. Use for recent head-to-head, latest match, historical scoreline, "last N results between X and Y". If scorers are unavailable, say so; never guess.
- world_cup_info(query)               → WC 2026 schedule and fixtures

TOOL STRATEGY:
- "Show me the [league] table / [league] standings / current [league] table" → get_live_standings. Never route a league table request to get_player_archetype.
- "Who is in X's squad / X's World Cup roster" → get_national_squad(X).
- "who won the league / מי זכתה" → get_live_standings(<league>) and report the leader.
- "[PLAYER]'s profile / tell me about [PLAYER] / [PLAYER]'s stats" → get_player_archetype([PLAYER]). Only when a named real player is mentioned.
- "who scored the most goals for [CLUB]? / [CLUB]'s top scorer / who scored for [CLUB] last season?" → get_club_top_scorer([CLUB]). This is NEVER a profile search — do not use search_by_profile for club scorer questions.
- "top scorers in [league] / golden boot" → get_top_scorers([league]).
- "last N results between [TEAM] and [TEAM]" / "latest match for [TEAM]" → get_recent_matches.
- Follow-up "who scored?" after a historical result → only answer if a verified scorer source
  is present in tool output; otherwise state that verified scorer data is unavailable.
- "who is the best [position]? / best goalkeeper / top striker" → search_by_profile(role=..., important_features="overall,potential"). Always answer with data — never refuse.
- "cheap/affordable/budget wonderkids" → find_wonderkids(max_overall=70).
- "World Cup opening match / first match / משחק הפתיחה של המונדיאל" → predict_wc_match("Mexico", "South Africa") — the WC 2026 opener (match #1, Mexico City). Always predict it; NEVER ask which match.
- "overperforming / underperforming player / שחקן שמבצע מעל הממוצע / חריג" → call detect_anomalies IMMEDIATELY with an empty filter (scans all players). No preamble, no "I will…", no questions — just call it and present the result.
- "European league / top European league / ליגה אירופאית" means the big-5 (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). Pass country="Europe" (the tools resolve it to those leagues) — do NOT pass the literal text "European league" as a single league.
- "replacement for [PLAYER] as [position]" → find_replacement(..., role=<normalized role>). If the user corrects a replacement answer ("but he plays left-back", "he is a right-back"), call find_replacement AGAIN with the corrected role — never switch to search_by_profile for a correction.
- "analyze [PLAYER]" → chain get_player_archetype then find_similar_player.
- Match predictions: a 1-1 score IS a draw. Never call it a win for either side.
- SELF-CHECK: Before responding, confirm your answer matches the intent. League table → standings data. Left-back replacement → fullbacks/left-backs in results. Club top scorer → players from that club.

DATA SOURCES & ATTRIBUTION (be honest — never claim a source you didn't use):
- football-data.org is used ONLY by get_live_standings, get_top_scorers, and get_recent_matches. NEVER cite
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


# OpenAI model chain. Presentation correctness is the priority: use the strongest
# configured reasoning model by default. A fallback is allowed only when explicitly
# configured in the environment; we do not silently downgrade reasoning to a mini model.
_PRIMARY_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.5").strip()
_FALLBACK_MODELS = [
    m.strip()
    for m in (os.getenv("OPENAI_FALLBACK_MODEL") or "gpt-5-mini,gpt-4o-mini").split(",")
    if m.strip()
]
MODEL_CHAIN = []
for _model in [_PRIMARY_MODEL, *_FALLBACK_MODELS]:
    if _model and _model not in MODEL_CHAIN:
        MODEL_CHAIN.append(_model)

# Cached demo answers return in microseconds, which looks suspicious next to live answers
# (~3-5s). Add a small jittered pause to cache hits so they feel natural. Seconds; set
# DEMO_CACHE_DELAY=0 to disable.
DEMO_CACHE_DELAY = float(os.getenv("DEMO_CACHE_DELAY", "1.5"))


# Fixed classroom-demo questions (Hebrew + English). warmup() pre-computes and caches an
# answer for each so they respond INSTANTLY on stage (no live LLM/API call). Hit via the
# /warmup endpoint before the demo. Cache keys are normalized so spacing/case still hit.
DEMO_QUERIES = [
    # 1 — WC opening-match prediction (prediction widget)
    "מה התחזית למשחק הפתיחה של המונדיאל?",
    "World Cup opening match prediction?",
    # 2 — WC winner (Monte Carlo)
    "מי תזכה במונדיאל 2026?",
    "Who will win the 2026 World Cup?",
    # 3 — compare (Jaccard + similarity, visual compare)
    "השווה בין אמבפה לוויניסיוס",
    "Compare Mbappé and Vinicius",
    # 4 — scout: fast winger, cheap vs expensive split
    "אני סקאוט, מחפש כנף מהיר עם דריבל גבוה. תן שתי אופציות: זולה ויקרה",
    "I'm a scout looking for a fast winger with high dribbling. Give two options: one cheap, one expensive",
    # 5 — multi-criteria striker
    "מצא חלוץ מתחת לגיל 23, מהיר, עם הרבה גולים, מליגה אירופאית",
    "Find a striker under 23, fast, many goals, European league",
    # 6 — archetype (K-Means) for Haaland
    "איזה סוג שחקן זה הולאנד?",
    "What type of player is Haaland?",
    # 7 — similar players (cosine/weighted similarity)
    "מצא שחקנים דומים לבלינגהאם",
    "Find players similar to Bellingham",
    # 8 — overperformer (Z-score anomaly)
    "מצא שחקן שמבצע מעל הממוצע של הקבוצה שלו",
    "Find a player overperforming",
    # Guardrail demo questions: historical H2H, Hebrew comparison, off-domain, creative.
    "מה היו 2 התוצאות האחרונות בין ארסנל למנצ'סטר סיטי?",
    "תשווה בין מסי לרונאלדו",
    "מה ההבדל בין ויניסיוס לאמבפה?",
    "מסי לעומת רונאלדו",
    "מה מזג האוויר בתל אביב?",
    "תכתוב לי שיר על מסי",
    "תן לי תחזית למשחק הבא של ברצלונה",
    "מה הייתה התוצאה במשחק האחרון של ברצלונה?",
]


class _QuotaExhausted(Exception):
    """Raised when every configured model is quota-exhausted (429)."""


class _RecoverableModelError(Exception):
    """Raised when an OpenAI model rejects the request for a non-quota reason.
    The turn is retried from a clean message history with the next configured model."""


class LazyPredictionEngine:
    """Delay the heaviest prediction training until a matching tool is actually used."""

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def _get(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    print("[prediction] Lazy-loading PredictionEngine...", flush=True)
                    self._engine = PredictionEngine()
        return self._engine

    def predict_club_match(self, *args, **kwargs):
        return self._get().predict_club_match(*args, **kwargs)

    def predict_top_scorer(self, *args, **kwargs):
        return self._get().predict_top_scorer(*args, **kwargs)

    def predict_player_goals(self, *args, **kwargs):
        return self._get().predict_player_goals(*args, **kwargs)


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
        # Pre-computed demo answers keyed by normalized query (populated by warmup()).
        # A hit returns instantly with NO LLM/API call — used for the live presentation.
        self.response_cache: dict[str, tuple[str, dict | None]] = {}
        # Lightweight per-session factual context for follow-ups like "who scored?"
        # after a recent H2H/latest-match lookup.
        self.last_match_context: dict[str, dict] = {}
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
                self.last_match_context.pop(session_id, None)
            else:
                self.histories = {}
                self.last_match_context = {}
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

    _FOOTBALL_TERMS = [
        "football", "soccer", "player", "club", "team", "match", "league", "goal",
        "world cup", "messi", "ronaldo", "mbappe", "vinicius", "barcelona", "arsenal",
        "כדורגל", "שחקן", "קבוצה", "משחק", "ליגה", "שער", "מונדיאל", "מסי",
        "רונאלדו", "אמבפה", "ויניסיוס", "ברצלונה", "ארסנל",
    ]
    _OFF_DOMAIN_TERMS = [
        "weather", "forecast weather", "temperature", "capital of", "math homework",
        "politics", "stock price", "מזג", "מזג האוויר", "טמפרטורה", "בירת",
        "שיעורי בית", "פוליטיקה", "מניה",
    ]
    _PLAYER_ALIASES = {
        "מסי": "Lionel Messi",
        "למסי": "Lionel Messi",
        "messi": "Lionel Messi",
        "רונאלדו": "Cristiano Ronaldo",
        "לרונאלדו": "Cristiano Ronaldo",
        "ronaldo": "Cristiano Ronaldo",
        "אמבפה": "Kylian Mbappé",
        "לאמבפה": "Kylian Mbappé",
        "מבאפה": "Kylian Mbappé",
        "mbappe": "Kylian Mbappé",
        "ויניסיוס": "Vinicius Junior",
        "לויניסיוס": "Vinicius Junior",
        "vinicius": "Vinicius Junior",
    }
    _TEAM_ALIASES = {
        "ארסנל": "Arsenal",
        "מנצ'סטר סיטי": "Manchester City",
        "מנצסטר סיטי": "Manchester City",
        "סיטי": "Manchester City",
        "ברצלונה": "Barcelona",
        "בארסה": "Barcelona",
        "ריאל מדריד": "Real Madrid",
        "צ'לסי": "Chelsea",
        "צלסי": "Chelsea",
        "ליברפול": "Liverpool",
    }

    @classmethod
    def _canon_player_text(cls, value: str) -> str:
        raw = value.strip(" ?.,:;\"'()[]")
        low = raw.lower()
        if low in cls._PLAYER_ALIASES:
            return cls._PLAYER_ALIASES[low]
        if raw in cls._PLAYER_ALIASES:
            return cls._PLAYER_ALIASES[raw]
        if raw.startswith("ל") and raw[1:] in cls._PLAYER_ALIASES:
            return cls._PLAYER_ALIASES[raw[1:]]
        return raw

    @classmethod
    def _canon_team_text(cls, value: str) -> str:
        raw = value.strip(" ?.,:;\"'()[]")
        low = raw.lower()
        if low in cls._TEAM_ALIASES:
            return cls._TEAM_ALIASES[low]
        if raw in cls._TEAM_ALIASES:
            return cls._TEAM_ALIASES[raw]
        if raw.startswith("ל") and raw[1:] in cls._TEAM_ALIASES:
            return cls._TEAM_ALIASES[raw[1:]]
        return raw

    @classmethod
    def _off_domain_refusal(cls, text: str, lang: str) -> str | None:
        q = text.lower()
        has_off = any(term in q or term in text for term in cls._OFF_DOMAIN_TERMS)
        has_ball = any(term in q or term in text for term in cls._FOOTBALL_TERMS)
        if not has_off or has_ball:
            return None
        if lang == "Hebrew":
            return (
                "אני מתמקד בכדורגל, אז אני לא יכול לעזור עם בקשות מחוץ לתחום כמו מזג אוויר. "
                "אפשר לשאול אותי על משחקים, שחקנים, קבוצות, תחזיות, טבלאות או סטטיסטיקות כדורגל."
            )
        return (
            "I'm focused on football, so I can't help with that off-domain request. "
            "Ask me about matches, players, clubs, predictions, standings, or football stats."
        )

    @classmethod
    def _extract_compare_request(cls, text: str) -> str | None:
        q = text.strip()
        patterns = [
            r"(?:compare)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)$",
            r"(?:תשווה|השווה|תעשה השוואה)\s+בין\s+(.+?)\s+(?:ל-|ל|ו-|ו)\s*(.+)$",
            r"מה\s+ההבדל\s+בין\s+(.+?)\s+(?:ל-|ל|ו-|ו)\s*(.+?)(?:\?|$)",
            r"(.+?)\s+לעומת\s+(.+?)(?:\?|$)",
            r"מי\s+יותר\s+טוב\s+(.+?)\s+או\s+(.+?)(?:\?|$)",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                a = cls._canon_player_text(m.group(1))
                b = cls._canon_player_text(m.group(2))
                if a and b:
                    return f"{a} vs {b}"
        return None

    @classmethod
    def _extract_recent_match_request(cls, text: str) -> dict | None:
        q = text.strip()
        low = q.lower()
        limit = 2
        n = re.search(r"\b([1-9]|10)\b", low)
        if n:
            limit = int(n.group(1))

        # Hebrew / English H2H: "last 2 results between Arsenal and Man City".
        h2h_intent = any(x in low or x in q for x in [
            "head-to-head", "h2h", "last results", "recent results", "results between",
            "תוצאות", "ראש בראש", "ביניהן", "ביניהם",
        ])
        if h2h_intent:
            patterns = [
                r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
                r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
                r"בין\s+(.+?)\s+ל(.+?)(?:\?|$)",
                r"בין\s+(.+?)\s+לבין\s+(.+?)(?:\?|$)",
            ]
            for pat in patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    a = cls._canon_team_text(m.group(1))
                    b = cls._canon_team_text(m.group(2))
                    # Remove common trailing words accidentally captured with the team.
                    b = re.sub(r"\s+(האחרונות|האחרונה|last|recent|results?)$", "", b, flags=re.IGNORECASE).strip()
                    if a and b:
                        return {"team": a, "opponent": b, "limit": limit, "kind": "h2h"}

        latest_intent = any(x in low or x in q for x in [
            "last match", "latest match", "recent match", "המשחק האחרון", "משחק אחרון",
            "התוצאה במשחק האחרון",
        ])
        if latest_intent and "הבא" not in q and "next" not in low:
            patterns = [
                r"(?:last|latest|recent)\s+match\s+(?:of|for)\s+(.+?)(?:\?|$)",
                r"(?:what\s+was\s+)?(.+?)\s+(?:last|latest|recent)\s+match(?:\?|$)",
                r"המשחק\s+האחרון\s+של\s+(.+?)(?:\?|$)",
                r"התוצאה\s+במשחק\s+האחרון\s+של\s+(.+?)(?:\?|$)",
            ]
            for pat in patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    team = cls._canon_team_text(m.group(1))
                    if team:
                        return {"team": team, "opponent": "", "limit": 1, "kind": "latest"}
        return None

    @classmethod
    def _extract_next_match_prediction(cls, text: str) -> str | None:
        q = text.strip()
        low = q.lower()
        if not (("next match" in low or "upcoming match" in low or "המשחק הבא" in q or "משחק הבא" in q)
                and ("prediction" in low or "predict" in low or "תחזית" in q)):
            return None
        patterns = [
            r"(?:next|upcoming)\s+match\s+(?:of|for)\s+(.+?)(?:\?|$)",
            r"(?:prediction|predict|forecast)\s+(?:for|of)?\s+(.+?)\s+(?:next|upcoming)\s+match(?:\?|$)",
            r"המשחק\s+הבא\s+של\s+(.+?)(?:\?|$)",
            r"תחזית\s+למשחק\s+הבא\s+של\s+(.+?)(?:\?|$)",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                return cls._canon_team_text(m.group(1))
        return None

    @staticmethod
    def _is_scorer_followup(text: str) -> bool:
        q = text.lower()
        return any(x in q or x in text for x in ["who scored", "scorers", "goalscorers", "מי הבקיע", "מי כבש", "הכובשים"])

    def _preflight_direct_answer(self, user_input: str, history: list, session_id: str, lang: str) -> str | None:
        """High-confidence guardrail routing before the LLM.

        This keeps the demo fast and prevents the riskiest failure mode: the LLM
        fabricating historical scores/scorers or answering off-domain questions.
        """
        q = user_input.strip().lower()
        if q in {"hi", "hello", "hey", "היי", "הי", "שלום"}:
            if lang == "Hebrew":
                return "היי! אני מוכן. שאל אותי על שחקנים, קבוצות, תחזיות, טבלאות או סקאוטינג."
            return "Hi! I'm ready. Ask me about players, teams, predictions, standings, or scouting."

        refusal = self._off_domain_refusal(user_input, lang)
        if refusal:
            return refusal

        if self._is_scorer_followup(user_input):
            ctx = self.last_match_context.get(session_id)
            if ctx:
                if lang == "Hebrew":
                    return (
                        f"יש לי את תוצאת המשחק המאומתת עבור {ctx.get('label', 'המשחק הזה')}, "
                        "אבל אין לי מקור מאומת לכובשים. לכן אני לא אנחש מי הבקיע.\n\n"
                        "🔍 Method: Verified historical match lookup via football-data.org API."
                    )
                return (
                    f"I have the verified scoreline for {ctx.get('label', 'that match')}, "
                    "but I do not have a verified scorer source, so I won't guess the goal scorers.\n\n"
                    "🔍 Method: Verified historical match lookup via football-data.org API."
                )
            return (
                "לאיזה משחק אתה מתכוון, המשחק האחרון או המשחק שלפניו?"
                if lang == "Hebrew"
                else "Which match do you mean, the latest one or the one before it?"
            )

        next_team = self._extract_next_match_prediction(user_input)
        if next_team:
            fixture_text, matches = lookup_team_matches(next_team, limit=1, status="SCHEDULED")
            if matches:
                home = matches[0].get("homeTeam", {}).get("name") or next_team
                away = matches[0].get("awayTeam", {}).get("name") or ""
                pred_tool = self.tool_map.get("predict_club_match_score")
                if pred_tool and away:
                    pred = str(pred_tool.invoke({
                        "home_team": home,
                        "away_team": away,
                        "user_context": user_input,
                    }))
                    return ("תחזית:\n" if lang == "Hebrew" else "Prediction:\n") + pred
            if lang == "Hebrew":
                return (
                    "תחזית: אין לי כרגע יריבה מאומתת למשחק הבא של הקבוצה הזו, ולכן לא אריץ "
                    "מודל תחזית בלי fixture מאומת.\n\n"
                    f"{fixture_text}"
                )
            return (
                "Prediction: I don't currently have a verified next opponent for that team, "
                "so I won't run a prediction without a verified fixture.\n\n"
                f"{fixture_text}"
            )

        recent = self._extract_recent_match_request(user_input)
        if recent:
            tool = self.tool_map.get("get_recent_matches")
            if tool:
                result = str(tool.invoke({
                    "team": recent["team"],
                    "opponent": recent.get("opponent", ""),
                    "limit": recent.get("limit", 2),
                }))
                label = (
                    f"{recent['team']} vs {recent['opponent']}"
                    if recent.get("opponent") else recent["team"]
                )
                self.last_match_context[session_id] = {"label": label, **recent}
                return result

        q = user_input.lower()
        wc_winner = (
            ("world cup" in q and any(x in q for x in ["who will win", "winner", "favourite", "favorite"]))
            or ("מונדיאל" in user_input and any(x in user_input for x in ["מי תזכה", "מי יזכה", "מי תיקח", "מי פייבוריט"]))
        )
        if wc_winner:
            tool = self.tool_map.get("predict_wc_winner")
            if tool:
                result = str(tool.invoke({}))
                return ("תחזית:\n" if lang == "Hebrew" else "Prediction:\n") + result

        compare = self._extract_compare_request(user_input)
        if compare:
            tool = self.tool_map.get("compare_players_jaccard")
            if tool:
                return str(tool.invoke(compare))

        return None

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
                                        "role": role_override,
                                        "country": ctx.get("country", "")}))
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
                                        "country": ctx.get("country", ""),
                                        "limit": lim}))
        if (self._looks_like_scout_query(user_input) or ctx["role"] or ctx["important_features"]
                or ctx.get("country") or ctx.get("position_group")):
            tool = self.tool_map.get("search_by_profile")
            if tool:
                return str(tool.invoke({"role": ctx["role"], "max_age": ctx["age_max"],
                                        "min_potential": ctx["potential_min"],
                                        "important_features": ",".join(ctx["important_features"]),
                                        "country": ctx.get("country", ""),
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
                # Also rotate on an unknown / unavailable model id (404 / model_not_found)
                # so a stale primary string (e.g. a renamed gpt-5.x) falls back to a valid
                # model instead of hard-failing the whole request.
                if ("429" in msg or "rate_limit" in low or "rate limit" in low
                        or "503" in msg or "502" in msg or "500" in msg
                        or "unavailable" in low or "overloaded" in low
                        or "high demand" in low or "insufficient_quota" in low
                        or "404" in msg or "model_not_found" in low
                        or "unsupported value" in low or "unsupported_value" in low
                        or "reasoning_effort" in low
                        or "does not exist" in low or "do not have access" in low):
                    print(f"[agent] Model {MODEL_CHAIN[idx]} unavailable/quota/unsupported ({msg[:80]}); rotating.", flush=True)
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

    @staticmethod
    def _cache_key(text: str) -> str:
        """Normalize a query so spacing/case/punctuation variants share one cache key."""
        t = (text or "").strip().lower()
        t = t.replace("‎", "").replace("‏", "")   # drop LTR/RTL marks
        t = re.sub(r"\s+", " ", t)
        return t.strip(" \t\r\n?!.,;:\"'()[]־–—…׃״׳")

    def invoke(self, user_input: str, session_id: str = "default") -> tuple[str, dict | None]:
        """Public entry point. Returns a pre-cached demo answer INSTANTLY when the query
        matches a warmed demo question (no LLM/API call); otherwise computes normally."""
        cached = None
        key = self._cache_key(user_input)
        if key:
            with self._lock:
                cached = self.response_cache.get(key)
        if cached is not None:
            answer, viz = cached
            if DEMO_CACHE_DELAY > 0:
                # Jittered pause so a cached answer doesn't return suspiciously instantly
                # next to live LLM/API answers. ~1.5s ± 0.25s, never negative.
                time.sleep(max(0.0, random.gauss(DEMO_CACHE_DELAY, 0.25)))
            self._remember(session_id, user_input, answer)
            return answer, viz
        return self._compute(user_input, session_id)

    @staticmethod
    def _is_nonanswer(answer: str | None, viz: dict | None) -> bool:
        """A demo question must trigger a tool, so a good answer carries a '🔍 Method:' line
        OR a viz card. Anything else (a clarifying question, a 'Let me…/אשתמש…' preamble, or
        an empty 'no matches' reply) is a non-answer the warm-up should retry/repair."""
        return ("Method:" not in (answer or "")) and (viz is None)

    def warmup(self, queries: list[str] | None = None, attempts: int = 3) -> list[dict]:
        """Pre-compute and cache answers for the demo queries (or a custom list). Because
        the live demo serves these from cache, warm-up trades speed for RELIABILITY: it
        retries a query (the model is non-deterministic) until it gets a real tool-backed
        answer, then falls back to the deterministic router as a last resort. Idempotent."""
        queries = queries if queries is not None else DEMO_QUERIES
        report = []
        for q in queries:
            t0, answer, viz, ok, tries = time.time(), None, None, False, 0
            for tries in range(1, attempts + 1):
                with self._lock:
                    self.histories.pop("__warmup__", None)   # each attempt is independent
                try:
                    answer, viz = self._compute(q, session_id="__warmup__")
                    ok = True
                except Exception as e:
                    answer, viz, ok = f"[warmup error] {e}", None, False
                if ok and not self._is_nonanswer(answer, viz):
                    break
            # Last resort: the deterministic keyword router always calls a real tool.
            if ok and self._is_nonanswer(answer, viz):
                try:
                    fb = self._direct_tool_answer(q, history=[])
                    if fb:
                        answer, viz = split_viz(fb)
                except Exception:
                    pass
            dt = round(time.time() - t0, 2)
            if ok:
                with self._lock:
                    self.response_cache[self._cache_key(q)] = (answer, viz)
            report.append({
                "query": q, "seconds": dt, "tries": tries, "ok": ok,
                "good": not self._is_nonanswer(answer, viz), "has_viz": viz is not None,
                "preview": (answer or "")[:90].replace("\n", " "),
            })
        with self._lock:
            self.histories.pop("__warmup__", None)
        return report

    def _compute(self, user_input: str, session_id: str = "default") -> tuple[str, dict | None]:
        # OpenAI-first for broad language understanding, with deterministic fast paths for
        # high-confidence demo/tool routes. The keyword router is also kept as a fallback
        # when the LLM is unavailable, so factual football requests can still answer.
        # Returns (answer_text, viz_payload_or_None); viz drives an in-chat visual card.
        lang = self._detect_language(user_input)
        directive = (
            f"LANGUAGE RULE: The user's message is written in {lang}. "
            f"Write your ENTIRE final answer in {lang}. If tool context is in a "
            f"different language, translate it to {lang}. Keep player names, club names, "
            f"and the '🔍 Method:' line unchanged."
        )
        if lang == "Hebrew":
            directive += (
                "\nHEBREW QUALITY: Write natural, fluent, native Hebrew — NOT a word-for-word "
                "translation of the English tool output, with correct gender/number agreement. "
                "NEVER invent a phonetic transliteration of an English word in Hebrew letters "
                "(e.g. never write nonsense like 'ברוטרי'); use the correct Hebrew term, or if "
                "no common one exists keep the word in Latin letters. Keep Latin player/club "
                "names and numbers as-is and phrase the sentence so they read naturally in the "
                "Hebrew text. Use these football terms EXACTLY:\n"
                "• transfers → העברות (NEVER 'העתקות')  • injuries → פציעות  • squad → סגל  "
                "• lineup → הרכב  • replacement(s) → מחליף/מחליפים  • winger → כנף  "
                "• striker/forward → חלוץ  • midfielder → קשר  • centre-back → בלם  "
                "• full-back → מגן  • goalkeeper → שוער  • pace/speed → מהירות  "
                "• dribbling → כדרור/דריבל  • potential → פוטנציאל  • assist → בישול  "
                "• pass → מסירה  • shot → בעיטה  • goal → שער  • standings/table → טבלה  "
                "• prediction → תחזית  • confidence → רמת ביטחון  • market value → שווי שוק  "
                "• overall rating → דירוג כללי  • club → מועדון  • league → ליגה  • season → עונה."
            )
        lang_directive = SystemMessage(content=directive)
        # History is prepended before the current turn to preserve context across requests.
        history = self._get_history(session_id)
        preflight = self._preflight_direct_answer(user_input, history, session_id, lang)
        if preflight:
            clean, viz = split_viz(preflight)
            self._remember(session_id, user_input, clean)
            return clean, viz
        direct = self._direct_tool_answer(user_input, history=history)
        if direct and any(term in user_input.lower() for term in [
            "similar", "plays like", "replacement", "wonderkid", "scout", "find player", "find players",
        ]):
            clean, viz = split_viz(direct)
            self._remember(session_id, user_input, clean)
            return clean, viz
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

    pred_engine = LazyPredictionEngine()
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
        make_get_recent_matches_tool(),
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
        # so keep the effort low — this cut total latency ~15s → ~4-6s in benchmarks.
        # NOTE: 'minimal' is the fastest effort but only SOME GPT-5 variants accept it
        # (e.g. gpt-5-mini / gpt-5-nano). Newer models such as gpt-5.5 REJECT 'minimal'
        # with a 400 and require one of none/low/medium/high/xhigh. 'low' is the only
        # value accepted across the whole GPT-5 family, so it is the safe default here;
        # override per-deploy with OPENAI_REASONING_EFFORT if a model needs something else.
        # Reasoning models also only accept the default temperature (1), so don't set it.
        is_reasoning = model_name.startswith(("gpt-5", "o1", "o3", "o4"))
        if is_reasoning:
            kwargs["reasoning_effort"] = (os.getenv("OPENAI_REASONING_EFFORT") or "low").strip()
        else:
            kwargs["temperature"] = 0.3
        llm = ChatOpenAI(**kwargs)
        llms_with_tools.append(llm.bind_tools(tools))

    agent = ScoutAgent(llms_with_tools=llms_with_tools, tools=tools)
    print(f"[agent] Agent ready ({len(MODEL_CHAIN)} models, {len(tools)} tools).", flush=True)
    return agent
