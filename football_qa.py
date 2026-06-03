"""
football_qa.py
Smart football question-answering pipeline for ScoutAI.

Pipeline:
User question -> preprocess -> intent detection with TF-IDF + Jaccard ->
entity matching -> dataset/API retrieval -> clustering/similarity context ->
Gemini natural-language answer.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_manager import (
    CURRENT_STATS_SHEET,
    FOOTBALL_WORKBOOK_FILE,
    PLAYER_PROFILES_FILE,
    load_player_profiles,
    normalize_league,
    rank_current_players,
)
from ds_engine import jaccard, normalize_nation


API_BASE = "https://api.football-data.org/v4"


LEAGUE_ALIASES = {
    "premier league": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "epl": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "פרמייר ליג": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "פריימר ליג": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "פריימרליג": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "ליגה אנגלית": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "הליגה האנגלית": {"dataset": "GB1", "api": "PL", "label": "Premier League"},
    "bundesliga": {"dataset": "L1", "api": "BL1", "label": "Bundesliga"},
    "בונדסליגה": {"dataset": "L1", "api": "BL1", "label": "Bundesliga"},
    "ליגה גרמנית": {"dataset": "L1", "api": "BL1", "label": "Bundesliga"},
    "ליגה הגרמנית": {"dataset": "L1", "api": "BL1", "label": "Bundesliga"},
    "הליגה הגרמנית": {"dataset": "L1", "api": "BL1", "label": "Bundesliga"},
    "la liga": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "es la liga": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "לה ליגה": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "ליגה ספרדית": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "ליגה הספרדית": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "הליגה הספרדית": {"dataset": "ES1", "api": "PD", "label": "La Liga"},
    "serie a": {"dataset": "IT1", "api": "SA", "label": "Serie A"},
    "סרייה א": {"dataset": "IT1", "api": "SA", "label": "Serie A"},
    "ליגה איטלקית": {"dataset": "IT1", "api": "SA", "label": "Serie A"},
    "ligue 1": {"dataset": "FR1", "api": "FL1", "label": "Ligue 1"},
    "ליגה צרפתית": {"dataset": "FR1", "api": "FL1", "label": "Ligue 1"},
    "champions league": {"dataset": None, "api": "CL", "label": "Champions League"},
    "ליגת האלופות": {"dataset": None, "api": "CL", "label": "Champions League"},
}

NATION_ALIASES = {
    "ברזיל": "Brazil",
    "ארגנטינה": "Argentina",
    "צרפת": "France",
    "גרמניה": "Germany",
    "אנגליה": "England",
    "ספרד": "Spain",
    "איטליה": "Italy",
    "פורטוגל": "Portugal",
    "הולנד": "Netherlands",
    "בלגיה": "Belgium",
    "קרואטיה": "Croatia",
    "אורוגוואי": "Uruguay",
    "קולומביה": "Colombia",
    "מקסיקו": "Mexico",
    "ארצות הברית": "United States",
    "ארהב": "United States",
    'ארה"ב': "United States",
    "קנדה": "Canada",
    "מרוקו": "Morocco",
    "ניגריה": "Nigeria",
    "סנגל": "Senegal",
    "יפן": "Japan",
    "קוריאה": "Korea, South",
}

PLAYER_ALIASES = {
    "enzo fernandez": "Enzo Fernández",
    "enzo fernández": "Enzo Fernández",
    "אנזו פרננדס": "Enzo Fernández",
    "אנסו פרננדס": "Enzo Fernández",
    "קול פאלמר": "Cole Palmer",
    "פאלמר": "Cole Palmer",
    "גוד בלינגהאם": "Jude Bellingham",
    "ג'וד בלינגהאם": "Jude Bellingham",
    "בלינגהאם": "Jude Bellingham",
    "אמבפה": "Kylian Mbappé",
    "מבפה": "Kylian Mbappé",
    "האלנד": "Erling Haaland",
    "מסי": "Lionel Messi",
    "רונאלדו": "Cristiano Ronaldo",
    "ויניסיוס": "Vinicius Junior",
    "סאקה": "Bukayo Saka",
}

TEAM_ALIASES = {
    "arsenal": "Arsenal",
    "ארסנל": "Arsenal",
    "liverpool": "Liverpool",
    "ליברפול": "Liverpool",
    "chelsea": "Chelsea",
    "צ'לסי": "Chelsea",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "מנצ'סטר סיטי": "Manchester City",
    "manchester united": "Manchester United",
    "man united": "Manchester United",
    "tottenham": "Tottenham",
    "spurs": "Tottenham",
    "bayern": "Bayern",
    "bayern munich": "Bayern",
    "באיירן מינכן": "Bayern",
    "באיירן": "Bayern",
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "barcelona": "Barcelona",
    "ברצלונה": "Barcelona",
    "real madrid": "Real Madrid",
    "ריאל מדריד": "Real Madrid",
    "psg": "Paris Saint-Germain",
    "פריז": "Paris Saint-Germain",
}

TEAM_LEAGUE_HINTS = {
    "Chelsea": "Premier League",
    "Arsenal": "Premier League",
    "Liverpool": "Premier League",
    "Manchester City": "Premier League",
    "Manchester United": "Premier League",
    "Tottenham": "Premier League",
    "Bayern": "Bundesliga",
    "Borussia Dortmund": "Bundesliga",
    "Barcelona": "La Liga",
    "Real Madrid": "La Liga",
    "Paris Saint-Germain": "Ligue 1",
}

POSITION_KEYWORDS = {
    "Attack": ["attack", "attacker", "forward", "striker", "winger", "חלוץ", "חלוצים", "שחקן התקפה", "התקפה", "התקפי", "קיצוני"],
    "Midfield": ["midfield", "midfielder", "playmaker", "קשר", "קשרים", "קשר אחורי", "קשר התקפי", "אמצע", "פליימייקר"],
    "Defender": ["defender", "defence", "defense", "centre-back", "fullback", "בלם", "מגן", "הגנה"],
    "Goalkeeper": ["goalkeeper", "keeper", "שוער"],
}

INTENT_EXAMPLES = {
    "best_player": [
        "Who is the best player in the Premier League?",
        "Who is the best football player?",
        "Best player in Brazil",
        "מי השחקן הכי טוב בפריימר ליג?",
        "מי השחקן הכי טוב בברזיל?",
        "מי השחקן הכי טוב בנבחרת צרפת?",
    ],
    "best_attacking_player": [
        "Who is the best attacking player in La Liga?",
        "Best attacker in the Spanish league",
        "Who is the best forward in Bayern Munich?",
        "מי שחקן ההתקפה הכי טוב בליגה הספרדית?",
        "מי שחקן ההתקפה הכי טוב בבאיירן מינכן?",
    ],
    "top_scorer": [
        "Who is the Bundesliga top scorer?",
        "Top scorer in Premier League",
        "Who scored the most goals?",
        "מלך השערים בבונדסליגה",
        "מי הכובש המוביל בליגה הגרמנית?",
    ],
    "player_replacement": [
        "Who can replace Cole Palmer?",
        "Best replacement for Cole Palmer",
        "Alternative for Cole Palmer",
        "Recommend a player to replace Saka",
        "מי יכול להיות מחליף טוב לקול פאלמר?",
        "תחליף טוב לפאלמר",
    ],
    "player_similarity": [
        "Who are the most similar players to Jude Bellingham?",
        "Find players similar to Mbappe",
        "Who plays like Cole Palmer?",
        "Give me similar players to Haaland",
        "מי דומה לבלינגהאם?",
        "שחקנים דומים לאמבפה",
    ],
    "team_comparison": [
        "Which team is stronger, Arsenal or Liverpool?",
        "Compare Arsenal and Liverpool",
        "Who is better Chelsea or Manchester City?",
        "איזו קבוצה חזקה יותר ארסנל או ליברפול?",
        "השווה בין ארסנל לליברפול",
    ],
    "match_prediction": [
        "Predict Brazil vs France",
        "What will be the score between Brazil and France?",
        "Give me a prediction for Chelsea vs Manchester City",
        "Who wins Arsenal against Liverpool?",
        "נבא ברזיל נגד צרפת",
        "מה תהיה התוצאה בין צ'לסי למנצ'סטר סיטי?",
    ],
    "league_analysis": [
        "Which team has the best attack?",
        "Best attack in Premier League",
        "Analyze the Bundesliga",
        "Who has the strongest squad in the league?",
        "לאיזו קבוצה יש את ההתקפה הכי טובה?",
    ],
    "national_team_analysis": [
        "Who is the best player in France national team?",
        "Analyze Brazil national team",
        "Best players for Germany",
        "מי השחקן הכי טוב בנבחרת צרפת?",
        "נתח את נבחרת ברזיל",
    ],
    "player_comparison": [
        "Compare Messi and Ronaldo",
        "Compare Cole Palmer vs Saka",
        "Who is better Bellingham or Musiala?",
        "השווה בין מסי לרונאלדו",
        "מי טוב יותר סאקה או פאלמר?",
    ],
    "general_football_question": [
        "Explain why this player is good",
        "What makes a striker effective?",
        "What do you think about this team?",
        "תסביר לי על כדורגל",
        "מה הופך חלוץ לטוב?",
    ],
}


@dataclass
class IntentResult:
    intent: str
    tfidf_score: float
    jaccard_score: float
    combined_score: float
    matched_template: str


@dataclass
class EntityMatch:
    name: str
    entity_type: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    intent: IntentResult
    entities: list[EntityMatch]
    context: str
    answer: str
    clustering_used: bool
    methods: list[str]
    retrieved_count: int = 0
    top_candidates: list[str] = field(default_factory=list)
    prompt: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    last_intent: str | None = None
    last_player: str | None = None
    last_team: str | None = None
    target_team: str | None = None
    league: str | None = None
    position: str | None = None
    requirements: list[str] = field(default_factory=list)
    previous_question: str | None = None
    previous_answer: str | None = None


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nf = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in nf if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s'\"-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) > 1}


def calculateJaccardSimilarity(text1: str, text2: str) -> float:
    return jaccard(tokenize(text1), tokenize(text2))


calculate_jaccard_similarity = calculateJaccardSimilarity


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _pct(x: float) -> str:
    return f"{round(x * 100, 1)}%"


def _pct_capped(x: float) -> str:
    return f"{round(min(max(x, 0.0), 1.0) * 100, 1)}%"


class FootballDataClient:
    def __init__(self):
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self.api_key}

    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_standings(self, api_competition: str) -> list[dict[str, Any]]:
        if not self.enabled() or not api_competition:
            return []
        try:
            resp = requests.get(
                f"{API_BASE}/competitions/{api_competition}/standings",
                headers=self._headers(),
                timeout=8,
            )
            if resp.status_code != 200:
                return []
            standings = resp.json().get("standings", [])
            for table in standings:
                if table.get("type") == "TOTAL":
                    return table.get("table", [])
            return standings[0].get("table", []) if standings else []
        except requests.RequestException:
            return []

    def get_scorers(self, api_competition: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.enabled() or not api_competition:
            return []
        try:
            resp = requests.get(
                f"{API_BASE}/competitions/{api_competition}/scorers",
                headers=self._headers(),
                params={"limit": limit},
                timeout=8,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("scorers", [])
        except requests.RequestException:
            return []


class FootballQAPipeline:
    def __init__(self, engine, national_strength: pd.DataFrame, schedule: pd.DataFrame, llms: list | None = None):
        self.engine = engine
        self.selected_data_source = "player_profiles"
        self.selected_source_file = str(PLAYER_PROFILES_FILE).replace("\\", "/")
        self.df = self._load_current_profiles(engine)
        self.national_strength = national_strength
        self.schedule = schedule
        self.llms = llms or []
        self.model_idx = 0
        self.api = FootballDataClient()
        self.conversation = ConversationContext()
        self.last_debug: dict[str, Any] = {}
        self._last_context_is_followup = False
        self.debug_enabled = os.getenv("SCOUTAI_DEBUG", "1").lower() not in {"0", "false", "no"}

        self.intent_rows = [
            (intent, example)
            for intent, examples in INTENT_EXAMPLES.items()
            for example in examples
        ]
        self.intent_texts = [row[1] for row in self.intent_rows]
        self.intent_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.intent_matrix = self.intent_vectorizer.fit_transform(self.intent_texts)

        self.player_names = self.df["player_name"].fillna("").astype(str).tolist()
        self.player_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
        self.player_matrix = self.player_vectorizer.fit_transform(self.player_names)

        clubs = sorted({c for c in self.df["club"].dropna().astype(str).unique() if c and c.lower() != "nan"})
        nations = sorted({n for n in self.df["nationality"].dropna().astype(str).unique() if n and n.lower() != "nan"})
        self.team_entities = (
            [(c, "club") for c in clubs]
            + [(n, "national_team") for n in nations]
        )
        self.team_names = [name for name, _ in self.team_entities]
        self.team_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
        self.team_matrix = self.team_vectorizer.fit_transform(self.team_names)

    def _load_current_profiles(self, engine) -> pd.DataFrame:
        """Use current regenerated profiles while preserving feature-matrix row alignment."""
        try:
            profiles = load_player_profiles()
            if len(profiles) != len(engine.df):
                raise ValueError(f"profile row count {len(profiles)} != feature row count {len(engine.df)}")
            if "player_id" in profiles.columns and "player_id" in engine.df.columns:
                same_order = profiles["player_id"].astype(str).reset_index(drop=True).equals(
                    engine.df["player_id"].astype(str).reset_index(drop=True)
                )
                if not same_order:
                    raise ValueError("player_profiles.csv is not aligned to player_features.npy")
            return profiles.reset_index(drop=True)
        except Exception as exc:
            print(f"[football_qa] data_manager profile load failed, falling back to engine.df: {exc}", flush=True)
            self.selected_data_source = "players_clean_fallback"
            self.selected_source_file = "data/players_clean.csv"
            return engine.df.reset_index(drop=True)

    def reset_context(self):
        self.conversation = ConversationContext()
        self.last_debug = {}

    @staticmethod
    def _dedupe_entities(entities: list[EntityMatch]) -> list[EntityMatch]:
        deduped: list[EntityMatch] = []
        seen = set()
        for ent in entities:
            key = (ent.entity_type, ent.name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ent)
        return deduped

    def detectIntent(self, question: str) -> IntentResult:
        q_vec = self.intent_vectorizer.transform([question])
        tfidf_scores = cosine_similarity(q_vec, self.intent_matrix)[0]

        best = IntentResult("unknown_question", 0.0, 0.0, 0.0, "")
        for i, (intent, template) in enumerate(self.intent_rows):
            tfidf_score = float(tfidf_scores[i])
            jac = calculateJaccardSimilarity(question, template)
            combined = 0.68 * tfidf_score + 0.32 * jac
            if combined > best.combined_score:
                best = IntentResult(intent, tfidf_score, jac, combined, template)

        q = normalize_text(question)
        if any(w in q for w in ["young", "talent", "potential", "prospect", "צעיר", "כישרון", "פוטנציאל"]):
            best = IntentResult("league_analysis", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.58), best.matched_template)
            return best

        best_words = ["best", "top", "strongest", "הכי טוב", "הטוב ביותר", "מוביל"]
        attack_words = ["attack", "attacker", "forward", "winger", "striker", "התקפה", "חלוץ", "חלוצים", "קיצוני"]
        if any(w in q for w in best_words) and any(w in q for w in attack_words):
            return IntentResult(
                "best_attacking_player",
                best.tfidf_score,
                best.jaccard_score,
                max(best.combined_score, 0.62),
                best.matched_template,
            )

        if any(w in q for w in ["vs", "against", "נגד", "תוצאה", "score", "predict", "נבא", "תחזה"]):
            if best.intent in {"team_comparison", "general_football_question", "unknown_question"}:
                best = IntentResult("match_prediction", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.55), best.matched_template)
        elif any(w in q for w in ["replace", "replacement", "alternative", "תחליף", "מחליף"]):
            best = IntentResult("player_replacement", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.60), best.matched_template)
        elif any(w in q for w in ["similar", "plays like", "דומה", "דומים", "כמו"]):
            best = IntentResult("player_similarity", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.60), best.matched_template)
        elif any(w in q for w in ["compare", "השווה", "better", "טוב יותר", "חזקה יותר"]):
            if best.intent not in {"player_comparison", "team_comparison"}:
                best = IntentResult("team_comparison", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.50), best.matched_template)
        elif any(w in q for w in ["top scorer", "scorer", "goals", "מלך", "כובש", "שערים"]):
            best = IntentResult("top_scorer", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.55), best.matched_template)
        elif any(w in q for w in ["national team", "נבחרת"]):
            if best.intent == "best_player":
                best = IntentResult("national_team_analysis", best.tfidf_score, best.jaccard_score, max(best.combined_score, 0.55), best.matched_template)

        if best.combined_score < 0.16:
            if any(w in q for w in ["football", "soccer", "כדורגל", "שחקן", "קבוצה", "נבחרת"]):
                return IntentResult("general_football_question", best.tfidf_score, best.jaccard_score, best.combined_score, best.matched_template)
            return IntentResult("unknown_question", best.tfidf_score, best.jaccard_score, best.combined_score, best.matched_template)
        return best

    detect_intent = detectIntent

    def calculateTfidfSimilarity(self, question: str, templates: list[str]) -> list[float]:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        matrix = vectorizer.fit_transform(templates + [question])
        return cosine_similarity(matrix[-1], matrix[:-1])[0].tolist()

    calculate_tfidf_similarity = calculateTfidfSimilarity

    def _extract_requirements(self, question: str) -> list[str]:
        q = normalize_text(question)
        reqs = []
        groups = {
            "defensive": ["defensive", "defend", "tackles", "interceptions", "הגנתי", "הגנה", "קשר אחורי"],
            "creative": ["creative", "create", "creator", "playmaker", "assists", "בישולים", "יצירתי", "יוצר", "מבשל", "קשר התקפי"],
            "possession": ["possession", "control", "passing", "passes", "פוזשן", "החזקה", "מסירות", "שליטה"],
            "high budget": ["high budget", "expensive", "elite", "big budget", "תקציב גבוה", "יקר", "עלית"],
            "low budget": ["low budget", "cheap", "bargain", "value", "תקציב נמוך", "זול", "מציאה"],
            "young": ["young", "talent", "potential", "prospect", "צעיר", "כישרון", "פוטנציאל"],
        }
        for label, words in groups.items():
            if any(normalize_text(w) in q for w in words):
                reqs.append(label)
        return reqs

    def _is_followup_or_correction(self, question: str) -> bool:
        q = normalize_text(question)
        phrase_markers = [
            "i want", "i meant", "actually", "team is", "also",
            "אני רוצה", "אני מחפש", "התכוונתי", "בעצם", "עבור", "בשביל", "לקבוצה", "וגם",
        ]
        token_markers = {"no", "for", "לא"}
        return (
            any(m in q for m in phrase_markers)
            or bool(tokenize(q) & token_markers)
            or bool(self._extract_requirements(question))
        )

    def _mentions_national_team(self, question: str) -> bool:
        q = normalize_text(question)
        return any(marker in q for marker in ["national team", "nation", "squad", "נבחרת"])

    def _should_reset_context(self, question: str) -> bool:
        q = normalize_text(question)
        markers = [
            "forget that", "new question", "start over", "reset", "ignore previous",
            "in general", "generally", "different question",
            "עזוב", "תעזוב רגע", "שאלתי באופן כללי", "בכללי", "שאלה חדשה", "תתחיל מחדש",
        ]
        if any(normalize_text(marker) in q for marker in markers):
            return True
        if q in {"no", "לא"}:
            return True
        return False

    def _contextualize_question(self, question: str, intent: IntentResult) -> tuple[str, IntentResult, list[str]]:
        additions: list[str] = []
        new_intent = intent
        ctx = self.conversation
        is_followup = self._is_followup_or_correction(question)

        explicit_players = [e for e in self.findBestEntityMatch(question, "player", 1) if e.score >= 0.65]
        explicit_teams = [e for e in self.findBestEntityMatch(question, "team", 3) if e.entity_type in {"club", "national_team"} and e.score >= 0.45]
        league = self._match_league(question)
        position = self._match_position(question)
        requirements = self._extract_requirements(question)

        if is_followup and ctx.last_intent:
            if intent.intent in {"unknown_question", "general_football_question", "league_analysis", "best_player", "match_prediction", "national_team_analysis"} or requirements:
                if ctx.last_intent in {"player_replacement", "player_similarity"}:
                    new_intent = IntentResult(ctx.last_intent, intent.tfidf_score, intent.jaccard_score, max(intent.combined_score, 0.62), intent.matched_template)
                    additions.append(f"follow-up intent inherited: {ctx.last_intent}")

        enriched = question
        if is_followup and new_intent.intent in {"player_replacement", "player_similarity"} and not explicit_players and ctx.last_player:
            enriched += f" {ctx.last_player}"
            additions.append(f"player inherited: {ctx.last_player}")
        if is_followup and not explicit_teams and ctx.target_team:
            enriched += f" for {ctx.target_team}"
            additions.append(f"target team inherited: {ctx.target_team}")
        if is_followup and not league and ctx.league:
            enriched += f" in {ctx.league}"
            additions.append(f"league inherited: {ctx.league}")
        if is_followup and not position and ctx.position:
            enriched += f" {ctx.position}"
            additions.append(f"position inherited: {ctx.position}")
        merged_reqs = list(dict.fromkeys([*(ctx.requirements if is_followup else []), *requirements]))
        if merged_reqs:
            enriched += " requirements: " + ", ".join(merged_reqs)
            additions.append(f"requirements: {', '.join(merged_reqs)}")
        self._last_context_is_followup = is_followup
        self.last_debug["is_followup"] = is_followup

        return enriched, new_intent, additions

    def _update_context(self, question: str, intent: IntentResult, entities: list[EntityMatch], answer: str):
        self.conversation.last_intent = intent.intent
        self.conversation.previous_question = question
        self.conversation.previous_answer = answer
        reqs = self._extract_requirements(question)
        if self._last_context_is_followup:
            self.conversation.requirements = list(dict.fromkeys([*self.conversation.requirements, *reqs]))[-8:]
        else:
            self.conversation.requirements = reqs

        position = self._match_position(question)
        if position:
            self.conversation.position = position

        league = next((e for e in entities if e.entity_type == "league"), None)
        if league:
            self.conversation.league = league.name

        player = next((e for e in entities if e.entity_type == "player"), None)
        if player:
            self.conversation.last_player = player.name

        teams = [e for e in entities if e.entity_type in {"club", "national_team"}]
        if teams:
            self.conversation.last_team = teams[0].name
            if intent.intent in {"player_replacement", "player_similarity", "best_player", "league_analysis"}:
                self.conversation.target_team = teams[0].name

    def _match_league(self, question: str) -> EntityMatch | None:
        q = normalize_text(question)
        for alias, info in LEAGUE_ALIASES.items():
            if normalize_text(alias) in q:
                return EntityMatch(info["label"], "league", 1.0, info)
        return None

    def _match_position(self, question: str) -> str | None:
        q = normalize_text(question)
        for pos, words in POSITION_KEYWORDS.items():
            if any(normalize_text(w) in q for w in words):
                return pos
        return None

    def _match_sub_positions(self, question: str) -> list[str]:
        q = normalize_text(question)
        roles = []
        role_map = {
            "Centre-Forward": ["striker", "center forward", "centre forward", "חלוץ", "חלוצים"],
            "Attacking Midfield": ["attacking midfielder", "playmaker", "קשר התקפי", "פליימייקר"],
            "Defensive Midfield": ["defensive midfielder", "dm", "cdm", "קשר אחורי", "הגנתי"],
            "Central Midfield": ["central midfielder", "cm", "קשר מרכזי", "קשר"],
            "Right Winger": ["right winger", "קיצוני ימני"],
            "Left Winger": ["left winger", "קיצוני שמאלי"],
            "Centre-Back": ["centre back", "center back", "בלם"],
        }
        for subpos, words in role_map.items():
            if any(normalize_text(w) in q for w in words):
                roles.append(subpos)
        return roles

    def findBestEntityMatch(self, question: str, entity_type: str = "player", top_n: int = 3) -> list[EntityMatch]:
        q_norm = normalize_text(question)

        if entity_type == "player":
            for alias, canonical in PLAYER_ALIASES.items():
                if normalize_text(alias) in q_norm:
                    idx = self.engine.find_index(canonical)
                    if idx is not None:
                        return [EntityMatch(self.df.iloc[idx]["player_name"], "player", 1.0, {"iloc": idx})]

            vec = self.player_vectorizer.transform([question])
            scores = cosine_similarity(vec, self.player_matrix)[0]
            matches: list[EntityMatch] = []
            for idx in np.argsort(scores)[::-1][:top_n * 6]:
                name = self.player_names[int(idx)]
                if not name:
                    continue
                contains = normalize_text(name) in q_norm or any(part in q_norm for part in normalize_text(name).split() if len(part) > 3)
                jac = calculateJaccardSimilarity(question, name)
                score = max(float(scores[idx]), jac) + (0.25 if contains else 0.0)
                if score >= 0.30 or contains:
                    matches.append(EntityMatch(name, "player", min(score, 1.0), {"iloc": int(idx)}))
                if len(matches) >= top_n:
                    break
            return matches

        league = self._match_league(question)
        matches = [league] if league else []

        known_nations = sorted(set(self.df["nationality"].dropna().astype(str)))
        for nation in known_nations:
            n = normalize_text(nation)
            if len(n) > 3 and re.search(rf"\b{re.escape(n)}\b", q_norm):
                matches.append(EntityMatch(nation, "national_team", 1.0, {}))

        for club in self.df["club"].dropna().astype(str).unique():
            c = normalize_text(club)
            if len(c) > 4 and c in q_norm:
                matches.append(EntityMatch(club, "club", 1.0, {}))

        for alias, canonical in {**TEAM_ALIASES, **NATION_ALIASES}.items():
            if normalize_text(alias) in q_norm:
                nation = normalize_nation(canonical, set(known_nations))
                if nation:
                    matches.append(EntityMatch(nation, "national_team", 1.0, {}))
                else:
                    club = self._best_club_name(canonical)
                    if club:
                        meta = {"league_hint": TEAM_LEAGUE_HINTS.get(canonical)}
                        matches.append(EntityMatch(club, "club", 1.0, meta))

        vec = self.team_vectorizer.transform([question])
        scores = cosine_similarity(vec, self.team_matrix)[0]
        for idx in np.argsort(scores)[::-1][:top_n * 8]:
            name, kind = self.team_entities[int(idx)]
            n_name = normalize_text(name)
            contains = n_name in q_norm or any(part in q_norm for part in n_name.split() if len(part) > 4)
            jac = calculateJaccardSimilarity(question, name)
            score = max(float(scores[idx]), jac) + (0.2 if contains else 0.0)
            if score >= 0.38 or contains:
                candidate = EntityMatch(name, kind, min(score, 1.0), {})
                if not any(m.name == candidate.name and m.entity_type == candidate.entity_type for m in matches):
                    matches.append(candidate)
            if len(matches) >= top_n:
                break
        return matches[:top_n]

    find_best_entity_match = findBestEntityMatch

    def _best_club_name(self, partial: str) -> str | None:
        p = normalize_text(partial)
        candidates = []
        for club in self.df["club"].dropna().astype(str).unique():
            c = normalize_text(club)
            if p in c:
                sub = self.df[self.df["club"] == club]
                value = sub["market_value_in_eur"].fillna(0).sum()
                candidates.append((value, club))
        return max(candidates)[1] if candidates else None

    def _extract_match_teams(self, question: str) -> list[EntityMatch]:
        between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)", question, flags=re.IGNORECASE)
        if between:
            parts = [between.group(1), between.group(2)]
        else:
            parts = re.split(r"\bvs\b|\bagainst\b|\bversus\b|\bbetween\b|נגד|מול|בין", question, flags=re.IGNORECASE)
        teams: list[EntityMatch] = []
        if len(parts) >= 2:
            left = self.findBestEntityMatch(parts[0], "team", 1)
            right = self.findBestEntityMatch(parts[1], "team", 1)
            teams.extend([m for m in left + right if m.entity_type in {"club", "national_team"}])
        if len(teams) < 2:
            teams = [m for m in self.findBestEntityMatch(question, "team", 4) if m.entity_type in {"club", "national_team"}]
        dedup = []
        for team in teams:
            if not any(t.name == team.name for t in dedup):
                dedup.append(team)
        return dedup[:2]

    def buildPlayerClusters(self, dataset: pd.DataFrame | None = None) -> dict[str, Any]:
        df = dataset if dataset is not None else self.df
        return {
            "algorithm": "Position-aware K-Means",
            "k": self.engine.k,
            "archetypes": df.get("position_archetype", pd.Series(dtype=str)).value_counts().to_dict(),
            "cluster_counts": df["position_cluster"].value_counts().sort_index().to_dict() if "position_cluster" in df.columns else df["cluster"].value_counts().sort_index().to_dict(),
        }

    build_player_clusters = buildPlayerClusters

    def _profile_trait_set(self, iloc: int) -> set[str]:
        row = self.df.iloc[iloc]
        traits = set()
        for col, prefix in [
            ("position", "pos"), ("sub_position", "subpos"), ("nationality", "nat"),
            ("league", "league"), ("age_bucket", "age"), ("value_tier", "val"),
            ("position_archetype", "profile"), ("position_cluster", "pcluster"),
        ]:
            value = row.get(col)
            if pd.notna(value) and str(value):
                traits.add(f"{prefix}:{value}")
        return traits

    def _player_line(self, row: pd.Series, extra: str = "") -> str:
        value = int(_safe_float(row.get("market_value_in_eur")))
        profile = row.get("position_archetype", row.get("archetype"))
        cluster = row.get("position_cluster", row.get("cluster"))
        return (
            f"{row.get('player_name')} | {row.get('position')} / {row.get('sub_position')} | "
            f"{row.get('club')} | {row.get('nationality')} | age {int(_safe_float(row.get('age')))} | "
            f"goals {int(_safe_float(row.get('goals')))}, assists {int(_safe_float(row.get('assists')))}, "
            f"minutes {int(_safe_float(row.get('minutes_played'))):,}, value EUR {value:,} | "
            f"profile cluster {cluster} ({profile}){extra}"
        )

    def _current_player_line(self, row: pd.Series, extra: str = "") -> str:
        return (
            f"{row.get('Player_Name')} | {row.get('Position')} | {row.get('Club')} | "
            f"{normalize_league(row.get('League'))} | age {int(_safe_float(row.get('Age')))} | "
            f"goals {int(_safe_float(row.get('Goals')))}, assists {int(_safe_float(row.get('Assists')))}, "
            f"shots {int(_safe_float(row.get('Shots')))}, SOT {int(_safe_float(row.get('Shots_On_Target')))}, "
            f"minutes {int(_safe_float(row.get('Minutes_Played'))):,}{extra}"
        )

    def findSimilarPlayers(
        self,
        playerName: str,
        replacement: bool = False,
        limit: int = 5,
        question: str = "",
        target_team: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        match = self.findBestEntityMatch(playerName, "player", 1)
        if not match:
            return f"No player match found for '{playerName}'.", [], False
        idx = int(match[0].meta["iloc"])
        target = self.df.iloc[idx]
        if "position_group" in self.df.columns and pd.notna(target.get("position_group")):
            same_position = self.df["position_group"] == target["position_group"]
        else:
            same_position = self.df["position"] == target["position"]
        same_sub_position = self.df["sub_position"] == target.get("sub_position")
        cluster_col = "position_cluster" if "position_cluster" in self.df.columns else "cluster"
        same_cluster = self.df[cluster_col] == target[cluster_col]
        mask = (same_sub_position | same_position) & (self.df.index != idx)
        if replacement:
            mask = mask & same_cluster
        if target_team:
            # For replacement recommendations, prefer outside candidates, not the player already in the target club.
            mask = mask & (self.df["club"] != target_team)
        cand_ilocs = np.where(mask.values)[0]
        if len(cand_ilocs) < limit:
            cand_ilocs = np.where(((same_sub_position | same_position) & (self.df.index != idx)).values)[0]
        if len(cand_ilocs) == 0:
            return f"No comparable candidates found for {target['player_name']}.", [], True

        cosine_scores = self.engine.cosine(idx, cand_ilocs)
        target_traits = self._profile_trait_set(idx)
        market = self.df.iloc[cand_ilocs]["market_value_in_eur"].fillna(0).clip(lower=0).values
        quality = np.log1p(market) / (np.log1p(market.max()) if market.max() > 0 else 1.0)

        reqs = self._extract_requirements(question)
        q = normalize_text(question)
        target_league = None
        if target_team:
            club_rows = self.df[self.df["club"] == target_team]
            if not club_rows.empty:
                target_league = club_rows["league"].mode().iloc[0]

        rows = []
        for pos, iloc in enumerate(cand_ilocs):
            row = self.df.iloc[int(iloc)]
            jac = jaccard(target_traits, self._profile_trait_set(int(iloc)))
            cluster_bonus = 1.0 if int(row[cluster_col]) == int(target[cluster_col]) else 0.0
            score = 0.48 * float(cosine_scores[pos]) + 0.22 * jac + 0.18 * float(quality[pos]) + 0.12 * cluster_bonus
            if replacement and _safe_float(row.get("age")) <= _safe_float(target.get("age")) + 4:
                score += 0.04
            if target_league and row.get("league") == target_league:
                score += 0.04
            if "creative" in reqs:
                score += 0.08 * _safe_float(row.get("assists_per90"))
            if "defensive" in reqs:
                if row.get("sub_position") in {"Defensive Midfield", "Central Midfield"}:
                    score += 0.08
            if "possession" in reqs:
                score += 0.03 * min(_safe_float(row.get("minutes_played_log")), 10) / 10
                if row.get("sub_position") in {"Central Midfield", "Defensive Midfield", "Attacking Midfield"}:
                    score += 0.03
            if "low budget" in reqs:
                score -= 0.04 * float(quality[pos])
            elif "high budget" in reqs or "chelsea" in q or "צ'לסי" in q:
                score += 0.04 * float(quality[pos])
            rows.append({
                "iloc": int(iloc),
                "score": score,
                "cosine": float(cosine_scores[pos]),
                "jaccard": jac,
                "same_cluster": bool(cluster_bonus),
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        top = rows[:limit]
        self.last_debug["retrieved_count"] = int(len(cand_ilocs))
        self.last_debug["top_candidates"] = [str(self.df.iloc[item["iloc"]]["player_name"]) for item in top]
        lines = [
            f"Target player: {self._player_line(target)}",
            f"Target profile cluster: {target.get(cluster_col)} ({target.get('position_archetype', target.get('archetype'))}).",
            f"Replacement constraints: {', '.join(reqs) if reqs else 'none'}; target team: {target_team or 'not specified'}.",
            "Recommended similar/replacement candidates:",
        ]
        for rank, item in enumerate(top, 1):
            row = self.df.iloc[item["iloc"]]
            extra = (
                f" | combined score {_pct_capped(item['score'])}, cosine {_pct_capped(item['cosine'])}, "
                f"Jaccard {_pct(item['jaccard'])}, same_cluster={item['same_cluster']}"
            )
            lines.append(f"{rank}. {self._player_line(row, extra)}")
        lines.append("Method: TF-IDF/Jaccard entity matching + position-aware K-Means profile filtering + cosine performance similarity + Jaccard trait similarity.")
        return "\n".join(lines), top, True

    find_similar_players = findSimilarPlayers

    def _filter_players(self, question: str, entities: list[EntityMatch]) -> tuple[pd.DataFrame, list[str]]:
        cand = self.df.copy()
        filters: list[str] = []
        q = normalize_text(question)
        current_question = any(w in q for w in ["current", "today", "now", "היום", "נוכחי", "עכשיו"])
        if current_question or any(e.entity_type in {"league", "club"} for e in entities):
            if "last_season" in cand.columns:
                cand = cand[pd.to_numeric(cand["last_season"], errors="coerce").fillna(0) >= 2025]
                filters.append("active_player_filter=last_season>=2025")
        league = next((e for e in entities if e.entity_type == "league"), None)
        if league and league.meta.get("dataset"):
            cand = cand[cand["league"] == league.meta["dataset"]]
            filters.append(f"league={league.name}")

        position = self._match_position(question)
        if position:
            cand = cand[cand["position"] == position]
            filters.append(f"position={position}")

        sub_positions = self._match_sub_positions(question)
        if sub_positions:
            cand = cand[cand["sub_position"].isin(sub_positions)]
            filters.append(f"role={'+'.join(sub_positions)}")

        for ent in entities:
            if ent.entity_type == "national_team":
                cand = cand[cand["nationality"] == ent.name]
                filters.append(f"nationality={ent.name}")
                break
            if ent.entity_type == "club":
                cand = cand[cand["club"] == ent.name]
                filters.append(f"club={ent.name}")
                break

        age_numbers = [int(n) for n in re.findall(r"\b(1[6-9]|2[0-9]|3[0-9])\b", q)]
        if any(w in q for w in ["young", "talent", "potential", "prospect", "צעיר", "כישרון", "פוטנציאל"]):
            limit = age_numbers[0] if age_numbers else 23
            cand = cand[(cand["age"] > 0) & (cand["age"] <= limit)]
            filters.append(f"age<={limit}")

        return cand, filters

    def _current_ranking_context(self, question: str, entities: list[EntityMatch], intent: str) -> str | None:
        league = next((e for e in entities if e.entity_type == "league"), None)
        club = next((e for e in entities if e.entity_type == "club"), None)
        if any(e.entity_type == "national_team" for e in entities):
            return None
        position = self._match_position(question)
        ranking_type = intent
        if intent == "top_scorer":
            position = position or "Attack"
        elif intent == "best_attacking_player":
            position = "Attack"
        reqs = self._extract_requirements(question)
        if "creative" in reqs and position == "Midfield":
            ranking_type = "creative_midfielder"
        elif "defensive" in reqs and position == "Midfield":
            ranking_type = "defensive_midfielder"

        ranked = rank_current_players(
            league=league.name if league else None,
            team=club.name if club else None,
            position_filter=position,
            ranking_type=ranking_type,
            limit=10,
        )
        if ranked.empty:
            return None

        self.last_debug["selected_data_source"] = "current_player_rankings"
        self.last_debug["selected_source_file"] = str(FOOTBALL_WORKBOOK_FILE).replace("\\", "/")
        self.last_debug["selected_sheet"] = CURRENT_STATS_SHEET
        self.last_debug["active_player_filter_applied"] = True
        self.last_debug["player_profiles_usage"] = "secondary context only"
        self.last_debug["retrieved_count"] = int(len(ranked))
        self.last_debug["top_candidates"] = ranked["Player_Name"].astype(str).head(10).tolist()
        self.last_debug["detected_league"] = league.name if league else None
        self.last_debug["detected_team"] = club.name if club else None

        filters = []
        if league:
            filters.append(f"league={league.name}")
        if club:
            filters.append(f"club={club.name}")
        if position:
            filters.append(f"position={position}")
        lines = [
            f"Current-season player ranking from {FOOTBALL_WORKBOOK_FILE} / sheet {CURRENT_STATS_SHEET}.",
            f"Filters: {', '.join(filters) if filters else 'none'}; active/current stats source used.",
            "Top candidates before Gemini:",
        ]
        for rank, (_, row) in enumerate(ranked.head(10).iterrows(), 1):
            score = round(float(row.get("current_rank_score", 0)) * 100, 1)
            lines.append(f"{rank}. {self._current_player_line(row, f' | current ranking score {score}')}")
        lines.append("Method: current-stat source selection + league/team/position filtering + weighted season ranking by goals, assists, shots on target, shots, and minutes. Player profiles are secondary context only.")
        return "\n".join(lines)

    def _score_best_players(self, cand: pd.DataFrame, potential: bool = False, question: str = "") -> pd.DataFrame:
        if cand.empty:
            return cand
        data = cand.copy()
        q = normalize_text(question)
        reqs = self._extract_requirements(question)
        mv = data["market_value_in_eur"].fillna(0).clip(lower=0)
        goals = data["goals"].fillna(0).clip(lower=0)
        assists = data["assists"].fillna(0).clip(lower=0)
        minutes = data["minutes_played"].fillna(0).clip(lower=0)
        goals90 = data["goals_per90"].fillna(0).clip(lower=0)
        assists90 = data["assists_per90"].fillna(0).clip(lower=0)
        ga90 = data["ga_per90"].fillna(0).clip(lower=0)
        age = data["age"].fillna(data["age"].median()).clip(lower=16, upper=45)

        def norm(series):
            max_val = series.max()
            return series / max_val if max_val and max_val > 0 else series * 0

        if any(w in q for w in ["striker", "forward", "goals", "scorer", "חלוץ", "חלוצים", "שערים", "כובש"]):
            data["_qa_score"] = (
                0.30 * norm(goals)
                + 0.24 * norm(goals90)
                + 0.18 * norm(np.log1p(mv))
                + 0.14 * norm(np.log1p(minutes))
                + 0.08 * norm(assists)
                + 0.06 * norm(ga90)
            )
        elif "creative" in reqs or any(w in q for w in ["creative", "assist", "playmaker", "מבשל", "יצירתי", "בישולים"]):
            data["_qa_score"] = (
                0.30 * norm(assists)
                + 0.24 * norm(assists90)
                + 0.18 * norm(np.log1p(mv))
                + 0.14 * norm(np.log1p(minutes))
                + 0.08 * norm(goals)
                + 0.06 * norm(ga90)
            )
        elif "defensive" in reqs:
            role_bonus = data["sub_position"].isin(["Defensive Midfield", "Central Midfield", "Centre-Back"]).astype(float)
            data["_qa_score"] = (
                0.24 * norm(np.log1p(minutes))
                + 0.24 * norm(np.log1p(mv))
                + 0.18 * role_bonus
                + 0.14 * norm(assists)
                + 0.10 * norm(goals)
                + 0.08 * norm(ga90)
            )
        else:
            data["_qa_score"] = (
                0.40 * norm(np.log1p(mv))
                + 0.22 * norm(ga90)
                + 0.14 * norm(np.log1p(minutes))
                + 0.12 * norm(goals)
                + 0.12 * norm(assists)
            )
        if potential:
            data["_qa_score"] += 0.22 * (1 - ((age - 16) / 29).clip(lower=0, upper=1))
        return data.sort_values("_qa_score", ascending=False)

    def _best_player_context(self, question: str, entities: list[EntityMatch], intent: str) -> str:
        current_context = self._current_ranking_context(question, entities, intent)
        if current_context:
            return current_context
        cand, filters = self._filter_players(question, entities)
        potential = intent == "league_analysis" and any(w in normalize_text(question) for w in ["young", "talent", "potential", "צעיר", "כישרון"])
        ranked = self._score_best_players(cand, potential=potential, question=question)
        if ranked.empty:
            self.last_debug["retrieved_count"] = 0
            self.last_debug["top_candidates"] = []
            return "No matching players were found in the dataset for the requested filters."
        self.last_debug["retrieved_count"] = int(len(ranked))
        self.last_debug["top_candidates"] = ranked.head(8)["player_name"].astype(str).tolist()
        title = "Best-player candidates from dataset"
        lines = [f"{title}. Filters: {', '.join(filters) if filters else 'none'}."]
        for rank, (_, row) in enumerate(ranked.head(8).iterrows(), 1):
            rating = round(float(row.get("_qa_score", 0)) * 100, 1)
            lines.append(f"{rank}. {self._player_line(row, f' | rating {rating}')}")
        lines.append("Method: TF-IDF/Jaccard intent/entity matching + weighted player rating using market value, goals, assists, per-90 output, minutes, and K-Means archetype context.")
        return "\n".join(lines)

    def _top_scorer_context(self, question: str, entities: list[EntityMatch]) -> str:
        current_context = self._current_ranking_context(question, entities, "top_scorer")
        if current_context:
            return current_context
        league = next((e for e in entities if e.entity_type == "league"), None)
        lines = []
        if league and league.meta.get("api"):
            scorers = self.api.get_scorers(league.meta["api"], limit=10)
            if scorers:
                lines.append(f"Fresh Football-Data API top scorers for {league.name}:")
                for rank, s in enumerate(scorers[:8], 1):
                    player = s.get("player", {}).get("name", "?")
                    team = s.get("team", {}).get("name", "?")
                    goals = s.get("goals", "?")
                    assists = s.get("assists", "?")
                    lines.append(f"{rank}. {player} | {team} | goals {goals}, assists {assists}")
                lines.append("Fresh API data was available and should be preferred for current-season top scorer questions.")

        cand, filters = self._filter_players(question, entities)
        if cand.empty:
            self.last_debug["retrieved_count"] = 0
            self.last_debug["top_candidates"] = []
            if lines:
                return "\n".join(lines)
            return "No scorer context found in the dataset."
        top = cand.sort_values(["goals", "ga_per90", "market_value_in_eur"], ascending=False).head(8)
        self.last_debug["retrieved_count"] = int(len(cand))
        self.last_debug["top_candidates"] = top["player_name"].astype(str).tolist()
        lines.append(f"Dataset scorer candidates. Filters: {', '.join(filters) if filters else 'none'}.")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(f"{rank}. {self._player_line(row)}")
        lines.append("Method: TF-IDF/Jaccard entity matching + scorer ranking by goals, per-90 output, and market-value tie-break.")
        return "\n".join(lines)

    def _club_strength(self, club: str) -> dict[str, Any]:
        sub = self.df[self.df["club"] == club].copy()
        if sub.empty:
            return {"name": club, "found": False}
        top = sub.nlargest(18, "market_value_in_eur")
        attack = top[top["position"] == "Attack"]
        return {
            "name": club,
            "found": True,
            "players": len(sub),
            "top18_value": float(top["market_value_in_eur"].fillna(0).sum()),
            "top18_mean": float(top["market_value_in_eur"].fillna(0).mean()),
            "goals": float(top["goals"].fillna(0).sum()),
            "assists": float(top["assists"].fillna(0).sum()),
            "ga90": float(top["ga_per90"].fillna(0).mean()),
            "attack_value": float(attack["market_value_in_eur"].fillna(0).sum()) if not attack.empty else 0.0,
            "depth": len(sub),
        }

    def _team_score(self, metrics: dict[str, Any]) -> float:
        if not metrics.get("found"):
            return 0.0
        return (
            0.45 * math.log1p(metrics["top18_value"])
            + 0.20 * math.log1p(metrics["attack_value"])
            + 0.15 * math.log1p(metrics["goals"])
            + 0.10 * math.log1p(metrics["assists"])
            + 0.10 * metrics["ga90"]
        )

    def _team_comparison_context(self, question: str, teams: list[EntityMatch], prediction: bool = False) -> str:
        if len(teams) < 2:
            self.last_debug["retrieved_count"] = len(teams)
            self.last_debug["top_candidates"] = [t.name for t in teams]
            return "I could not identify two teams clearly enough for comparison."
        a, b = teams[0], teams[1]

        if a.entity_type == "national_team" and b.entity_type == "national_team":
            known = set(self.national_strength.index)
            n1, n2 = normalize_nation(a.name, known), normalize_nation(b.name, known)
            if n1 and n2:
                r1, r2 = self.national_strength.loc[n1], self.national_strength.loc[n2]
                s1, s2 = float(r1["strength"]), float(r2["strength"])
                p1, draw, p2 = self._prediction_probs(s1, s2)
                winner = a.name if p1 > p2 and p1 > draw else b.name if p2 > p1 and p2 > draw else "Draw"
                score = self._scoreline(p1, draw, p2)
                return (
                    f"National-team comparison/prediction: {a.name} vs {b.name}\n"
                    f"{a.name}: strength {round(s1*100, 1)}, squad mean value EUR {int(r1['squad_value_mean']):,}, depth {int(r1['depth'])}\n"
                    f"{b.name}: strength {round(s2*100, 1)}, squad mean value EUR {int(r2['squad_value_mean']):,}, depth {int(r2['depth'])}\n"
                    f"Predicted probabilities: {a.name} {_pct(p1)}, draw {_pct(draw)}, {b.name} {_pct(p2)}. Suggested score: {score}. Likely edge: {winner}.\n"
                    "Method: squad-strength model from aggregated dataset player market values, softmax probabilities, and draw adjustment."
                )

        m1, m2 = self._club_strength(a.name), self._club_strength(b.name)
        self.last_debug["retrieved_count"] = 2
        self.last_debug["top_candidates"] = [a.name, b.name]
        s1, s2 = self._team_score(m1), self._team_score(m2)
        p1, draw, p2 = self._prediction_probs(s1, s2)
        score = self._scoreline(p1, draw, p2)
        return (
            f"Club comparison/prediction: {a.name} vs {b.name}\n"
            f"{a.name}: top18 value EUR {int(m1.get('top18_value', 0)):,}, squad goals {int(m1.get('goals', 0))}, "
            f"assists {int(m1.get('assists', 0))}, attack value EUR {int(m1.get('attack_value', 0)):,}, depth {m1.get('depth', 0)}\n"
            f"{b.name}: top18 value EUR {int(m2.get('top18_value', 0)):,}, squad goals {int(m2.get('goals', 0))}, "
            f"assists {int(m2.get('assists', 0))}, attack value EUR {int(m2.get('attack_value', 0)):,}, depth {m2.get('depth', 0)}\n"
            f"Model edge: {a.name if s1 > s2 else b.name if s2 > s1 else 'very close'}."
            + (f" Predicted probabilities: {a.name} {_pct(p1)}, draw {_pct(draw)}, {b.name} {_pct(p2)}. Suggested score: {score}." if prediction else "")
            + "\nMethod: TF-IDF/Jaccard team matching + dataset squad-strength model using top-player market value, attacking output, assists, and depth."
        )

    def _prediction_probs(self, s1: float, s2: float) -> tuple[float, float, float]:
        if s1 <= 1.5 and s2 <= 1.5:
            s1, s2 = s1 / 40.0, s2 / 40.0
        exp1, exp2 = math.exp(4.0 * s1), math.exp(4.0 * s2)
        raw1, raw2 = exp1 / (exp1 + exp2), exp2 / (exp1 + exp2)
        closeness = 1.0 - abs(s1 - s2) / max(abs(s1) + abs(s2), 1e-6)
        draw = min(max(0.18 + 0.14 * closeness, 0.16), 0.32)
        return raw1 * (1 - draw), draw, raw2 * (1 - draw)

    def _scoreline(self, p1: float, draw: float, p2: float) -> str:
        if draw >= max(p1, p2) - 0.03:
            return "1-1"
        if p1 > p2 + 0.18:
            return "2-0"
        if p2 > p1 + 0.18:
            return "0-2"
        return "2-1" if p1 > p2 else "1-2"

    def _league_analysis_context(self, question: str, entities: list[EntityMatch]) -> str:
        league = next((e for e in entities if e.entity_type == "league"), None)
        lines = []
        if league and league.meta.get("api"):
            table = self.api.get_standings(league.meta["api"])
            if table:
                best_attack = sorted(table, key=lambda r: r.get("goalsFor", 0), reverse=True)[:5]
                lines.append(f"Fresh Football-Data API attacking table for {league.name}:")
                for rank, row in enumerate(best_attack, 1):
                    team = row.get("team", {}).get("name", "?")
                    lines.append(f"{rank}. {team} | GF {row.get('goalsFor', 0)}, GA {row.get('goalsAgainst', 0)}, points {row.get('points', 0)}")

        cand = self.df
        if league and league.meta.get("dataset"):
            cand = cand[cand["league"] == league.meta["dataset"]]
        grouped = []
        for club, sub in cand.groupby("club"):
            if not isinstance(club, str) or len(sub) < 5:
                continue
            attack = sub[sub["position"] == "Attack"]
            if attack.empty:
                continue
            grouped.append({
                "club": club,
                "attack_goals": float(attack["goals"].fillna(0).sum()),
                "attack_assists": float(attack["assists"].fillna(0).sum()),
                "attack_value": float(attack["market_value_in_eur"].fillna(0).sum()),
                "attack_ga90": float(attack["ga_per90"].fillna(0).mean()),
            })
        if grouped:
            ranked = sorted(grouped, key=lambda r: (math.log1p(r["attack_value"]) + math.log1p(r["attack_goals"]) + r["attack_ga90"]), reverse=True)[:8]
            self.last_debug["retrieved_count"] = int(len(grouped))
            self.last_debug["top_candidates"] = [row["club"] for row in ranked]
            lines.append(f"Dataset attacking-strength candidates. League filter: {league.name if league else 'none'}.")
            for rank, row in enumerate(ranked, 1):
                lines.append(
                    f"{rank}. {row['club']} | attack goals {int(row['attack_goals'])}, assists {int(row['attack_assists'])}, "
                    f"attack value EUR {int(row['attack_value']):,}, avg GA/90 {round(row['attack_ga90'], 2)}"
                )
        lines.append("Method: Football-Data API when available + dataset team attacking aggregation from player goals, assists, GA/90, and market value.")
        return "\n".join(lines)

    def getRelevantContext(self, question: str, intent: IntentResult) -> tuple[str, list[EntityMatch], bool, list[str]]:
        self.last_debug = {
            "retrieved_count": 0,
            "top_candidates": [],
            "context_question": question,
            "selected_data_source": "player_profiles",
            "selected_source_file": self.selected_source_file,
            "selected_sheet": None,
            "active_player_filter_applied": False,
            "player_profiles_usage": "primary for similarity/replacement or fallback ranking",
        }
        entities: list[EntityMatch] = []
        methods = ["TF-IDF intent matching", "Jaccard token similarity"]
        clustering_used = False

        league = self._match_league(question)
        if league:
            entities.append(league)

        if intent.intent in {"player_similarity", "player_replacement"}:
            players = self.findBestEntityMatch(question, "player", 1)
            entities.extend(players)
            team_entities = [
                e for e in self.findBestEntityMatch(question, "team", 3)
                if e.entity_type in {"club", "national_team"} and e.score >= 0.80
            ]
            entities.extend([e for e in team_entities if not any(x.name == e.name and x.entity_type == e.entity_type for x in entities)])
            inherited_target_team = self.conversation.target_team if self._last_context_is_followup else None
            target_team = next((e.name for e in team_entities if e.entity_type == "club"), inherited_target_team)
            if players:
                context, _, clustering_used = self.findSimilarPlayers(
                    players[0].name,
                    replacement=intent.intent == "player_replacement",
                    question=question,
                    target_team=target_team,
                )
                methods.extend(["K-Means clustering", "Cosine performance similarity", "Jaccard trait similarity"])
                return context, entities, clustering_used, methods

        if intent.intent == "player_comparison":
            players = self.findBestEntityMatch(question, "player", 2)
            entities.extend(players)
            if len(players) >= 2:
                i1, i2 = int(players[0].meta["iloc"]), int(players[1].meta["iloc"])
                r1, r2 = self.df.iloc[i1], self.df.iloc[i2]
                jac = jaccard(self.engine.trait_set(i1), self.engine.trait_set(i2))
                cos = float(self.engine.cosine(i1, np.array([i2]))[0])
                context = (
                    f"Player comparison:\n"
                    f"1. {self._player_line(r1)}\n"
                    f"2. {self._player_line(r2)}\n"
                    f"Similarity: cosine performance {_pct(cos)}, Jaccard categorical traits {_pct(jac)}.\n"
                    "Method: TF-IDF/Jaccard player matching + cosine vector similarity + Jaccard trait-set comparison + K-Means archetypes."
                )
                return context, entities, True, methods + ["Cosine performance similarity", "K-Means clustering"]

        team_intents = {"team_comparison", "match_prediction"}
        if intent.intent in team_intents:
            teams = self._extract_match_teams(question)
            entities.extend(teams)
            context = self._team_comparison_context(question, teams, prediction=intent.intent == "match_prediction")
            return context, entities, False, methods + ["Squad-strength model", "Football-Data API if available"]

        if intent.intent == "top_scorer":
            has_league = any(e.entity_type == "league" for e in entities)
            allow_national_team = self._mentions_national_team(question)
            for ent in self.findBestEntityMatch(question, "team", 2):
                if ent.entity_type != "league" and ent.score < 0.70:
                    continue
                if ent.entity_type == "national_team" and has_league and not allow_national_team:
                    continue
                if not any(e.name == ent.name and e.entity_type == ent.entity_type for e in entities):
                    entities.append(ent)
            return self._top_scorer_context(question, entities), entities, False, methods + ["Football-Data API scorers if available"]

        if intent.intent in {"best_player", "best_attacking_player", "national_team_analysis"}:
            team_entities = self.findBestEntityMatch(question, "team", 2)
            has_league = any(e.entity_type == "league" for e in entities)
            allow_national_team = intent.intent == "national_team_analysis" or self._mentions_national_team(question)
            entities.extend([
                e for e in team_entities
                if e.entity_type != "league"
                and e.score >= 0.70
                and (e.entity_type != "national_team" or allow_national_team or not has_league)
            ])
            return self._best_player_context(question, entities, intent.intent), entities, False, methods + ["Weighted player rating", "K-Means archetype context"]

        if intent.intent == "league_analysis":
            if any(w in normalize_text(question) for w in ["young", "talent", "potential", "צעיר", "כישרון"]):
                return self._best_player_context(question, entities, intent.intent), entities, False, methods + ["Potential rating"]
            team_entities = self.findBestEntityMatch(question, "team", 2)
            entities.extend([e for e in team_entities if e.entity_type != "league" and e.score >= 0.70])
            return self._league_analysis_context(question, entities), entities, False, methods + ["Team attacking aggregation"]

        players = self.findBestEntityMatch(question, "player", 3)
        teams = self.findBestEntityMatch(question, "team", 3)
        entities.extend(players + teams)
        context_lines = ["General football question context from available data:"]
        for player in players:
            row = self.df.iloc[int(player.meta["iloc"])]
            context_lines.append(self._player_line(row))
        for team in teams:
            if team.entity_type == "club":
                context_lines.append(str(self._club_strength(team.name)))
            elif team.entity_type == "national_team" and team.name in self.national_strength.index:
                row = self.national_strength.loc[team.name]
                context_lines.append(f"{team.name}: strength {round(float(row['strength'])*100, 1)}, depth {int(row['depth'])}")
        context_lines.append("If context is insufficient, answer carefully and state the limitation.")
        return "\n".join(context_lines), entities, False, methods

    get_relevant_context = getRelevantContext

    def _fallback_answer(self, question: str, intent: IntentResult, context: str) -> str:
        if intent.intent == "unknown_question":
            return "I could not confidently connect this to the football dataset. Try mentioning a player, team, league, or match."
        return (
            f"Based on the available data, here is the relevant football context:\n\n"
            f"{context}\n\n"
            "Confidence: medium. The answer uses the project dataset; fresh live data is used only when Football-Data API is available."
        )

    def callGemini(self, question: str, context: str, intent: IntentResult, language: str) -> str:
        if not self.llms:
            return self._fallback_answer(question, intent, context)

        system = SystemMessage(content=(
            "You are a professional football analyst assistant. Answer football questions using "
            "the provided dataset context, statistics, clustering results, TF-IDF/Jaccard matches, "
            "and Football-Data API context. The context is the source of truth. Do not answer from "
            "general football memory when candidate rows or API rows are provided. If the data is "
            "incomplete, say so clearly and provide a careful football-based explanation. Do not "
            "invent exact statistics. Explain reasoning simply and clearly. Answer in the same language as the user. "
            "For replacements: short answer, ranked players, why each fits, similarity score, cluster/profile, stats, confidence. "
            "For best-player/striker questions: short answer, ranked list, stats used, confidence. "
            "For predictions: predicted score, key reasons, data used, confidence, prediction disclaimer. "
            "Always include the Method line(s) from the context when present."
        ))
        prompt_text = (
            f"User question: {question}\n"
            f"Detected intent: {intent.intent}\n"
            f"Intent confidence: TF-IDF={round(intent.tfidf_score, 3)}, "
            f"Jaccard={round(intent.jaccard_score, 3)}, combined={round(intent.combined_score, 3)}\n"
            f"User language: {language}\n\n"
            f"Available football context:\n{context}\n\n"
            "Write the final answer naturally as a football analyst. Use the candidate rows and scores. Do not dump raw data only."
        )
        self.last_debug["prompt"] = prompt_text
        human = HumanMessage(content=prompt_text)

        last_err = None
        for offset in range(len(self.llms)):
            idx = (self.model_idx + offset) % len(self.llms)
            try:
                resp = self.llms[idx].invoke([system, human])
                self.model_idx = idx
                return self._extract_text(resp.content)
            except Exception as e:
                last_err = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    continue
                break
        fallback = self._fallback_answer(question, intent, context)
        if last_err:
            fallback += f"\n\nGemini note: generation failed, so this is a data-context fallback."
        return fallback

    call_gemini = callGemini

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return str(content)

    def answerFootballQuestion(self, question: str, use_gemini: bool = True) -> QAResult:
        context_reset = self._should_reset_context(question)
        if context_reset:
            self.reset_context()
        base_intent = self.detectIntent(question)
        context_question, intent, context_notes = self._contextualize_question(question, base_intent)
        context, entities, clustering_used, methods = self.getRelevantContext(context_question, intent)
        entities = self._dedupe_entities(entities)
        self.last_debug["original_question"] = question
        self.last_debug["context_reset"] = context_reset
        self.last_debug["is_followup"] = self._last_context_is_followup
        self.last_debug["context_notes"] = context_notes
        self.last_debug["detected_entities"] = [f"{e.entity_type}:{e.name}:{e.score:.3f}" for e in entities]
        self.last_debug["detected_intent"] = intent.intent
        self.last_debug.setdefault("selected_data_source", self.selected_data_source)
        self.last_debug.setdefault("selected_source_file", self.selected_source_file)
        self.last_debug.setdefault("selected_sheet", None)
        self.last_debug.setdefault("active_player_filter_applied", False)
        self.last_debug.setdefault("player_profiles_usage", "primary for similarity/replacement or fallback ranking")
        self.last_debug["gemini_used"] = bool(use_gemini and self.llms)
        language = "Hebrew" if any("֐" <= ch <= "׿" for ch in question) else "English"
        question_for_prompt = question if context_question == question else f"{question}\nResolved context query: {context_question}"
        answer = self.callGemini(question_for_prompt, context, intent, language) if use_gemini else self._fallback_answer(question, intent, context)
        self.last_debug["final_answer_preview"] = answer[:500]
        self._update_context(context_question, intent, entities, answer)
        result = QAResult(
            question=question,
            intent=intent,
            entities=entities,
            context=context,
            answer=answer,
            clustering_used=clustering_used,
            methods=methods,
            retrieved_count=int(self.last_debug.get("retrieved_count", 0)),
            top_candidates=list(self.last_debug.get("top_candidates", [])),
            prompt=str(self.last_debug.get("prompt", "")),
            debug=dict(self.last_debug),
        )
        if self.debug_enabled:
            print(
                "[football_qa] "
                f"intent={intent.intent} tfidf={intent.tfidf_score:.3f} "
                f"jaccard={intent.jaccard_score:.3f} combined={intent.combined_score:.3f} "
                f"source={result.debug.get('selected_source_file')} "
                f"sheet={result.debug.get('selected_sheet')} "
                f"reset={result.debug.get('context_reset')} "
                f"entities={result.debug.get('detected_entities', [])} "
                f"active_filter={result.debug.get('active_player_filter_applied')} "
                f"profiles={result.debug.get('player_profiles_usage')} "
                f"rows={result.retrieved_count} clustering={clustering_used} "
                f"gemini={result.debug.get('gemini_used')} "
                f"top={result.top_candidates[:5]}",
                flush=True,
            )
        return result

    answer_football_question = answerFootballQuestion
