"""
scouting.py
Data-driven scouting engine for ScoutAI.

Architecture (ML course friendly):
    User question
      -> parse_scouting_query / LLM           (intent + entities)
      -> route_scouting_intent                (similar / replacement / profile / wonderkid)
      -> candidate filtering                   (position, age, potential, real-attribute pool)
      -> build_player_vector / build_target_profile
      -> calculate_weighted_similarity         (role-aware feature weights)
      -> cluster / archetype interpretation
      -> rank_candidates                       (multi-factor final score)
      -> generate_scouting_response            (structured candidate cards the LLM narrates)

The LLM only understands the question and phrases the answer; every ranking decision is
made here from the data (FC26 attributes + per-90 output + market value + K-Means archetype).
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# Scouting feature name -> column in the engine dataframe.
SCOUT_FEATURES = {
    "pace": "fc_pace", "shooting": "fc_shooting", "passing": "fc_passing",
    "dribbling": "fc_dribbling", "defending": "fc_defending", "physic": "fc_physic",
    "potential": "fc_potential", "overall": "fc_overall", "height": "height_in_cm",
    "goals_per90": "goals_per90", "assists_per90": "assists_per90",
    "minutes": "minutes_played", "market_value": "market_value_in_eur", "age": "age",
}
# Features that are log-scaled before normalising (heavy right tails).
_LOG_FEATURES = {"minutes", "market_value"}

# Role-aware feature weights (different roles are judged on different things).
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "striker":              {"shooting": .25, "goals_per90": .25, "physic": .15, "height": .10, "pace": .10, "dribbling": .05, "potential": .05, "age": .05},
    "creative_attacker":    {"dribbling": .25, "shooting": .20, "passing": .20, "potential": .15, "goals_per90": .10, "assists_per90": .10},
    "winger":               {"pace": .22, "dribbling": .22, "assists_per90": .15, "shooting": .13, "passing": .10, "goals_per90": .08, "potential": .05, "age": .05},
    "playmaker":            {"passing": .28, "dribbling": .18, "assists_per90": .15, "potential": .10, "defending": .09, "physic": .08, "minutes": .07, "age": .05},
    "box_to_box":           {"passing": .20, "defending": .18, "physic": .15, "dribbling": .12, "goals_per90": .10, "assists_per90": .10, "potential": .10, "age": .05},
    "defensive_midfielder": {"defending": .28, "physic": .18, "passing": .18, "minutes": .10, "potential": .10, "pace": .06, "assists_per90": .05, "age": .05},
    "centre_back":          {"defending": .30, "physic": .22, "height": .15, "pace": .10, "passing": .10, "potential": .08, "age": .05},
    "fullback":             {"pace": .22, "defending": .22, "physic": .15, "dribbling": .13, "passing": .13, "potential": .10, "age": .05},
    "left_back":            {"pace": .22, "defending": .22, "physic": .15, "dribbling": .13, "passing": .13, "potential": .10, "age": .05},
    "right_back":           {"pace": .22, "defending": .22, "physic": .15, "dribbling": .13, "passing": .13, "potential": .10, "age": .05},
    "goalkeeper":           {"overall": .50, "potential": .30, "height": .10, "age": .10},
    "default":              {"overall": .20, "potential": .15, "shooting": .12, "passing": .12, "dribbling": .12, "defending": .12, "pace": .10, "physic": .07},
}

# Archetype (position-aware K-Means label) -> role preset.
ARCHETYPE_ROLE = {
    "Finisher / Poacher": "striker", "Elite all-round forward": "striker",
    "Creative forward": "creative_attacker", "Pace & dribbling winger": "winger",
    "Playmaker / Creator": "playmaker", "Goal-scoring midfielder": "box_to_box",
    "Box-to-box midfielder": "box_to_box", "Defensive midfielder / Ball-winner": "defensive_midfielder",
    "Ball-playing defender": "centre_back", "Defensive stopper": "centre_back",
    "Aerial / physical defender": "centre_back", "Pace-reliant defender": "fullback",
    "Goalkeeper": "goalkeeper",
}
# Sub-position -> role preset (fallback when no archetype).
SUBPOS_ROLE = {
    "Centre-Forward": "striker", "Second Striker": "striker",
    "Right Winger": "winger", "Left Winger": "winger", "Right Midfield": "winger", "Left Midfield": "winger",
    "Attacking Midfield": "creative_attacker", "Central Midfield": "box_to_box",
    "Defensive Midfield": "defensive_midfielder", "Centre-Back": "centre_back",
    "Left-Back": "left_back", "Right-Back": "right_back", "Goalkeeper": "goalkeeper",
}
# Role -> (position group, allowed sub-positions) for candidate filtering.
ROLE_POSITIONS = {
    "striker": ("Attack", {"Centre-Forward", "Second Striker"}),
    "creative_attacker": ("Attack", {"Attacking Midfield", "Second Striker", "Centre-Forward", "Left Winger", "Right Winger"}),
    "winger": ("Attack", {"Left Winger", "Right Winger", "Left Midfield", "Right Midfield"}),
    "playmaker": ("Midfield", {"Attacking Midfield", "Central Midfield"}),
    "box_to_box": ("Midfield", {"Central Midfield", "Attacking Midfield", "Defensive Midfield"}),
    "defensive_midfielder": ("Midfield", {"Defensive Midfield", "Central Midfield"}),
    "centre_back": ("Defender", {"Centre-Back"}),
    "fullback": ("Defender", {"Left-Back", "Right-Back"}),   # both sides — use when side unknown
    "left_back":  ("Defender", {"Left-Back"}),               # explicit left side
    "right_back": ("Defender", {"Right-Back"}),              # explicit right side
    "goalkeeper": ("Goalkeeper", {"Goalkeeper"}),
}

# Free-text role / position keywords -> role.
ROLE_KEYWORDS = {
    "striker": ["striker", "finisher", "poacher", "number 9", "goalscorer", "goal scorer", "cf", "st", "centre-forward", "centre forward",
                "חלוץ", "חלוצים", "מבקיע", "מספר 9"],
    "winger": ["winger", "wing", "wide forward", "rw", "lw", "fast and creative winger",
               "קיצוני", "קיצונים", "כנף", "כנפיים"],
    "creative_attacker": ["creative attack", "creative forward", "attacking midfielder", "playmaker", "number 10", "cam", "creative attacking",
                          "תוקף יוצר", "קשר התקפי"],
    "playmaker": ["deep playmaker", "passer", "regista", "מסדר", "מחלק"],
    "defensive_midfielder": ["defensive midfield", "holding midfield", "ball winner", "ball-winner", "cdm", "dm", "destroyer", "physical defensive midfielder",
                             "קשר הגנתי", "קשר מגן"],
    "box_to_box": ["box to box", "box-to-box", "central midfield", "complete midfield",
                   "קשר מרכזי", "קשר"],
    "centre_back": ["centre back", "center back", "centre-back", "defender", "cb", "defensive wonderkid", "defensive player", "stopper",
                    "בלם", "בלמים", "מגן מרכזי"],
    "fullback":  ["full back", "full-back", "fullback", "wing back", "wing-back", "מגן צד"],
    "left_back":  ["left back", "left-back", "lb", "lwb", "left wing back", "מגן שמאל", "מגן שמאלי"],
    "right_back": ["right back", "right-back", "rb", "rwb", "right wing back", "מגן ימין", "מגן ימני"],
    "goalkeeper": ["goalkeeper", "keeper", "gk", "שוער", "שוערים"],
}
# Trait keywords -> important scouting features.
TRAIT_KEYWORDS = {
    "pace": ["fast", "quick", "pace", "speed", "rapid"],
    "dribbling": ["dribble", "dribbling", "skilful", "skillful", "technical", "1v1"],
    "shooting": ["scoring", "score", "goals", "finishing", "shooting", "clinical", "goal-scoring", "goal scoring"],
    "passing": ["passing", "playmaking", "creator", "creative", "vision", "chance creation", "ball progression", "progressive"],
    "assists_per90": ["assists", "assisting", "chance creation", "creator"],
    "defending": ["defensive", "defending", "tackling", "ball winner", "solid", "stopper"],
    "physic": ["physical", "strong", "strength", "powerful", "aggressive"],
    "potential": ["potential", "wonderkid", "prospect", "talent", "promising", "high ceiling"],
    "goals_per90": ["goal-scoring", "goal scoring", "prolific", "scoring ability"],
}

# ---- region / nationality / league filtering -------------------------------
# Transfermarkt league code -> the country whose domestic league it is. Decoded
# from the data (IT1=Lazio/Juve, GB1=Arsenal, ES1=Levante, L1=Dortmund, FR1=Monaco...).
LEAGUE_CODE_COUNTRY = {
    "IT1": "Italy", "GB1": "England", "ES1": "Spain", "L1": "Germany", "FR1": "France",
    "NL1": "Netherlands", "PO1": "Portugal", "BE1": "Belgium", "SC1": "Scotland",
    "TR1": "Turkey", "RU1": "Russia", "GR1": "Greece", "DK1": "Denmark", "A1": "Austria",
    "UKR1": "Ukraine", "BRA1": "Brazil", "ARG1": "Argentina", "MLS1": "United States",
    "MEX1": "Mexico", "COL1": "Colombia", "JAP1": "Japan", "KR1": "Korea, South",
    "SA1": "Saudi Arabia", "NO1": "Norway", "SE1": "Sweden", "PL1": "Poland",
    "RO1": "Romania", "AUS1": "Australia", "C1": "Switzerland", "TS1": "Czech Republic",
    "SER1": "Serbia",
}
LEAGUE_CODES = set(LEAGUE_CODE_COUNTRY)
# Inverse: country -> its domestic league code (for the nationality-OR-league filter).
COUNTRY_LEAGUE = {country: code for code, country in LEAGUE_CODE_COUNTRY.items()}
# Major European league codes — for "from a European league" style asks (Q5 demo).
EUROPEAN_LEAGUE_CODES = {"GB1", "ES1", "IT1", "L1", "FR1", "NL1", "PO1", "BE1", "SC1",
                         "TR1", "RU1", "GR1", "DK1", "A1", "UKR1", "SE1", "NO1", "PL1",
                         "RO1", "SER1", "TS1", "C1"}
EUROPE_TOKENS = {"europe", "european", "european league", "european leagues",
                 "top european league", "top european leagues", "top 5", "top five",
                 "big 5", "big five", "big-5", "top-5", "ליגה אירופאית", "אירופה"}

# Spoken league name (any language) -> code, for explicit "Premier League"/"Serie A" asks.
# When the user names a LEAGUE we filter by that league only (not nationality).
LEAGUE_NAME_TO_CODE = {
    "premier league": "GB1", "epl": "GB1", "english premier league": "GB1",
    "english league": "GB1", "la liga": "ES1", "laliga": "ES1", "primera division": "ES1",
    "serie a": "IT1", "bundesliga": "L1", "ligue 1": "FR1", "ligue1": "FR1",
    "eredivisie": "NL1", "primeira liga": "PO1", "liga portugal": "PO1",
    "scottish premiership": "SC1", "super lig": "TR1", "mls": "MLS1",
    "brasileirao": "BRA1", "brazilian league": "BRA1", "saudi pro league": "SA1",
    # Hebrew league names
    "הליגה האנגלית": "GB1", "פריימר ליג": "GB1", "פרמייר ליג": "GB1", "ליגה אנגלית": "GB1",
    "לה ליגה": "ES1", "הליגה הספרדית": "ES1", "סריה א": "IT1", "סדרה א": "IT1",
    "הליגה האיטלקית": "IT1", "בונדסליגה": "L1", "הליגה הגרמנית": "L1", "ליגה 1": "FR1",
    "הליגה הצרפתית": "FR1",
}
# Demonym / localized country name -> canonical nationality value in the data.
COUNTRY_ALIASES = {
    "italian": "Italy", "italia": "Italy", "איטליה": "Italy", "איטלקי": "Italy", "איטלקים": "Italy",
    "brazilian": "Brazil", "brasil": "Brazil", "ברזיל": "Brazil", "ברזילאי": "Brazil", "ברזילאים": "Brazil",
    "spanish": "Spain", "espana": "Spain", "españa": "Spain", "ספרד": "Spain", "ספרדי": "Spain", "ספרדים": "Spain",
    "german": "Germany", "deutschland": "Germany", "גרמניה": "Germany", "גרמני": "Germany", "גרמנים": "Germany",
    "english": "England", "אנגליה": "England", "אנגלי": "England", "אנגלים": "England",
    "french": "France", "צרפת": "France", "צרפתי": "France", "צרפתים": "France",
    "dutch": "Netherlands", "holland": "Netherlands", "הולנד": "Netherlands", "הולנדי": "Netherlands",
    "portuguese": "Portugal", "פורטוגל": "Portugal", "פורטוגלי": "Portugal",
    "argentine": "Argentina", "argentinian": "Argentina", "ארגנטינה": "Argentina", "ארגנטינאי": "Argentina",
    "belgian": "Belgium", "בלגיה": "Belgium",
    "croatian": "Croatia", "קרואטיה": "Croatia",
    "uruguayan": "Uruguay", "אורוגוואי": "Uruguay",
    "american": "United States", "usa": "United States",
}
# Country names the rule-based parser recognises in free text.
KNOWN_COUNTRIES = (set(COUNTRY_LEAGUE) | set(COUNTRY_ALIASES.values()) | {
    "Croatia", "Uruguay", "Nigeria", "Senegal", "Ghana", "Morocco", "Egypt",
    "Wales", "Ireland", "Switzerland", "Serbia", "Cote d'Ivoire", "Colombia",
})
# Coarse position group (data 'position' column) keywords -> group, used when no
# specific role is detected ("attacking player from Italy" -> Attack).
GROUP_KEYWORDS = {
    "Attack":     ["attack", "attacking", "attacker", "forward", "offensive", "front",
                   "התקפה", "התקפי", "התקפיים"],
    "Midfield":   ["midfield", "midfielder", "central mid", "אמצע"],
    "Defender":   ["defender", "defensive", "defence", "defense", "backline", "back line",
                   "הגנה", "הגנתי"],
    "Goalkeeper": ["goalkeeper", "keeper", "שוער"],
}


def normalize_country(name: str) -> str:
    """Map a demonym / localized / raw country string to a canonical nationality value."""
    if not name:
        return ""
    raw = name.strip()
    return COUNTRY_ALIASES.get(raw.lower(), raw)


def _minmax(series: pd.Series, log: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if log:
        s = np.log1p(s.clip(lower=0))
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


class ScoutingEngine:
    """Ranks players for similarity / replacement / profile / wonderkid searches."""

    def __init__(self, engine):
        self.engine = engine
        df = engine.df
        # Work on the real-attribute pool only (players with genuine FC26 attributes),
        # so weighted similarity uses real pace/shooting/passing/... not median fill-ins.
        self.pool = df[df["fc_overall"].notna()].copy()
        # Normalised (0-1) named feature frame over the pool.
        self.N = pd.DataFrame(index=self.pool.index)
        for name, col in SCOUT_FEATURES.items():
            self.N[name] = _minmax(self.pool[col], log=name in _LOG_FEATURES)
        # Data-reliability = how much we can trust the per-90 numbers (minutes played).
        self.N["reliability"] = _minmax(self.pool["minutes_played"], log=True)

    # ---- lookup -------------------------------------------------------------
    def resolve(self, name: str):
        """Resolve a (possibly misspelled) player name to a pool row index."""
        idx = self.engine.find_index(name)
        if idx is None:
            return None
        pid = self.engine.df.iloc[idx]["player_id"]
        hit = self.pool.index[self.pool["player_id"] == pid]
        return int(hit[0]) if len(hit) else None

    def role_of(self, idx: int) -> str:
        row = self.pool.loc[idx]
        sub_role = SUBPOS_ROLE.get(row.get("sub_position"))
        # Side-specific positions (Left-Back / Right-Back) take priority over the K-Means
        # archetype, which doesn't distinguish sides — a "Ball-playing defender" could be
        # a LCB or a true LB. The sub_position column is authoritative for lateral roles.
        if sub_role in ("left_back", "right_back"):
            return sub_role
        return ARCHETYPE_ROLE.get(row.get("archetype")) or sub_role or "default"

    # ---- vectors & target profiles -----------------------------------------
    def build_player_vector(self, idx: int) -> dict:
        """Normalized (0-1) named feature vector for one player."""
        return self.N.loc[idx].to_dict()

    def build_target_profile(self, important_features=None, role: str = "") -> tuple[dict, dict]:
        """Ideal target vector (max on the desired traits) plus role-aware weights."""
        preset = WEIGHT_PRESETS.get(role, WEIGHT_PRESETS["default"])
        important = [f for f in (important_features or []) if f in self.N.columns]
        if not important:
            important = [f for f in preset if f in self.N.columns]
        weights = {f: preset.get(f, 1.0 / len(important)) for f in important}
        target = {f: 1.0 for f in important}
        return target, weights

    # ---- core ranking -------------------------------------------------------
    def calculate_weighted_similarity(self, target: dict, weights: dict, cand_idx) -> np.ndarray:
        feats = [f for f in weights if f in self.N.columns]
        w = np.array([weights[f] for f in feats], dtype=float)
        w = w / w.sum() if w.sum() > 0 else w
        C = self.N.loc[cand_idx, feats].fillna(0.5).values
        t = np.array([target.get(f, 0.5) for f in feats])
        dist = np.sqrt((((C - t) ** 2) * w).sum(axis=1))
        return 1.0 / (1.0 + dist)

    def _age_fit(self, cand_idx, mode: str, ref_age: float | None = None, age_max: int | None = None) -> np.ndarray:
        ages = pd.to_numeric(self.pool.loc[cand_idx, "age"], errors="coerce").fillna(27).values
        if mode == "young":
            hi = age_max or 23
            return np.clip((hi + 2 - ages) / 8.0, 0, 1)          # younger -> higher
        if mode == "ref" and ref_age is not None:
            return np.clip(1.0 - np.maximum(ages - ref_age, 0) / 6.0, 0, 1)  # not older than ref
        return np.ones(len(cand_idx))

    def rank_candidates(self, cand_idx, sim, composite, age_ctx, limit=5):
        pot = self.N.loc[cand_idx, "potential"].fillna(self.N["potential"].median()).values
        cur = self.N.loc[cand_idx, "overall"].fillna(self.N["overall"].median()).values
        rel = self.N.loc[cand_idx, "reliability"].fillna(0.0).values
        agefit = self._age_fit(cand_idx, **age_ctx)
        final = (composite["sim"] * sim + composite["pot"] * pot + composite["cur"] * cur
                 + composite["age"] * agefit + composite["rel"] * rel)
        order = np.argsort(final)[::-1][:limit]
        out = []
        cand_idx = np.asarray(cand_idx)
        for o in order:
            out.append({
                "idx": int(cand_idx[o]), "score": float(final[o]),
                "similarity": float(sim[o]), "potential_score": float(pot[o]),
                "current_score": float(cur[o]), "age_fit": float(agefit[o]),
                "reliability": float(rel[o]),
            })
        return out

    # ---- explanation --------------------------------------------------------
    def _describe(self, idx: int, role: str, reference_idx: int | None = None,
                  important: list[str] | None = None) -> tuple[list[str], list[str]]:
        row = self.pool.loc[idx]
        focus = important or list(WEIGHT_PRESETS.get(role, WEIGHT_PRESETS["default"]).keys())
        # Goalkeepers: outfield attributes are meaningless — show overall/potential/physic instead.
        if role == "goalkeeper":
            focus = [f for f in focus if f in ("overall", "potential", "physic")]
        else:
            focus = [f for f in focus if f in ("pace", "shooting", "passing", "dribbling", "defending", "physic")]
        raw = {f: row.get(SCOUT_FEATURES[f]) for f in focus}
        raw = {f: v for f, v in raw.items() if pd.notna(v)}
        strengths = sorted(raw, key=lambda f: raw[f], reverse=True)[:3]
        strengths = [f"{f} {int(raw[f])}" for f in strengths if raw[f] >= 70]
        if reference_idx is not None:
            ref = self.pool.loc[reference_idx]
            weak = [f"{f} {int(row.get(SCOUT_FEATURES[f]))}" for f in raw
                    if pd.notna(ref.get(SCOUT_FEATURES[f])) and row.get(SCOUT_FEATURES[f]) <= ref.get(SCOUT_FEATURES[f]) - 8]
        else:
            weak = [f"{f} {int(raw[f])}" for f in raw if raw[f] < 60]
        return strengths or ["balanced profile"], weak[:2]

    def _card(self, item: dict, role: str, reference_idx=None, important=None) -> dict:
        row = self.pool.loc[item["idx"]]
        strengths, weaknesses = self._describe(item["idx"], role, reference_idx, important)
        return {
            "player_name": row.get("player_name"), "age": int(row.get("age") or 0),
            "position": row.get("sub_position") or row.get("position"),
            "club": row.get("club"), "nationality": row.get("nationality"),
            "league": row.get("league"), "player_id": row.get("player_id"),
            "overall": int(row["fc_overall"]) if pd.notna(row.get("fc_overall")) else None,
            "potential": int(row["fc_potential"]) if pd.notna(row.get("fc_potential")) else None,
            "archetype": row.get("archetype"),
            "fit": round(item["score"] * 100, 1), "similarity": round(item["similarity"] * 100, 1),
            "strengths": strengths, "weaknesses": weaknesses,
        }

    # ---- search types -------------------------------------------------------
    def find_similar_player(self, name: str, limit: int = 5):
        idx = self.resolve(name)
        if idx is None:
            return None, f"Could not find a player matching '{name}' with attribute data."
        role = self.role_of(idx)
        weights = WEIGHT_PRESETS.get(role, WEIGHT_PRESETS["default"])
        group, _ = ROLE_POSITIONS.get(role, (self.pool.loc[idx, "position"], None))
        mask = (self.pool["position"] == group) & (self.pool.index != idx)
        cand = self.pool.index[mask]
        target = self.build_player_vector(idx)
        sim = self.calculate_weighted_similarity(target, weights, cand)
        composite = {"sim": .60, "pot": .10, "cur": .15, "age": .00, "rel": .15}
        ranked = self.rank_candidates(cand, sim, composite, {"mode": "neutral"}, limit)
        cards = [self._card(it, role, reference_idx=idx) for it in ranked]
        return {"type": "similar", "reference": self.pool.loc[idx, "player_name"], "role": role,
                "reference_archetype": self.pool.loc[idx, "archetype"], "candidates": cards}, None

    def find_replacement(self, name: str, club: str = "", max_age: int = 0, limit: int = 5,
                          role_override: str = "", country: str = ""):
        idx = self.resolve(name)
        if idx is None:
            return None, f"Could not find a player matching '{name}' with attribute data."
        role = (role_override if role_override and role_override in ROLE_POSITIONS
                else self.role_of(idx))
        ref_age = float(self.pool.loc[idx, "age"] or 27)
        weights = WEIGHT_PRESETS.get(role, WEIGHT_PRESETS["default"])
        group, subs = ROLE_POSITIONS.get(role, (self.pool.loc[idx, "position"], None))
        mask = (self.pool["position"] == group) & (self.pool.index != idx)
        if subs:
            mask &= self.pool["sub_position"].isin(subs)
        if club:
            mask &= ~self.pool["club"].fillna("").str.contains(club, case=False, na=False)
        if max_age:
            mask &= pd.to_numeric(self.pool["age"], errors="coerce").fillna(99) <= max_age
        region = self._region_mask(country)
        if region is not None:
            mask &= region
        cand = self.pool.index[mask]
        if len(cand) == 0:
            return None, f"No replacement candidates found for {self.pool.loc[idx, 'player_name']}."
        target = self.build_player_vector(idx)
        sim = self.calculate_weighted_similarity(target, weights, cand)
        composite = {"sim": .45, "pot": .20, "cur": .15, "age": .10, "rel": .10}
        ranked = self.rank_candidates(cand, sim, composite, {"mode": "ref", "ref_age": ref_age}, limit)
        cards = [self._card(it, role, reference_idx=idx) for it in ranked]
        return {"type": "replacement", "reference": self.pool.loc[idx, "player_name"], "role": role,
                "club": club or None, "candidates": cards}, None

    def search_by_profile(self, role: str = "", positions=None, age_max: int = 0,
                          potential_min: int = 0, important_features=None,
                          country: str = "", position_group: str = "", limit: int = 5):
        target, weights = self.build_target_profile(important_features, role)
        important = list(weights.keys())
        mask = self._position_mask(role, positions)
        # Coarse position group ("Attack"/"Midfield"/"Defender"/"Goalkeeper") when the
        # query named a line but not a specific role ("attacking player from Italy").
        if not role and not positions and position_group:
            mask &= self.pool["position"] == position_group
        if age_max:
            mask &= pd.to_numeric(self.pool["age"], errors="coerce").fillna(99) <= age_max
        if potential_min:
            mask &= pd.to_numeric(self.pool["fc_potential"], errors="coerce").fillna(0) >= potential_min
        region = self._region_mask(country)
        if region is not None:
            mask &= region
        cand = self.pool.index[mask]
        if len(cand) == 0:
            return None, f"No players matched those filters{(' for ' + country) if country else ''}."
        sim = self.calculate_weighted_similarity(target, weights, cand)
        composite = {"sim": .50, "pot": .20, "cur": .15, "age": .10, "rel": .05}
        age_ctx = {"mode": "young", "age_max": age_max} if age_max else {"mode": "neutral"}
        ranked = self.rank_candidates(cand, sim, composite, age_ctx, limit)
        cards = [self._card(it, role or "default", important=important) for it in ranked]
        return {"type": "profile", "role": role or position_group or "custom profile",
                "important_features": important,
                "filters": {"age_max": age_max or None, "potential_min": potential_min or None,
                            "country": country or None},
                "candidates": cards}, None

    def find_wonderkids(self, role: str = "", positions=None, age_max: int = 21,
                        potential_min: int = 80, important_features=None, limit: int = 5,
                        max_overall: int = 0, country: str = "", position_group: str = ""):
        target, weights = self.build_target_profile(important_features, role)
        important = list(weights.keys())
        mask = self._position_mask(role, positions)
        if not role and not positions and position_group:
            mask &= self.pool["position"] == position_group
        mask &= pd.to_numeric(self.pool["age"], errors="coerce").fillna(99) <= (age_max or 21)
        mask &= pd.to_numeric(self.pool["fc_potential"], errors="coerce").fillna(0) >= (potential_min or 80)
        if max_overall:
            mask &= pd.to_numeric(self.pool["fc_overall"], errors="coerce").fillna(99) <= max_overall
        region = self._region_mask(country)
        if region is not None:
            mask &= region
        cand = self.pool.index[mask]
        if len(cand) == 0:
            return None, "No wonderkids matched those filters (try relaxing age or potential)."
        sim = self.calculate_weighted_similarity(target, weights, cand)
        composite = {"sim": .25, "pot": .40, "cur": .15, "age": .15, "rel": .05}   # potential-led
        ranked = self.rank_candidates(cand, sim, composite, {"mode": "young", "age_max": age_max}, limit)
        cards = [self._card(it, role or "default", important=important) for it in ranked]
        return {"type": "wonderkid", "role": role or position_group or "any position",
                "important_features": important,
                "filters": {"age_max": age_max, "potential_min": potential_min,
                            "max_overall": max_overall or None, "country": country or None},
                "candidates": cards}, None

    def _position_mask(self, role: str, positions) -> pd.Series:
        if role and role in ROLE_POSITIONS:
            group, subs = ROLE_POSITIONS[role]
            m = self.pool["position"] == group
            return m & self.pool["sub_position"].isin(subs) if subs else m
        if positions:
            subs = _positions_to_subpositions(positions)
            if subs:
                return self.pool["sub_position"].isin(subs)
        return pd.Series(True, index=self.pool.index)

    def _region_mask(self, country: str = ""):
        """Hard filter for a country/league constraint, applied to the candidate pool
        BEFORE ranking. Returns None when there is nothing to filter on.

        - An explicit LEAGUE ("Premier League", "Serie A", or a raw code like "IT1")
          filters by that league only.
        - A COUNTRY / demonym ("Italy", "Italian", "Brazil") matches players of that
          NATIONALITY *or* players in that country's domestic league — so "from Italy"
          returns Italians anywhere plus anyone in Serie A.
        """
        raw = (country or "").strip()
        if not raw:
            return None
        low = raw.lower()
        league = self.pool["league"].fillna("")
        # "Europe" / "European league" -> the major European leagues (not one country).
        if low in EUROPE_TOKENS:
            return league.isin(EUROPEAN_LEAGUE_CODES)
        # Explicit spoken league name -> that league only.
        if low in LEAGUE_NAME_TO_CODE:
            return league == LEAGUE_NAME_TO_CODE[low]
        # Raw league code (e.g. "IT1") -> that league only.
        if raw.upper() in LEAGUE_CODES:
            return league == raw.upper()
        # Country / demonym -> nationality OR that country's domestic league.
        nat = normalize_country(raw)
        nat_mask = self.pool["nationality"].fillna("").str.strip().str.lower() == nat.lower()
        code = COUNTRY_LEAGUE.get(nat)
        return (nat_mask | (league == code)) if code else nat_mask


# ---- query parsing & response formatting (reusable, LLM-independent) --------
_FIFA_POS = {
    "st": "Centre-Forward", "cf": "Centre-Forward", "ss": "Second Striker",
    "rw": "Right Winger", "lw": "Left Winger", "rm": "Right Midfield", "lm": "Left Midfield",
    "cam": "Attacking Midfield", "cm": "Central Midfield", "cdm": "Defensive Midfield",
    "cb": "Centre-Back", "lb": "Left-Back", "rb": "Right-Back", "gk": "Goalkeeper",
}


_POSITION_TO_ROLE: dict[str, str] = {
    "left-back": "left_back",  "left back": "left_back",  "lb": "left_back",
    "lwb": "left_back",        "left wing back": "left_back",
    "right-back": "right_back","right back": "right_back","rb": "right_back",
    "rwb": "right_back",       "right wing back": "right_back",
    "full back": "fullback",   "fullback": "fullback",    "wing back": "fullback",
    "wing-back": "fullback",
    "centre-back": "centre_back", "center-back": "centre_back", "centre back": "centre_back",
    "center back": "centre_back", "central defender": "centre_back", "cb": "centre_back",
    "central midfield": "box_to_box", "cm": "box_to_box", "box to box": "box_to_box",
    "defensive midfield": "defensive_midfielder", "cdm": "defensive_midfielder",
    "holding mid": "defensive_midfielder", "holding midfielder": "defensive_midfielder",
    "attacking midfield": "creative_attacker", "cam": "creative_attacker",
    "attacking mid": "creative_attacker", "number 10": "creative_attacker", "no 10": "creative_attacker",
    "striker": "striker", "centre forward": "striker", "center forward": "striker",
    "st": "striker", "cf": "striker", "number 9": "striker",
    "winger": "winger", "lw": "winger", "rw": "winger", "wide forward": "winger",
    "goalkeeper": "goalkeeper", "gk": "goalkeeper", "keeper": "goalkeeper",
    "playmaker": "playmaker",
}


def normalize_role(role_str: str) -> str:
    """Map a human-readable position/role string to the internal ROLE_POSITIONS key."""
    if not role_str:
        return ""
    key = role_str.strip().lower()
    if key in _POSITION_TO_ROLE:
        return _POSITION_TO_ROLE[key]
    if key in ROLE_POSITIONS:
        return key
    return ""


def _positions_to_subpositions(positions) -> set[str]:
    subs = set()
    for p in positions:
        key = str(p).strip().lower()
        if key in _FIFA_POS:
            subs.add(_FIFA_POS[key])
    return subs


def _kw_match(q: str, words: list[str]) -> bool:
    """Keyword match: short position codes ("st", "cb") need word boundaries so they don't
    match inside other words ("st" in "fast"); longer phrases match as substrings."""
    for w in words:
        if len(w) <= 3:
            if re.search(rf"\b{re.escape(w)}\b", q):
                return True
        elif w in q:
            return True
    return False


def parse_scouting_query(text: str) -> dict:
    """Rule-based fallback parser (used when the LLM is unavailable, and for the demo).

    Returns intent + entities so the same ScoutingEngine call can be made with or
    without the LLM.
    """
    q = text.lower().strip()
    ctx: dict = {"intent": "profile_search", "reference_player": None, "role": "",
                 "positions": [], "age_max": 0, "potential_min": 0, "important_features": [],
                 "country": "", "position_group": "", "limit": 5}

    # intent
    if re.search(r"replace|replacement|alternative|successor|תחליף|מחליף", q):
        ctx["intent"] = "replacement"
    elif re.search(r"similar to|similar|like|plays like|דומה|כמו", q):
        ctx["intent"] = "similar"
    elif re.search(r"wonderkid|wonder kid|young.*(prospect|talent|potential)|prospect|כשרון צעיר|עתודה", q):
        ctx["intent"] = "wonderkid"

    # reference player — English patterns then Hebrew patterns
    m = re.search(r"(?:replace(?:ment for)?|similar to|like|plays like|for)\s+([a-zà-ÿ.\-' ]{3,40})", text, re.IGNORECASE)
    if m:
        ref = re.split(r"\b(at|in|for|from|with)\b", m.group(1).strip(), 1)[0].strip(" ?.,")
        ctx["reference_player"] = ref or None
    if ctx["reference_player"] is None:
        # Hebrew: "מחליף ל<player>" / "דומה ל<player>" / "כמו <player>"
        m_he = re.search(
            r"(?:מחליף|תחליף|דומה|דומים)\s+ל(.{2,40}?)(?:\s*$|\s*\?|\s+ב|\s+עבור|\s+מ)",
            text,
        )
        if not m_he:
            m_he = re.search(r"כמו\s+(.{2,40}?)(?:\s*$|\s*\?|\s+ב)", text)
        if m_he:
            ctx["reference_player"] = m_he.group(1).strip(" ?.,") or None
    mclub = re.search(r"\bat\s+([A-Za-zÀ-ÿ.\-' ]{3,30})", text)
    if mclub:
        ctx["club"] = mclub.group(1).strip(" ?.,")

    # role
    for role, words in ROLE_KEYWORDS.items():
        if _kw_match(q, words):
            ctx["role"] = role
            break

    # requested result count ("top 3 strikers", "best 5 players")
    cnt = re.search(r"\b(?:top|best|הכי טוב(?:ים)?|ה-?)\s*(\d+)\b|\b(\d+)\s+(?:best|top|player|striker|forward|defender|goalkeeper)", q)
    if cnt:
        ctx["limit"] = int(cnt.group(1) or cnt.group(2))

    # age / potential / wonderkid defaults
    am = re.search(r"(?:under|below|younger than|max age|age)\s*(\d{2})", q) or re.search(r"u(\d{2})\b", q)
    if am:
        ctx["age_max"] = int(am.group(1))
    elif re.search(r"young|wonderkid|prospect|teenager|צעיר", q):
        ctx["age_max"] = 21 if ctx["intent"] == "wonderkid" else 23
    pm = re.search(r"potential\s*(?:>=|of|above|over)?\s*(\d{2})", q)
    if pm:
        ctx["potential_min"] = int(pm.group(1))
    elif ctx["intent"] == "wonderkid" or "high potential" in q:
        ctx["potential_min"] = 82 if ctx["intent"] == "wonderkid" else 80

    # important features from traits
    feats: list[str] = []
    for feat, words in TRAIT_KEYWORDS.items():
        if _kw_match(q, words):
            feats.append(feat)
    ctx["important_features"] = list(dict.fromkeys(feats))

    # country / league constraint ("from Italy", "Brazilian", "in the Premier League")
    ctx["country"] = _detect_country(text, q)

    # coarse position group when no specific role was detected ("attacking player")
    if not ctx["role"]:
        for grp, words in GROUP_KEYWORDS.items():
            if _kw_match(q, words):
                ctx["position_group"] = grp
                break
    return ctx


def _detect_country(text: str, q: str) -> str:
    """Detect a country or league mention in any language; returns a country name,
    a spoken league name, or '' (the engine's _region_mask resolves it)."""
    # 0. "European league" / "ליגה אירופאית" -> the major European leagues.
    for tok in EUROPE_TOKENS:
        if tok in q or tok in text:
            return "Europe"
    # 1. Explicit spoken league name (English in q, Hebrew in original text).
    for name in LEAGUE_NAME_TO_CODE:
        if name in q or name in text:
            return name
    # 2. Demonym / localized country name (skip <3-char aliases to avoid false hits).
    for alias, canon in COUNTRY_ALIASES.items():
        if len(alias) < 3:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q) or alias in text:
            return canon
    # 3. Bare country name ("from Italy", "Spain players").
    for c in KNOWN_COUNTRIES:
        if re.search(rf"\b{re.escape(c.lower())}\b", q):
            return c
    return ""


def generate_scouting_response(result: dict) -> str:
    """Render the engine result as structured scouting cards (the LLM then narrates)."""
    if not result or not result.get("candidates"):
        return "No suitable candidates were found."
    t = result["type"]
    _flt = result.get("filters", {})
    _ovr_str = (f", OVR ≤ {_flt['max_overall']}" if _flt.get("max_overall") else "")
    _country = _flt.get("country")
    _from_str = f" from {_country}" if _country else ""
    # Profile header: use a ranking-style label for named positions ("top goalkeepers"),
    # and the generic "best matches" label only for free-text custom profiles.
    _role = result.get("role") or ""
    _traits = ", ".join(result.get("important_features", []))
    if t == "profile" and _role in ROLE_POSITIONS:
        _profile_head = (f"Top {_role.replace('_', '-')}s{_from_str} ranked by {_traits}:"
                         if _traits else f"Top-rated {_role.replace('_', '-')}s{_from_str}:")
    else:
        _profile_head = f"Best matches for profile '{_role}'{_from_str} (key traits: {_traits}):"
    head = {
        "similar": f"Players most similar to {result.get('reference')} "
                   f"(role: {_role}, archetype: {result.get('reference_archetype')}):",
        "replacement": f"Replacement options for {result.get('reference')}"
                       + (f" (excluding {result.get('club')})" if result.get("club") else "") + ":",
        "profile": _profile_head,
        "wonderkid": (f"Top wonderkids ({_role}{_from_str}, "
                      f"age ≤ {_flt.get('age_max')}, "
                      f"potential ≥ {_flt.get('potential_min')}"
                      f"{_ovr_str}):"),
    }.get(t, "Scouting results:")
    # For position-ranking queries ("best goalkeeper?") show OVR/score; for profile matching
    # show fit/similarity. The distinction: profile type with a known ROLE_POSITIONS role
    # is a ranking, not a profile-matching search.
    _is_ranking = (t == "profile" and _role in ROLE_POSITIONS)
    lines = [f"**{head}**\n"]
    for i, c in enumerate(result["candidates"], 1):
        ovr = f"OVR {c['overall']}" if c["overall"] else "OVR n/a"
        pot = f"POT {c['potential']}" if c["potential"] else "POT n/a"
        if _is_ranking:
            score_str = f"{ovr} / {pot}"
        else:
            score_str = f"{ovr} / {pot} | fit {c['fit']}% (similarity {c['similarity']}%)"
        lines.append(
            f"{i}. **{c['player_name']}** — {c['position']} | {c['club']} | {c['nationality']} | age {c['age']}\n"
            f"   {score_str} | archetype: {c['archetype']}\n"
            f"   strengths: {', '.join(c['strengths'])}"
            + (f" | caveats: {', '.join(c['weaknesses'])}" if c["weaknesses"] else "")
        )
    method = {
        "similar": "Role-weighted similarity (weighted Euclidean on FC26 + per-90 features) within position group.",
        "replacement": "Role-weighted similarity + multi-factor score (potential, current ability, age fit, data reliability), same role, optional club exclusion.",
        "profile": "Target-profile vector + role-weighted similarity + multi-factor ranking, filtered by position/age/potential.",
        "wonderkid": "Age/potential filter + role-weighted profile similarity, potential-led multi-factor ranking.",
    }.get(t, "Weighted similarity ranking.")
    lines.append(f"\n🔍 Method: {method}")
    from viz import embed_viz
    title = {
        "similar":     f"Most similar to {result.get('reference')}",
        "replacement": f"Replacements for {result.get('reference')}",
        "profile":     (f"Top {_role.replace('_', '-')}s" if _role in ROLE_POSITIONS else f"Profile matches — {_role}"),
        "wonderkid":   f"Top wonderkids — {result.get('role')}",
    }.get(t, "Scouting results")
    items = [{
        "name": c["player_name"],
        "pct": float(c["fit"]),  # always composite fit so card order matches text order
        "pos": c["position"],
        "sub": c.get("club") or "",
        "tags": (c.get("strengths") or [])[:3],
    } for c in result["candidates"][:5]]
    viz = {"type": "similarity", "title": title, "metric": "fit", "items": items}
    return embed_viz("\n".join(lines), viz)
