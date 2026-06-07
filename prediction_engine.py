"""
prediction_engine.py
Data-driven football prediction engine for ScoutAI.

Pipeline (for every query type):
  User question
  → parse_prediction_query / LLM        (intent + entities)
  → route_prediction_intent              (match / top-scorer / player-goals)
  → resolve_home_away                    (explicit / fixture-lookup / assumption)
  → build_match_features                 (form, Elo, strength, odds-implied)
  → predict_match_result                 (RandomForestClassifier H/D/A)
  → predict_expected_goals               (Poisson regression)
  → classify_match_profile               (balanced / one-sided / open / low-scoring)
  → select_context_aware_scoreline       (context-weighted Poisson distribution)
  → calculate_prediction_confidence      (probability gap + data quality)
  → generate_prediction_response         (structured cards the LLM narrates)

Top scorer / player goals:
  → build_player_season_features         (FC26 + per-90 + team context)
  → predict_top_scorer / predict_player_goals (RandomForestRegressor)
"""

from __future__ import annotations

import re
import math
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────── constants ───────────────────────────────────────
CLUB_MATCHES_CSV     = "data/club_matches.csv"
NATIONAL_MATCHES_CSV = "data/national_matches.csv"
PLAYERS_CSV          = "data/players_clean.csv"
FBREF_CSV            = "data/players_data-2025_2026.csv"

ELO_BASE    = 1500.0
ELO_K       = 40.0
FORM_N      = 6          # last N matches for form
MIN_MATCHES = 5          # minimum team history to trust features

# attack-position sub-positions used for top-scorer candidates
ATTACKER_SUBS = {"Centre-Forward", "Second Striker", "Right Winger", "Left Winger",
                 "Attacking Midfield", "Right Midfield", "Left Midfield"}

# ─────────────────────────── name normalisation ───────────────────────────────
_SUFFIXES = [
    r"\bfootball club\b", r"\bfc\b", r"\bf\.c\.\b", r"\bass?oc(?:iation)?\b",
    r"\bsport(?:ing)?\b(?: club)?\b", r"\bclub\b", r"\bathletic\b", r"\bathlético\b",
    r"\batlético\b", r"\bunited\b", r"\bcity\b",
]

def _strip(name: str) -> str:
    """Lowercase + remove common club suffixes for fuzzy matching."""
    s = str(name).lower().strip()
    for pat in _SUFFIXES:
        s = re.sub(pat, "", s)
    return re.sub(r"\s+", " ", s).strip(" .")


_TEAM_ALIASES: dict[str, str] = {
    # short names that appear in football-data CSVs -> canonical
    "man city":   "Manchester City",  "man united": "Manchester United",
    "man utd":    "Manchester United", "psg":        "Paris SG",
    "paris saint-germain": "Paris SG", "bayer leverkusen": "Leverkusen",
    "atletico":   "Ath Madrid",       "atletico madrid": "Ath Madrid",
    "athletic":   "Ath Bilbao",       "athletic bilbao": "Ath Bilbao",
    "nott'm forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "rb leipzig":  "RB Leipzig",      "brighton & hove albion": "Brighton",
    "wolves": "Wolves", "wolverhampton": "Wolves",
    "spurs": "Tottenham",
}

def _canon(raw: str, known: set[str]) -> Optional[str]:
    """Map a free-text team name to a known team name in the dataset."""
    if not raw:
        return None
    q = str(raw).strip()
    if q in known:
        return q
    low = q.lower()
    if low in _TEAM_ALIASES:
        c = _TEAM_ALIASES[low]
        if c in known:
            return c
    sq = _strip(q)
    # exact stripped match
    for k in known:
        if _strip(k) == sq:
            return k
    # substring match
    for k in known:
        sk = _strip(k)
        if sq in sk or sk in sq:
            return k
    return None


# ─────────────────────────── Elo ─────────────────────────────────────────────
def calculate_elo_ratings(matches: pd.DataFrame,
                          k: float = ELO_K) -> dict[str, float]:
    """Walk-forward Elo from chronological match history."""
    elo: dict[str, float] = {}
    for r in matches.sort_values("date").itertuples(index=False):
        h, a = r.home, r.away
        eh, ea = elo.get(h, ELO_BASE), elo.get(a, ELO_BASE)
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
        sh = 1.0 if r.ftr == "H" else 0.0 if r.ftr == "A" else 0.5
        elo[h] = eh + k * (sh - exp_h)
        elo[a] = ea + k * ((1 - sh) - (1 - exp_h))
    return elo


# ─────────────────────────── form & home/away stats ──────────────────────────
def calculate_team_form(team: str, matches: pd.DataFrame,
                        asof: pd.Timestamp, n: int = FORM_N) -> dict:
    """Recent form stats for a team (last n matches before asof)."""
    m = matches[matches["date"] < asof].copy()
    mask = (m["home"] == team) | (m["away"] == team)
    recent = m[mask].sort_values("date").tail(n)
    if recent.empty:
        return {"points": 0, "wins": 0, "draws": 0, "goals_f": 0, "goals_a": 0,
                "shots_f": 0, "shots_a": 0, "n": 0}
    pts = wins = draws = gf = ga = sf = sa = 0
    # NaN is truthy, so `x or 0` leaks NaN into the feature vector and makes the
    # downstream sklearn predict() raise. Coerce missing shots to 0 explicitly.
    def _shots(v):
        return 0 if pd.isna(v) else v
    for r in recent.itertuples(index=False):
        if r.home == team:
            gf += r.fthg; ga += r.ftag
            sf += _shots(r.home_shots); sa += _shots(r.away_shots)
            if r.ftr == "H":   pts += 3; wins += 1
            elif r.ftr == "D": pts += 1; draws += 1
        else:
            gf += r.ftag; ga += r.fthg
            sf += _shots(r.away_shots); sa += _shots(r.home_shots)
            if r.ftr == "A":   pts += 3; wins += 1
            elif r.ftr == "D": pts += 1; draws += 1
    n_act = len(recent)
    return {"points": pts / n_act, "wins": wins / n_act, "draws": draws / n_act,
            "goals_f": gf / n_act, "goals_a": ga / n_act,
            "shots_f": sf / n_act, "shots_a": sa / n_act, "n": n_act}


def calculate_home_away_stats(team: str, matches: pd.DataFrame,
                               asof: pd.Timestamp, home: bool = True,
                               n: int = 15) -> dict:
    """Home (or away) specific stats for a team."""
    m = matches[matches["date"] < asof]
    sub = m[m["home"] == team].tail(n) if home else m[m["away"] == team].tail(n)
    if sub.empty:
        return {"goals_f": 0, "goals_a": 0, "shots_f": 0, "shots_a": 0,
                "win_rate": 0, "n": 0}
    gf = (sub["fthg"] if home else sub["ftag"]).fillna(0).mean()
    ga = (sub["ftag"] if home else sub["fthg"]).fillna(0).mean()
    sf = (sub["home_shots"] if home else sub["away_shots"]).fillna(0).mean()
    sa = (sub["away_shots"] if home else sub["home_shots"]).fillna(0).mean()
    res_col = "H" if home else "A"
    wins = (sub["ftr"] == res_col).mean()
    return {"goals_f": float(gf), "goals_a": float(ga),
            "shots_f": float(sf), "shots_a": float(sa),
            "win_rate": float(wins), "n": len(sub)}


# ─────────────────────────── team strength from players ──────────────────────
def calculate_player_based_team_strength(players_df: pd.DataFrame,
                                          team_col: str = "club") -> pd.DataFrame:
    """Aggregate FC26 attributes per club → attack / midfield / defense / GK scores."""
    df = players_df[players_df["fc_overall"].notna()].copy()
    rows = []
    for club, grp in df.groupby(team_col):
        if not isinstance(club, str) or not club.strip():
            continue
        top = grp.nlargest(18, "fc_overall")
        att  = top[top["position"] == "Attack"]
        mid  = top[top["position"] == "Midfield"]
        dfc  = top[top["position"] == "Defender"]
        gk   = top[top["position"] == "Goalkeeper"]
        rows.append({
            "club": club,
            "avg_overall":   float(top["fc_overall"].mean()),
            "avg_potential": float(top["fc_potential"].fillna(top["fc_overall"]).mean()),
            "attack_score":  float(att["fc_shooting"].mean()) if not att.empty else 65.0,
            "midfield_score":float(mid["fc_passing"].mean())  if not mid.empty else 65.0,
            "defense_score": float(dfc["fc_defending"].mean()) if not dfc.empty else 65.0,
            "gk_score":      float(gk["fc_overall"].mean())    if not gk.empty else 65.0,
        })
    return pd.DataFrame(rows).set_index("club")


def _strength(team: str, strength_df: pd.DataFrame, known_teams: set[str]) -> dict:
    canon = _canon(team, set(strength_df.index))
    if canon and canon in strength_df.index:
        r = strength_df.loc[canon]
        return r.to_dict()
    # fallback: look up in match known teams, try player club names
    canon2 = _canon(team, known_teams)
    if canon2 and canon2 in strength_df.index:
        return strength_df.loc[canon2].to_dict()
    return {"avg_overall": 75.0, "avg_potential": 75.0, "attack_score": 70.0,
            "midfield_score": 70.0, "defense_score": 70.0, "gk_score": 70.0}


# ─────────────────────────── feature engineering ─────────────────────────────
def build_match_features(home: str, away: str, matches: pd.DataFrame,
                          elo: dict, strength_df: pd.DataFrame,
                          asof: Optional[pd.Timestamp] = None) -> dict:
    """Build the full feature vector for one match."""
    if asof is None:
        asof = pd.Timestamp.now()
    known = set(matches["home"].dropna()) | set(matches["away"].dropna())
    hf = calculate_team_form(home, matches, asof)
    af = calculate_team_form(away, matches, asof)
    hha = calculate_home_away_stats(home, matches, asof, home=True)
    aha = calculate_home_away_stats(away, matches, asof, home=False)
    hs = _strength(home, strength_df, known)
    as_ = _strength(away, strength_df, known)
    eh  = elo.get(home, ELO_BASE)
    ea  = elo.get(away, ELO_BASE)
    feat = {
        # form
        "home_pts_last6":    hf["points"],   "away_pts_last6":    af["points"],
        "home_wins_last6":   hf["wins"],     "away_wins_last6":   af["wins"],
        "home_goals_f_last6":hf["goals_f"],  "away_goals_f_last6":af["goals_f"],
        "home_goals_a_last6":hf["goals_a"],  "away_goals_a_last6":af["goals_a"],
        "home_shots_last6":  hf["shots_f"],  "away_shots_last6":  af["shots_f"],
        "form_diff":         hf["points"] - af["points"],
        "goals_diff_last6":  hf["goals_f"] - af["goals_f"],
        # home/away specific
        "home_ha_goals_f":   hha["goals_f"], "home_ha_goals_a":   hha["goals_a"],
        "home_ha_shots":     hha["shots_f"], "home_ha_win_rate":  hha["win_rate"],
        "away_ha_goals_f":   aha["goals_f"], "away_ha_goals_a":   aha["goals_a"],
        "away_ha_shots":     aha["shots_f"], "away_ha_win_rate":  aha["win_rate"],
        # Elo
        "home_elo":          eh,             "away_elo":          ea,
        "elo_diff":          eh - ea,
        # squad strength
        "home_overall":      hs["avg_overall"],     "away_overall":  as_["avg_overall"],
        "overall_diff":      hs["avg_overall"] - as_["avg_overall"],
        "home_attack":       hs["attack_score"],    "away_attack":   as_["attack_score"],
        "home_defense":      hs["defense_score"],   "away_defense":  as_["defense_score"],
        "attack_vs_def":     hs["attack_score"] - as_["defense_score"],
        "away_attack_vs_home_def": as_["attack_score"] - hs["defense_score"],
        # data availability
        "home_data_n":       hf["n"],        "away_data_n":       af["n"],
    }
    return feat


def _feat_vector(feat: dict, cols: list[str]) -> np.ndarray:
    return np.array([feat.get(c, 0.0) for c in cols], dtype=float)


# ─────────────────────────── model training ──────────────────────────────────
_RESULT_COLS = [
    "home_pts_last6","away_pts_last6","home_wins_last6","away_wins_last6",
    "home_goals_f_last6","away_goals_f_last6","home_goals_a_last6","away_goals_a_last6",
    "home_shots_last6","away_shots_last6","form_diff","goals_diff_last6",
    "home_ha_goals_f","home_ha_goals_a","home_ha_shots","home_ha_win_rate",
    "away_ha_goals_f","away_ha_goals_a","away_ha_shots","away_ha_win_rate",
    "home_elo","away_elo","elo_diff",
    "home_overall","away_overall","overall_diff",
    "home_attack","away_attack","home_defense","away_defense",
    "attack_vs_def","away_attack_vs_home_def",
]

def _precompute_rolling_stats(matches: pd.DataFrame, n: int = FORM_N) -> pd.DataFrame:
    """Vectorised rolling team stats — O(matches) instead of O(matches²).

    For each match row, pre-fetch the last-n-matches stats for both home and away teams
    using pandas shift + groupby rather than iterating over all past rows.
    """
    m = matches.sort_values("date").reset_index(drop=True).copy()
    m["h_pts"]  = (m["ftr"]=="H").astype(float)*3 + (m["ftr"]=="D").astype(float)
    m["a_pts"]  = (m["ftr"]=="A").astype(float)*3 + (m["ftr"]=="D").astype(float)
    # Long-form: one row per team per match
    home_rows = m[["date","home","fthg","ftag","home_shots","away_shots","ftr","h_pts"]].copy()
    home_rows.columns = ["date","team","gf","ga","sf","sa","ftr","pts"]
    home_rows["is_home"] = True
    away_rows = m[["date","away","ftag","fthg","away_shots","home_shots","ftr","a_pts"]].copy()
    away_rows.columns = ["date","team","gf","ga","sf","sa","ftr","pts"]
    away_rows["is_home"] = False
    long = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("date")
    long["win"] = ((long["ftr"]=="H") & long["is_home"]) | ((long["ftr"]=="A") & ~long["is_home"])

    def roll(grp):
        grp = grp.sort_values("date")
        for c in ["gf","ga","sf","sa","pts","win"]:
            grp[f"r_{c}"] = grp[c].shift(1).rolling(n, min_periods=1).mean()
        grp["r_n"] = grp["gf"].shift(1).rolling(n, min_periods=1).count()
        return grp

    long = long.groupby("team", group_keys=False).apply(roll)

    # Merge back: home stats
    home_stats = long[long["is_home"]].rename(columns={
        "team":"home","r_gf":"h_gf","r_ga":"h_ga","r_sf":"h_sf","r_pts":"h_pts_r","r_win":"h_win","r_n":"h_n"
    })[["date","home","h_gf","h_ga","h_sf","h_pts_r","h_win","h_n"]]
    away_stats = long[~long["is_home"]].rename(columns={
        "team":"away","r_gf":"a_gf","r_ga":"a_ga","r_sf":"a_sf","r_pts":"a_pts_r","r_win":"a_win","r_n":"a_n"
    })[["date","away","a_gf","a_ga","a_sf","a_pts_r","a_win","a_n"]]

    m = m.merge(home_stats, on=["date","home"], how="left")
    m = m.merge(away_stats, on=["date","away"],  how="left")
    # Home-specific stats (how team performs at home vs away)
    home_only = long[long["is_home"]].rename(columns={"team":"home"})
    away_only  = long[~long["is_home"]].rename(columns={"team":"away"})
    ha_h = home_only.groupby("home").apply(lambda g: g.sort_values("date")[["r_gf","r_ga","r_sf","r_win"]].shift(1).rolling(10,min_periods=1).mean().iloc[-1] if len(g)>1 else pd.Series({"r_gf":0,"r_ga":0,"r_sf":0,"r_win":0}))
    ha_a = away_only.groupby("away").apply(lambda g: g.sort_values("date")[["r_gf","r_ga","r_sf","r_win"]].shift(1).rolling(10,min_periods=1).mean().iloc[-1] if len(g)>1 else pd.Series({"r_gf":0,"r_ga":0,"r_sf":0,"r_win":0}))
    ha_h.columns = ["hh_gf","hh_ga","hh_sf","hh_win"]
    ha_a.columns = ["ah_gf","ah_ga","ah_sf","ah_win"]
    m = m.merge(ha_h.reset_index(), on="home", how="left")
    m = m.merge(ha_a.reset_index(), on="away", how="left")
    return m


def _build_training_data(matches: pd.DataFrame, elo_at_time: bool,
                          strength_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix from historical matches using pre-computed rolling stats."""
    m = _precompute_rolling_stats(matches)
    # Compute Elo progression
    elo_at: list[tuple[float,float]] = []
    elo: dict[str,float] = {}
    for _, row in m.sort_values("date").iterrows():
        elo_at.append((elo.get(row["home"], ELO_BASE), elo.get(row["away"], ELO_BASE)))
        _update_elo(elo, row["home"], row["away"], row["ftr"])
    m["elo_h"] = [e[0] for e in elo_at]
    m["elo_a"] = [e[1] for e in elo_at]

    # Merge player strength
    str_reset = strength_df.reset_index()
    str_h = str_reset.rename(columns={c:f"ps_h_{c}" for c in strength_df.columns}).rename(columns={"club":"home"})
    str_a = str_reset.rename(columns={c:f"ps_a_{c}" for c in strength_df.columns}).rename(columns={"club":"away"})
    m = m.merge(str_h, on="home", how="left")
    m = m.merge(str_a, on="away", how="left")
    for c in strength_df.columns:
        m[f"ps_h_{c}"] = m[f"ps_h_{c}"].fillna(strength_df[c].median())
        m[f"ps_a_{c}"] = m[f"ps_a_{c}"].fillna(strength_df[c].median())

    # Drop rows without enough history
    m = m[(m["h_n"].fillna(0) >= MIN_MATCHES) & (m["a_n"].fillna(0) >= MIN_MATCHES)].copy()

    fill_zero = ["h_gf","h_ga","h_sf","h_pts_r","h_win","a_gf","a_ga","a_sf","a_pts_r","a_win",
                 "hh_gf","hh_ga","hh_sf","hh_win","ah_gf","ah_ga","ah_sf","ah_win"]
    for c in fill_zero:
        if c in m.columns:
            m[c] = m[c].fillna(0.0)

    def row_to_vec(r):
        return [
            r.get("h_pts_r",0),    r.get("a_pts_r",0),
            r.get("h_win",0),      r.get("a_win",0),
            r.get("h_gf",0),       r.get("a_gf",0),
            r.get("h_ga",0),       r.get("a_ga",0),
            r.get("h_sf",0),       r.get("a_sf",0),
            r.get("h_pts_r",0) - r.get("a_pts_r",0),
            r.get("h_gf",0)    - r.get("a_gf",0),
            r.get("hh_gf",0),   r.get("hh_ga",0), r.get("hh_sf",0), r.get("hh_win",0),
            r.get("ah_gf",0),   r.get("ah_ga",0), r.get("ah_sf",0), r.get("ah_win",0),
            r["elo_h"],  r["elo_a"],  r["elo_h"] - r["elo_a"],
            r.get("ps_h_avg_overall",75),  r.get("ps_a_avg_overall",75),
            r.get("ps_h_avg_overall",75) - r.get("ps_a_avg_overall",75),
            r.get("ps_h_attack_score",70), r.get("ps_a_attack_score",70),
            r.get("ps_h_defense_score",70),r.get("ps_a_defense_score",70),
            r.get("ps_h_attack_score",70) - r.get("ps_a_defense_score",70),
            r.get("ps_a_attack_score",70) - r.get("ps_h_defense_score",70),
            r.get("h_n",0),  r.get("a_n",0),
        ]

    rows = [row_to_vec(r) for r in m.to_dict(orient="records")]
    X = np.array(rows, dtype=float)
    y_res = m["ftr"].values
    y_hg  = m["fthg"].values.astype(float)
    y_ag  = m["ftag"].values.astype(float)
    return X, y_res, y_hg, y_ag


def _update_elo(elo: dict, h: str, a: str, result: str, k: float = ELO_K):
    eh = elo.get(h, ELO_BASE); ea = elo.get(a, ELO_BASE)
    exp_h = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
    sh = 1.0 if result == "H" else 0.0 if result == "A" else 0.5
    elo[h] = eh + k * (sh - exp_h)
    elo[a] = ea + k * ((1 - sh) - (1 - exp_h))


def train_match_result_model(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    yc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)
    clf.fit(X, yc)
    return clf, le


def train_goal_model(X: np.ndarray, y_home: np.ndarray, y_away: np.ndarray):
    from sklearn.ensemble import GradientBoostingRegressor
    gh = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=10, random_state=42)
    ga = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=10, random_state=42)
    gh.fit(X, y_home.clip(0, 7))
    ga.fit(X, y_away.clip(0, 7))
    return gh, ga


# ─────────────────────────── top-scorer model ────────────────────────────────
def build_player_season_features(players_df: pd.DataFrame,
                                  strength_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-player feature table for the goals projection model.

    Uses ONLY players with current-season FBref stats (stats_source='fbref_2025_26')
    in the top-5 leagues — ensuring training labels are current-season goals,
    not inflated career aggregates.
    """
    TOP5_LEAGUES = {"GB1","ES1","IT1","L1","FR1"}
    df = players_df[
        (players_df["stats_source"] == "fbref_2025_26") &
        players_df["fc_overall"].notna() &
        players_df["sub_position"].isin(ATTACKER_SUBS) &
        (pd.to_numeric(players_df["minutes_played"], errors="coerce").fillna(0) >= 450) &
        players_df["league"].isin(TOP5_LEAGUES)
    ].copy()

    for c in ["goals","goals_per90","minutes_played","fc_shooting","fc_pace",
              "fc_dribbling","fc_overall","fc_potential","age","market_value_in_eur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df.rename(columns={"market_value_in_eur":"market_value"}, inplace=True)
    # Cap goals_per90 at 1.0 — extreme values from short samples distort training
    df["goals_per90"] = df["goals_per90"].clip(upper=1.0)
    # Expected full-season minutes (project current rate to ~3000 min)
    df["exp_season_goals"] = df["goals_per90"] * 3000 / 90.0

    def team_atk(club):
        c = _canon(club, set(strength_df.index))
        return float(strength_df.loc[c, "attack_score"]) if (c and c in strength_df.index) else 70.0

    df["team_attack"] = df["club"].map(team_atk)
    feats = ["goals_per90","minutes_played","fc_shooting","fc_pace","fc_dribbling",
             "fc_overall","fc_potential","age","market_value","team_attack"]
    return df[["player_id","player_name","club","nationality","league",
               "sub_position","goals","exp_season_goals"] + feats].dropna(subset=feats)


def train_top_scorer_model(feat_df: pd.DataFrame):
    """RandomForestRegressor: current-season player attributes → projected full-season goals."""
    from sklearn.ensemble import RandomForestRegressor
    feat_cols = ["goals_per90","minutes_played","fc_shooting","fc_pace","fc_dribbling",
                 "fc_overall","fc_potential","age","market_value","team_attack"]
    X = feat_df[feat_cols].values
    # Target = actual goals (current season, real FBref data — not career aggregates)
    y = feat_df["goals"].values
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)
    rf.fit(X, y)
    return rf, feat_cols


# ─────────────────────────── scoreline logic ─────────────────────────────────
def generate_scoreline_distribution(xg_h: float, xg_a: float, max_g: int = 6) -> dict:
    """Full Poisson scoreline probability distribution."""
    dist = {}
    for h in range(max_g + 1):
        ph = math.exp(-xg_h) * xg_h ** h / math.factorial(h)
        for a in range(max_g + 1):
            pa = math.exp(-xg_a) * xg_a ** a / math.factorial(a)
            dist[(h, a)] = ph * pa
    return dist


def classify_match_profile(xg_h: float, xg_a: float, elo_diff: float) -> str:
    total  = xg_h + xg_a
    diff   = abs(xg_h - xg_a)
    abs_ed = abs(elo_diff)
    if total < 1.6:   return "low_scoring_match"
    if abs_ed > 150 and diff > 1.2: return "one_sided_match"
    if abs_ed > 80  and diff > 0.8: return "attacking_mismatch"
    if total > 3.2:   return "open_match"
    if total > 2.5 and diff < 0.5: return "high_tempo_match"
    return "balanced_match"


# Profile → plausible score range
_PROFILE_BOUNDS = {
    "low_scoring_match":   (0, 1, 0, 1),   # (min_h, max_h, min_a, max_a)
    "balanced_match":      (0, 2, 0, 2),
    "high_tempo_match":    (1, 3, 1, 3),
    "open_match":          (1, 4, 1, 4),
    "attacking_mismatch":  (2, 5, 0, 2),
    "one_sided_match":     (2, 5, 0, 1),
}


def select_context_aware_scoreline(dist: dict, profile: str,
                                    xg_h: float, xg_a: float,
                                    p_home: float, p_draw: float, p_away: float) -> tuple[int,int]:
    """
    Context-aware scoreline selector.
    final_score = poisson_prob*0.40 + rf_align*0.20 + profile_fit*0.20 + xg_fit*0.10
                  + margin_fit*0.05 + rarity_penalty*0.05

    rf_align ensures the selected scoreline agrees with the RF outcome classifier so the
    text summary and the predicted result are never contradictory (e.g. "1-1" when the
    model predicts a home win).
    """
    bounds = _PROFILE_BOUNDS.get(profile, (0, 5, 0, 5))
    hlo, hhi, alo, ahi = bounds
    best_score, best = -1.0, (1, 1)
    win_margin = xg_h - xg_a
    rf_outcome = ("H" if p_home >= max(p_home, p_draw, p_away)
                  else "A" if p_away >= max(p_home, p_draw, p_away)
                  else "D")

    for (h, a), prob in dist.items():
        # RF alignment: strongly prefer scorelines that agree with the RF outcome label
        is_draw = (h == a)
        if rf_outcome == "H":
            rf_align = 1.0 if h > a else (0.35 if is_draw else 0.65)
        elif rf_outcome == "A":
            rf_align = 1.0 if a > h else (0.35 if is_draw else 0.65)
        else:
            rf_align = 1.0 if is_draw else 0.65
        # profile fit: prefer scorelines within the profile's expected range
        prof_fit = 1.0 if (hlo <= h <= hhi and alo <= a <= ahi) else 0.3
        # xG rounding fit: prefer scorelines closest to xG
        xg_fit = max(0.0, 1.0 - (abs(h - xg_h) + abs(a - xg_a)) / (xg_h + xg_a + 1.0))
        # margin fit: prefer scorelines whose margin aligns with win probability
        score_margin = h - a
        if win_margin > 0.3:          expected_margin = max(score_margin, 0)
        elif win_margin < -0.3:       expected_margin = min(score_margin, 0)
        else:                         expected_margin = 0
        margin_fit = max(0.0, 1.0 - abs(score_margin - expected_margin) / 3.0)
        # extreme score penalty
        rarity = 1.0 if h + a <= 5 else 0.5

        combined = (0.40 * prob / max(dist.values()) +
                    0.20 * rf_align +
                    0.20 * prof_fit +
                    0.10 * xg_fit +
                    0.05 * margin_fit +
                    0.05 * rarity)
        if combined > best_score:
            best_score, best = combined, (h, a)
    return best


def calculate_prediction_confidence(p_home: float, p_draw: float, p_away: float,
                                     data_n: int, home_known: bool = True) -> str:
    gap    = max(p_home, p_draw, p_away) - sorted([p_home, p_draw, p_away])[-2]
    if gap > 0.30 and data_n >= MIN_MATCHES and home_known:  return "High"
    if gap > 0.20 and data_n >= MIN_MATCHES:                 return "Medium-High"
    if gap > 0.10:                                           return "Medium"
    return "Low"


# ─────────────────────────── home/away resolver ───────────────────────────────
def resolve_home_away(team_a: str, team_b: str, user_text: str,
                       matches: pd.DataFrame) -> dict:
    """
    Determines home/away from:
    1. Explicit user phrasing  2. Fixture lookup  3. Assumption (first = home)
    """
    ut = user_text.lower()
    home = away = None
    source = "unknown"
    # 1. Explicit phrasing
    host_pats = [
        r"([\w\s]+)\s+(?:host[s]?|welcome[s]?|at home(?:\s+against)?)\s+([\w\s]+)",
        r"([\w\s]+)\s+(?:vs?\.?|against)\s+([\w\s]+)\s+(?:at home|home game)",
        r"([\w\s]+)\s+(?:visit[s]?|travel[s]? to|away at)\s+([\w\s]+)",
    ]
    for pat in host_pats:
        m = re.search(pat, ut)
        if m:
            raw_h, raw_a = m.group(1).strip(), m.group(2).strip()
            if "visit" in pat or "away at" in pat or "travel" in pat:
                raw_h, raw_a = raw_a, raw_h
            known = set(matches["home"].dropna()) | set(matches["away"].dropna())
            home = _canon(raw_h, known)
            away = _canon(raw_a, known)
            if home and away:
                source = "explicit"
                break
    # 2. Fixture lookup in current / recent season
    if not (home and away):
        known = set(matches["home"].dropna()) | set(matches["away"].dropna())
        ca = _canon(team_a, known)
        cb = _canon(team_b, known)
        if ca and cb:
            future = matches[
                ((matches["home"]==ca) & (matches["away"]==cb)) |
                ((matches["home"]==cb) & (matches["away"]==ca))
            ].sort_values("date")
            if not future.empty:
                row = future.iloc[-1]
                home, away = row["home"], row["away"]
                source = "fixture_lookup"
            else:
                home, away = ca, cb   # assume first-mentioned = home
                source = "assumed"
        else:
            home, away = team_a, team_b
            source = "assumed"
    return {
        "home_team": home, "away_team": away,
        "home_away_source": source,
        "home_known": source != "assumed",
    }


# ─────────────────────────── prediction response formatter ───────────────────
def generate_prediction_response(pred: dict) -> str:
    t = pred.get("type")
    if t == "match":
        return _fmt_match(pred)
    if t == "top_scorer":
        return _fmt_top_scorer(pred)
    if t == "player_goals":
        return _fmt_player_goals(pred)
    return json.dumps(pred, indent=2)


def _pct(v): return f"{round(v*100)}%"

def _fmt_match(p: dict) -> str:
    h, a = p["home_team"], p["away_team"]
    sh, sa = p["score"]
    winner = h if sh > sa else a if sa > sh else "Draw"
    conf   = p["confidence"]
    profile = p["match_profile"].replace("_", " ")
    lines = [
        f"**Prediction: {h} {sh}–{sa} {a}**\n",
        f"{'Draw' if winner == 'Draw' else winner + ' win'} | "
        f"probabilities: {h} {_pct(p['p_home'])} | draw {_pct(p['p_draw'])} | {a} {_pct(p['p_away'])}",
        f"Expected goals: {h} {p['xg_home']:.2f} — {a} {p['xg_away']:.2f}",
        f"Match profile: {profile} | confidence: **{conf}**\n",
        "**Key factors:**",
    ]
    for f in p.get("factors", []):
        lines.append(f"- {f}")
    if p.get("home_assumption"):
        lines.append(f"\n_Assumption: {h} treated as home team (no confirmed fixture found)._")
    lines.append(f"\n🔍 Method: RandomForest result classifier + Gradient Boosting xG model + "
                 "context-aware Poisson scoreline selector trained on 10 seasons of top-5 league data.")
    return "\n".join(lines)


def _fmt_top_scorer(p: dict) -> str:
    lines = [f"**Projected top scorers — {p.get('scope', 'top-5 leagues')} (next season):**\n"]
    for i, c in enumerate(p["candidates"], 1):
        lines.append(
            f"{i}. **{c['player_name']}** ({c['club']} | {c['position']} | age {c['age']}) "
            f"— projected {c['projected_goals']:.0f} goals | "
            f"OVR {c['overall']} shooting {c['shooting']}"
        )
    lines.append(f"\nConfidence: **{p['confidence']}**")
    lines.append("_Caveat: projection uses current squad/attribute data. Future transfers, "
                 "injuries, or team changes are not modelled._")
    lines.append(f"\n🔍 Method: RandomForest regressor trained on current-season player attributes "
                 "→ goals. Ranked by projected_goals_next_season.")
    from viz import embed_viz
    viz = {"type": "ranking", "title": f"Projected top scorers — {p.get('scope', 'top-5 leagues')}",
           "unit": "goals",
           "items": [{"name": c["player_name"], "value": round(float(c["projected_goals"])), "sub": c["club"]}
                     for c in p["candidates"][:6]]}
    return embed_viz("\n".join(lines), viz)


def _fmt_player_goals(p: dict) -> str:
    c = p["candidate"]
    return (
        f"**Projected goals for {c['player_name']} next season: ~{c['projected_goals']:.0f}**\n\n"
        f"Current season: {c['actual_goals']} goals in {c['minutes']:.0f} min "
        f"({c['goals_per90']:.2f}/90)\n"
        f"OVR {c['overall']} | shooting {c['shooting']} | age {c['age']}\n"
        f"Club: {c['club']}\n\n"
        f"Confidence: **{p['confidence']}**\n"
        "_Caveat: does not account for future transfers, injuries, or tactical changes._\n\n"
        f"🔍 Method: RandomForest regressor on player attributes → goals projection."
    )


# ─────────────────────────── PredictionEngine class ──────────────────────────
class PredictionEngine:
    """
    Central prediction engine. Trains once at startup, serves all prediction queries.
    """

    def __init__(self, club_matches_path=CLUB_MATCHES_CSV,
                 national_matches_path=NATIONAL_MATCHES_CSV,
                 players_path=PLAYERS_CSV):
        print("[prediction] Loading match and player data...", flush=True)
        self.matches = self._load_matches(club_matches_path)
        self.nat_matches = self._load_national(national_matches_path)
        self.players = pd.read_csv(players_path, low_memory=False)
        self.known_clubs = set(self.matches["home"].dropna()) | set(self.matches["away"].dropna())

        print("[prediction] Computing team ratings...", flush=True)
        self.club_elo = calculate_elo_ratings(self.matches)

        print("[prediction] Building team strength from player data...", flush=True)
        self.strength_df = calculate_player_based_team_strength(self.players)

        print("[prediction] Building training data & training models...", flush=True)
        X, y_res, y_hg, y_ag = _build_training_data(self.matches, True, self.strength_df)
        self.feature_cols = _RESULT_COLS  # kept for reference (model trained on positional vecs)
        self.result_model, self.result_le = train_match_result_model(X, y_res)
        self.goal_model_h, self.goal_model_a = train_goal_model(X, y_hg, y_ag)

        print("[prediction] Training top-scorer model...", flush=True)
        self.player_feat_df = build_player_season_features(self.players, self.strength_df)
        self.scorer_model, self.scorer_cols = train_top_scorer_model(self.player_feat_df)
        self.player_feat_df["projected_goals"] = self.scorer_model.predict(
            self.player_feat_df[self.scorer_cols].values)

        print(f"[prediction] Ready: {len(X)} training matches, {len(self.player_feat_df)} attacker profiles.", flush=True)

    @staticmethod
    def _load_matches(path):
        df = pd.read_csv(path, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce").fillna(0)
        df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce").fillna(0)
        df["home_shots"] = pd.to_numeric(df.get("home_shots"), errors="coerce")
        df["away_shots"] = pd.to_numeric(df.get("away_shots"), errors="coerce")
        return df.dropna(subset=["date","home","away","ftr"])

    @staticmethod
    def _load_national(path):
        if not Path(path).exists():
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def resolve(self, team: str) -> Optional[str]:
        return _canon(team, self.known_clubs)

    # ── match prediction ────────────────────────────────────────────────────
    def predict_club_match(self, team_a: str, team_b: str,
                            user_text: str = "") -> dict:
        ctx = resolve_home_away(team_a, team_b, user_text, self.matches)
        home, away = ctx["home_team"], ctx["away_team"]
        asof = pd.Timestamp.now()
        feat  = build_match_features(home, away, self.matches, self.club_elo,
                                      self.strength_df, asof)
        # Build the same positional vector the model was trained on
        hs = _strength(home, self.strength_df, self.known_clubs)
        as_ = _strength(away, self.strength_df, self.known_clubs)
        eh = self.club_elo.get(home, ELO_BASE)
        ea = self.club_elo.get(away, ELO_BASE)
        hf = calculate_team_form(home, self.matches, asof)
        af = calculate_team_form(away, self.matches, asof)
        hha = calculate_home_away_stats(home, self.matches, asof, home=True)
        aha = calculate_home_away_stats(away, self.matches, asof, home=False)
        vec_list = [
            hf["points"], af["points"], hf["wins"], af["wins"],
            hf["goals_f"], af["goals_f"], hf["goals_a"], af["goals_a"],
            hf["shots_f"], af["shots_f"],
            hf["points"]-af["points"], hf["goals_f"]-af["goals_f"],
            hha["goals_f"], hha["goals_a"], hha["shots_f"], hha["win_rate"],
            aha["goals_f"], aha["goals_a"], aha["shots_f"], aha["win_rate"],
            eh, ea, eh-ea,
            hs["avg_overall"], as_["avg_overall"], hs["avg_overall"]-as_["avg_overall"],
            hs["attack_score"], as_["attack_score"], hs["defense_score"], as_["defense_score"],
            hs["attack_score"]-as_["defense_score"], as_["attack_score"]-hs["defense_score"],
            hf["n"], af["n"],
        ]
        vec = np.array([vec_list], dtype=float)

        probs = self.result_model.predict_proba(vec)[0]
        classes = list(self.result_le.classes_)
        p_home = float(probs[classes.index("H")])
        p_draw = float(probs[classes.index("D")])
        p_away = float(probs[classes.index("A")])

        xg_h = max(0.2, float(self.goal_model_h.predict(vec)[0]))
        xg_a = max(0.2, float(self.goal_model_a.predict(vec)[0]))

        profile = classify_match_profile(xg_h, xg_a, feat["elo_diff"])
        dist    = generate_scoreline_distribution(xg_h, xg_a)
        score   = select_context_aware_scoreline(dist, profile, xg_h, xg_a, p_home, p_draw, p_away)
        conf    = calculate_prediction_confidence(
            p_home, p_draw, p_away, min(feat["home_data_n"], feat["away_data_n"]),
            ctx["home_known"])

        factors = self._explain(home, away, feat, p_home, p_draw, p_away, xg_h, xg_a, profile)
        return {
            "type": "match", "home_team": home, "away_team": away,
            "score": score, "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "xg_home": round(xg_h,2), "xg_away": round(xg_a,2),
            "match_profile": profile, "confidence": conf, "factors": factors,
            "home_assumption": ctx["home_away_source"] == "assumed",
            "elo_diff": round(feat["elo_diff"],0),
        }

    def _explain(self, home, away, feat, ph, pd_, pa, xgh, xga, profile) -> list[str]:
        f = []
        if feat["elo_diff"] > 40:
            f.append(f"{home} have a stronger historical rating (+{feat['elo_diff']:.0f})")
        elif feat["elo_diff"] < -40:
            f.append(f"{away} have a stronger historical rating ({feat['elo_diff']:.0f})")
        if feat["form_diff"] > 0.3:
            f.append(f"{home} are in better recent form (+{feat['form_diff']:.1f} pts/game)")
        elif feat["form_diff"] < -0.3:
            f.append(f"{away} are in better recent form ({feat['form_diff']:.1f} pts/game)")
        if feat["overall_diff"] > 3:
            f.append(f"{home} have a stronger squad (avg OVR +{feat['overall_diff']:.0f})")
        elif feat["overall_diff"] < -3:
            f.append(f"{away} have a stronger squad (avg OVR {feat['overall_diff']:.0f})")
        if feat["home_ha_win_rate"] > 0.5:
            f.append(f"{home} strong at home ({feat['home_ha_win_rate']:.0%} home win rate)")
        f.append(f"xG model: {home} {xgh:.2f} — {away} {xga:.2f} expected goals")
        if not f:
            f.append("Teams are closely matched; this is a coin-flip contest.")
        return f[:5]

    # ── top scorer ──────────────────────────────────────────────────────────
    def predict_top_scorer(self, league: str = "", n: int = 5) -> dict:
        df = self.player_feat_df.copy()
        if league:
            # Map league name to league code
            _LEAGUE_MAP = {"premier league":"GB1","la liga":"ES1","bundesliga":"L1",
                           "serie a":"IT1","ligue 1":"FR1"}
            code = _LEAGUE_MAP.get(league.lower())
            if code:
                df = df[df["league"] == code]
        df = df.sort_values("projected_goals", ascending=False)
        candidates = []
        for _, r in df.head(n).iterrows():
            candidates.append({
                "player_name": r["player_name"], "club": r["club"],
                "position": r["sub_position"], "age": int(r["age"]),
                "overall": int(r["fc_overall"]), "shooting": int(r["fc_shooting"]),
                "goals_per90": round(float(r["goals_per90"]),2),
                "minutes": int(r["minutes_played"]),
                "projected_goals": round(float(r["projected_goals"]),1),
            })
        scope = league.title() if league else "top-5 leagues"
        conf  = "Medium" if not league else "Medium-High"
        return {"type":"top_scorer","scope":scope,"candidates":candidates,"confidence":conf}

    # ── player goals ────────────────────────────────────────────────────────
    def predict_player_goals(self, player_name: str) -> dict:
        df = self.player_feat_df
        q = player_name.strip().lower()
        hit = df[df["player_name"].str.lower().str.contains(q, na=False)]
        if hit.empty:
            return {"type":"player_goals","error":f"Player '{player_name}' not found in the attacker dataset."}
        r = hit.iloc[0]
        conf = "Medium" if r["minutes_played"] >= 1500 else "Low"
        return {
            "type": "player_goals",
            "candidate": {
                "player_name": r["player_name"], "club": r["club"],
                "age": int(r["age"]), "overall": int(r["fc_overall"]),
                "shooting": int(r["fc_shooting"]),
                "goals_per90": round(float(r["goals_per90"]),2),
                "actual_goals": int(r["goals"]),
                "minutes": int(r["minutes_played"]),
                "projected_goals": round(float(r["projected_goals"]),1),
            },
            "confidence": conf,
        }


# ─────────────────────────── query parser ─────────────────────────────────────
def parse_prediction_query(text: str) -> dict:
    """Rule-based parser (fallback when LLM unavailable)."""
    q = text.lower()
    intent = "match_result"
    if re.search(r"score|scoreline|result|final|finish", q):
        intent = "scoreline"
    if re.search(r"top scorer|top goal|golden boot|most goals|who will score the most", q):
        intent = "top_scorer"
    if re.search(r"how many goals will|goals? will .+ score|will .+ score more than", q):
        intent = "player_goals"
    # extract teams via "vs" / "against" / "between"
    team_a = team_b = player = league = None
    m = re.search(r"([a-z\s'.\-]+?)\s+(?:vs?\.?|against|versus|נגד|מול)\s+([a-z\s'.\-]+?)(?:\s+in|\?|$)", q)
    if m:
        team_a, team_b = m.group(1).strip(), m.group(2).strip()
    for comp in ["premier league","la liga","bundesliga","serie a","ligue 1",
                 "champions league","world cup","europe"]:
        if comp in q:
            league = comp; break
    if intent in ("top_scorer","player_goals"):
        pm = re.search(r"(?:will|will\s+\w+|for|about)\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:score|goals?)", q)
        if pm:
            player = pm.group(1).strip()
    return {"intent":intent,"team_a":team_a,"team_b":team_b,
            "player":player,"league":league,"raw":text}
