"""
team_strength.py
Squad-strength feature builder for the Logistic Regression match predictor.

For every nation we derive a small set of interpretable strength features from the
Kaggle player data (players_clean.csv): starting-XI market value, average player
rating, and attack / defense quality. For the 48 FIFA World Cup 2026 teams we use
the OFFICIAL called-up squad (world_cup_2026_squads.csv) joined to the player data;
for any other nation we fall back to that nation's strongest players in the data.

These features are the input to:
  - train_predictor.py   (builds the training set + trains the model)
  - match_predictor.py   (loads the model and predicts at runtime)
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

PLAYERS_CSV = "data/players_clean.csv"
SQUADS_CSV  = "data/world_cup_2026_squads.csv"

# squads-CSV team name -> players_clean.nationality value (only the ones that differ)
SQUAD_NATION_MAP = {
    "Bosnia And Herzegovina": "Bosnia-Herzegovina",
    "Cabo Verde":             "Cape Verde",
    "Congo DR":               "DR Congo",
    "Curaçao":                "Curacao",
    "Czechia":                "Czech Republic",
    "Côte D'Ivoire":          "Cote d'Ivoire",
    "IR Iran":                "Iran",
    "Korea Republic":         "Korea, South",
    "USA":                    "United States",
}

# The interpretable strength features, in model order.
FEATURES = ["value_xi_log", "value_mean_log", "rating_mean", "attack", "defense"]

# Match-level (team_a vs team_b) difference features fed to the classifier.
DIFF_FEATURES = ["d_value_xi", "d_value_mean", "d_rating", "d_att_def", "d_def_att"]
# Per-team features fed to the Poisson goals model (own side vs opponent).
GOAL_FEATURES = ["own_value_xi", "own_attack", "opp_defense", "rating_edge"]


def match_diff(fa: np.ndarray, fb: np.ndarray) -> list[float]:
    """Strength differences between team A and team B (FEATURES order)."""
    return [fa[0] - fb[0],   # starting-XI value
            fa[1] - fb[1],   # squad mean value
            fa[2] - fb[2],   # average rating
            fa[3] - fb[4],   # A attack vs B defense
            fa[4] - fb[3]]   # A defense vs B attack


def goal_feats(own: np.ndarray, opp: np.ndarray) -> list[float]:
    """Features predicting how many goals `own` scores against `opp`."""
    return [own[0], own[3], opp[4], own[2] - opp[2]]

# Neutral defaults for a team we cannot resolve at all.
_DEFAULTS = {"value_xi_log": 15.0, "value_mean_log": 13.0,
             "rating_mean": 70.0, "attack": 68.0, "defense": 68.0,
             "n_matched": 0, "source": "default"}


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _toks(s: str) -> list[str]:
    s = _strip(s).lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    return [t for t in s.split() if len(t) >= 3]


def canonical_nation(team: str, known: set) -> str | None:
    """Map any spelling of a national team to a players_clean.nationality value."""
    if team in known:
        return team
    mapped = SQUAD_NATION_MAP.get(team)
    if mapped and mapped in known:
        return mapped
    low = team.strip().lower()
    for k in known:
        if k.lower() == low:
            return k
    # last resort: the ds_engine fuzzy mapper
    try:
        from ds_engine import normalize_nation
        n = normalize_nation(team, known)
        if n:
            return n
    except Exception:
        pass
    return None


class TeamStrength:
    """Builds and serves per-nation strength feature vectors."""

    def __init__(self, players_df: pd.DataFrame, squads_df: pd.DataFrame | None = None):
        self.pl = players_df.copy()
        for c in ("market_value_in_eur", "fc_overall", "fc_shooting", "fc_defending"):
            if c in self.pl.columns:
                self.pl[c] = pd.to_numeric(self.pl[c], errors="coerce")
        self.known = set(self.pl["nationality"].dropna().unique())
        self.table = self._build(squads_df)

    # ── feature aggregation for one set of player rows ────────────────────────
    @staticmethod
    def _agg(rows: pd.DataFrame, source: str) -> dict:
        mv = rows["market_value_in_eur"].fillna(0)
        xi = mv.nlargest(11).sum()
        mean23 = mv.nlargest(23).mean() if len(mv) else 0.0
        ovr = rows["fc_overall"].dropna()
        rating = float(ovr.nlargest(16).mean()) if len(ovr) else 70.0

        att_pool = rows.loc[rows["position"] == "Attack", "fc_shooting"].dropna()
        attack = float(att_pool.nlargest(5).mean()) if len(att_pool) else (
            float(rows["fc_shooting"].dropna().nlargest(5).mean()) if rows["fc_shooting"].notna().any() else 68.0)

        def_pool = rows.loc[rows["position"] == "Defender", "fc_defending"].dropna()
        defense = float(def_pool.nlargest(5).mean()) if len(def_pool) else (
            float(rows["fc_defending"].dropna().nlargest(5).mean()) if rows["fc_defending"].notna().any() else 68.0)

        return {
            "value_xi_log":   float(np.log1p(xi)),
            "value_mean_log": float(np.log1p(mean23)),
            "rating_mean":    rating,
            "attack":         attack,
            "defense":        defense,
            "n_matched":      int(len(rows)),
            "source":         source,
        }

    # ── squad → player-data row matching (token overlap, >=2 shared tokens) ────
    def _build_name_index(self):
        self._row_toks = {i: set(_toks(n)) for i, n in self.pl["player_name"].items()}
        idx = defaultdict(list)
        for i, ts in self._row_toks.items():
            for t in ts:
                idx[t].append(i)
        self._tok_idx = idx

    def _match_squad(self, squad_rows: pd.DataFrame, nat: str) -> pd.DataFrame:
        """Return the players_clean rows for a squad's players (token-overlap match,
        preferring same-nationality rows on ambiguity)."""
        picked: list[int] = []
        used: set[int] = set()
        for name in squad_rows["player_name"]:
            ts = set(_toks(name))
            if not ts:
                continue
            cand = defaultdict(int)
            for t in ts:
                for i in self._tok_idx.get(t, []):
                    cand[i] += 1
            best = [i for i, c in cand.items() if c >= 2 and i not in used]
            if not best:
                continue
            # prefer a row of the right nationality, then highest market value
            best.sort(key=lambda i: (self.pl.at[i, "nationality"] == nat,
                                     self.pl.at[i, "market_value_in_eur"] or 0), reverse=True)
            picked.append(best[0])
            used.add(best[0])
        return self.pl.loc[picked]

    # ── build the full per-nation table ───────────────────────────────────────
    def _build(self, squads_df: pd.DataFrame | None) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        # 1. nationality-based for every nation (covers historical opponents + fallback)
        for nat, grp in self.pl.groupby("nationality"):
            if not isinstance(nat, str) or len(grp) < 3:
                continue
            rows[nat] = self._agg(grp.nlargest(40, "market_value_in_eur"), source="nationality")

        # 2. squad-based override for the 48 WC 2026 teams (actual called-up players)
        if squads_df is not None:
            self._build_name_index()
            for team, sgrp in squads_df.groupby("team"):
                nat = canonical_nation(team, self.known) or team
                matched = self._match_squad(sgrp, nat)
                if len(matched) >= 8:
                    rows[nat] = self._agg(matched, source="squad")
        return pd.DataFrame.from_dict(rows, orient="index")

    # ── public API ────────────────────────────────────────────────────────────
    def features_for(self, team: str) -> tuple[np.ndarray, str, dict]:
        """Return (feature_vector, resolved_name, full_row_dict) for a team."""
        key = canonical_nation(team, set(self.table.index))
        if key is None:
            key = canonical_nation(team, self.known)
        if key is not None and key in self.table.index:
            row = self.table.loc[key]
            vec = np.array([float(row[f]) for f in FEATURES], dtype=float)
            return vec, key, row.to_dict()
        vec = np.array([_DEFAULTS[f] for f in FEATURES], dtype=float)
        return vec, team, dict(_DEFAULTS)

    def has(self, team: str) -> bool:
        key = canonical_nation(team, set(self.table.index))
        return key is not None and key in self.table.index
