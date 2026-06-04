"""
ds_engine.py
ScoutAI data-science engine based on normalized numeric player vectors.

Methods:
  - Cosine similarity / Euclidean distance for numeric player vectors.
  - K-Means archetypes for role/profile clustering.
  - Z-score distance from cluster centroid for anomaly detection.
  - Jaccard similarity for categorical trait sets.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV     = "data/players_clean.csv"
FEATURES_NPY = "data/player_features.npy"
META_JSON    = "data/feature_meta.json"


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |intersection| / |union|."""
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class DSEngine:
    """Holds precomputed DS artifacts and provides similarity/anomaly helpers."""

    def __init__(self, df: pd.DataFrame, X: np.ndarray, meta: dict):
        self.df = df.reset_index(drop=True)
        self.X = X
        self.feature_names = meta["feature_names"]
        self.k = meta["k"]
        self.centroids = np.array(meta["centroids"])
        self.archetypes = {int(k): v for k, v in meta["archetypes"].items()}

        # Per-cluster stats for Z-score calculations.
        self.cluster_mean, self.cluster_std = {}, {}
        clusters = self.df["cluster"].values
        for c in np.unique(clusters):
            rows = self.X[clusters == c]
            self.cluster_mean[int(c)] = rows.mean(axis=0)
            self.cluster_std[int(c)]  = rows.std(axis=0) + 1e-9

        # Normalized player name -> row index lookup.
        import unicodedata
        def norm(s):
            if not isinstance(s, str): return ""
            nf = unicodedata.normalize("NFKD", s)
            return "".join(ch for ch in nf if not unicodedata.combining(ch)).lower().strip()
        self._norm = norm
        self.names_norm = self.df["player_name"].fillna("").map(norm)

    def find_index(self, query: str):
        q = self._norm(query)
        exact = self.df.index[self.names_norm == q]
        if len(exact):
            return self._most_prominent(exact)
        contains = self.df.index[self.names_norm.str.contains(q, na=False, regex=False)]
        if len(contains):
            return self._most_prominent(contains)
        for part in q.split():
            if len(part) < 3:
                continue
            m = self.df.index[self.names_norm.str.contains(part, na=False, regex=False)]
            if len(m):
                return self._most_prominent(m)
        return None

    def _most_prominent(self, idxs):
        """Choose the highest market-value match when several rows match a name."""
        sub = self.df.loc[idxs]
        return int(sub["market_value_in_eur"].fillna(0).idxmax())

    def cosine(self, target_iloc: int, cand_ilocs: np.ndarray) -> np.ndarray:
        return cosine_similarity(self.X[target_iloc].reshape(1, -1), self.X[cand_ilocs])[0]

    def euclidean(self, target_iloc: int, cand_ilocs: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.X[cand_ilocs] - self.X[target_iloc], axis=1)

    def cosine_to_vector(self, vec: np.ndarray, cand_ilocs: np.ndarray) -> np.ndarray:
        return cosine_similarity(vec.reshape(1, -1), self.X[cand_ilocs])[0]

    def zscores(self, iloc: int) -> dict:
        c = int(self.df.iloc[iloc]["cluster"])
        z = (self.X[iloc] - self.cluster_mean[c]) / self.cluster_std[c]
        return {self.feature_names[i]: float(z[i]) for i in range(len(self.feature_names))}

    def trait_set(self, iloc: int) -> set:
        r = self.df.iloc[iloc]
        traits = set()
        for col, prefix in [("position", "pos"), ("sub_position", "subpos"),
                            ("nationality", "nat"), ("foot", "foot"),
                            ("league", "league"), ("age_bucket", "age"),
                            ("value_tier", "val"), ("archetype", "arch")]:
            v = r.get(col)
            if isinstance(v, str) and v and v.lower() != "nan":
                traits.add(f"{prefix}:{v}")
        return traits

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)


def load_engine() -> DSEngine:
    df = pd.read_csv(DATA_CSV, low_memory=False)
    X  = np.load(FEATURES_NPY)
    with open(META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[ds_engine] DSEngine: {X.shape[0]} players x {X.shape[1]} features, "
          f"k={meta['k']} clusters", flush=True)
    return DSEngine(df, X, meta)


# National-team strength for World Cup predictions.

NATION_MAP = {
    "usa": "United States", "korea republic": "Korea, South", "ir iran": "Iran",
    "türkiye": "Turkey", "turkiye": "Turkey", "czechia": "Czech Republic",
    "côte d’ivoire": "Cote d'Ivoire", "cote d'ivoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire", "south korea": "Korea, South",
    "north korea": "Korea, North",
    "congo dr": "DR Congo", "cabo verde": "Cape Verde", "curaçao": "Curacao",
    "bosnia & herzegovina": "Bosnia-Herzegovina",
}


def normalize_nation(name: str, known: set):
    if not name:
        return None
    raw, low = name.strip(), name.strip().lower()
    if raw in known:
        return raw
    if low in NATION_MAP and NATION_MAP[low] in known:
        return NATION_MAP[low]
    for n in known:
        if n.lower() == low:
            return n
    for n in known:
        if low in n.lower() or n.lower() in low:
            return n
    return None


NATIONAL_MATCHES_CSV = "data/national_matches.csv"


def load_national_matches(path: str = NATIONAL_MATCHES_CSV):
    """Historical international results (committed CSV built from the WC workbook)."""
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    return m.sort_values("date").reset_index(drop=True)


def compute_elo(matches: pd.DataFrame, k: float = 40.0, base: float = 1500.0):
    """Walk-forward Elo from historical international results.

    Ratings update chronologically, so a match is always rated using only prior data.
    Also returns an out-of-sample hit-rate vs the bookmaker favourite on decisive
    matches that carry odds — a clean validation of the pedigree signal.
    """
    elo: dict[str, float] = {}
    n_matches: dict[str, int] = {}
    model_correct = book_correct = total = 0
    for r in matches.itertuples(index=False):
        h, a = r.home, r.away
        rh, ra = elo.get(h, base), elo.get(a, base)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        res = r.result
        s_h = 1.0 if res == "H" else 0.0 if res == "A" else 0.5

        odds_h, odds_a = getattr(r, "odds_home", None), getattr(r, "odds_away", None)
        if res in ("H", "A") and pd.notna(odds_h) and pd.notna(odds_a):
            total += 1
            if ("H" if rh >= ra else "A") == res:
                model_correct += 1
            if ("H" if odds_h <= odds_a else "A") == res:
                book_correct += 1

        elo[h] = rh + k * (s_h - exp_h)
        elo[a] = ra + k * ((1.0 - s_h) - (1.0 - exp_h))
        n_matches[h] = n_matches.get(h, 0) + 1
        n_matches[a] = n_matches.get(a, 0) + 1
    return elo, n_matches, (model_correct, book_correct, total)


def build_national_strength(df: pd.DataFrame, squad: int = 23,
                            pedigree_weight: float = 0.35) -> pd.DataFrame:
    """Hybrid national-team strength: current squad value blended with World Cup
    Elo pedigree.

    Squad strength (from current player market values) covers every nation, including
    2026 newcomers. Where historical World Cup results exist, an Elo pedigree term is
    blended in; teams with no history fall back to squad strength alone.
    """
    print("[ds_engine] Building national-team strength table...", flush=True)
    rows = []
    for nation, grp in df.groupby("nationality"):
        if not isinstance(nation, str):
            continue
        top = grp.nlargest(squad, "market_value_in_eur")
        if len(top) < 5:
            continue
        rows.append({
            "nationality": nation,
            "squad_value_mean": top["market_value_in_eur"].mean(),
            "squad_value_sum":  top["market_value_in_eur"].sum(),
            "top_scorers":      top.nlargest(5, "goals")["goals"].mean(),
            "depth":            len(grp),
        })
    nat = pd.DataFrame(rows).set_index("nationality")
    vmean = np.log1p(nat["squad_value_mean"]) / np.log1p(nat["squad_value_mean"].max())
    vsum  = np.log1p(nat["squad_value_sum"])  / np.log1p(nat["squad_value_sum"].max())
    depth = np.log1p(nat["depth"]) / np.log1p(nat["depth"].max())
    nat["squad_strength"] = 0.6 * vmean + 0.25 * vsum + 0.15 * depth

    matches = load_national_matches()
    if matches is not None and len(matches):
        elo, n_matches, (mc, bc, tot) = compute_elo(matches)
        nat["pedigree_elo"] = nat.index.map(lambda x: elo.get(x))
        nat["n_wc_matches"] = nat.index.map(lambda x: n_matches.get(x, 0)).astype(int)
        present = nat["pedigree_elo"].dropna()
        if len(present) > 1:
            lo, hi = present.min(), present.max()
            nat["pedigree_norm"] = ((nat["pedigree_elo"] - lo) / (hi - lo)).clip(0, 1)
        else:
            nat["pedigree_norm"] = np.nan
        has = nat["pedigree_elo"].notna()
        nat["has_history"] = has
        # Pedigree is a centered adjustment around the squad baseline: strong WC history
        # is a bonus, weak history a small penalty, and no history is neutral (so teams
        # without records are never unfairly outranked by mid-pedigree teams). The
        # combined score is min-max rescaled so the elite stay separated (no saturation).
        adj = pedigree_weight * (nat["pedigree_norm"] - 0.5)
        raw = nat["squad_strength"] + adj.fillna(0.0)
        nat["strength"] = (raw - raw.min()) / (raw.max() - raw.min())
        if tot:
            print(f"[ds_engine] Elo walk-forward backtest on {tot} decisive matches with "
                  f"odds: model {mc/tot:.0%} correct vs bookmaker {bc/tot:.0%}", flush=True)
        print(f"[ds_engine] Pedigree blended for {int(has.sum())}/{len(nat)} nations "
              f"(rest use squad strength).", flush=True)
    else:
        nat["pedigree_elo"] = np.nan
        nat["n_wc_matches"] = 0
        nat["has_history"] = False
        nat["strength"] = nat["squad_strength"]
        print("[ds_engine] No historical matches found; using squad strength only.", flush=True)

    print(f"[ds_engine] Computed strength for {len(nat)} national teams", flush=True)
    return nat
