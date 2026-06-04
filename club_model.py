"""
club_model.py
Club match-outcome predictor for the top-5 leagues, trained on data/club_matches.csv.

Data Science method: a recency-weighted Poisson goals model. Each team gets an attack
and a defence factor (relative to league scoring), plus a global home advantage. Expected
goals feed two Poisson distributions whose score matrix gives P(home win / draw / away
win), the expected score, and the most likely scoreline.

Validated with a time-split backtest (train on older seasons, test on the latest) against
the bookmaker favourite.
"""

import math
import numpy as np
import pandas as pd

from data_manager import normalize_team_name

CLUB_MATCHES_CSV = "data/club_matches.csv"
HALFLIFE_YEARS = 2.0      # recent seasons count for more; weight halves every 2 years
MAX_GOALS = 10            # Poisson score-matrix truncation


def load_club_matches(path: str = CLUB_MATCHES_CSV):
    import os
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path, low_memory=False)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    return m.dropna(subset=["date", "home", "away", "fthg", "ftag"]).reset_index(drop=True)


def _fit_factors(matches: pd.DataFrame, asof: pd.Timestamp, halflife: float = HALFLIFE_YEARS):
    """Recency-weighted attack/defence factors per team and global home/away baselines.

    Factors are normalised within each team's league so cross-league strength is
    comparable. Returns (factors, baselines) where factors[team] = dict(atk, def, league).
    """
    df = matches[matches["date"] <= asof].copy()
    age_yrs = (asof - df["date"]).dt.days / 365.25
    df["w"] = 0.5 ** (age_yrs / halflife)

    factors: dict[str, dict] = {}
    base = {}
    for league, lg in df.groupby("league"):
        w = lg["w"].to_numpy()
        wsum = w.sum()
        if wsum <= 0:
            continue
        home_avg = float((lg["fthg"] * w).sum() / wsum)   # league avg home goals
        away_avg = float((lg["ftag"] * w).sum() / wsum)   # league avg away goals
        team_avg = (home_avg + away_avg) / 2.0
        base[league] = {"home_avg": home_avg, "away_avg": away_avg, "team_avg": team_avg}

        # Weighted goals scored / conceded per team (home and away pooled).
        scored, conceded, weight = {}, {}, {}
        for r in lg.itertuples(index=False):
            for team, gf, ga in ((r.home, r.fthg, r.ftag), (r.away, r.ftag, r.fthg)):
                scored[team] = scored.get(team, 0.0) + r.w * gf
                conceded[team] = conceded.get(team, 0.0) + r.w * ga
                weight[team] = weight.get(team, 0.0) + r.w
        for team, wt in weight.items():
            if wt <= 0:
                continue
            atk = (scored[team] / wt) / team_avg if team_avg else 1.0
            dfc = (conceded[team] / wt) / team_avg if team_avg else 1.0
            factors[team] = {
                "atk": atk, "def": dfc, "league": league, "weight": wt,
            }

    # Global home/away baselines (for cross-league matchups).
    gw = df["w"].sum()
    baselines = {
        "home_avg": float((df["fthg"] * df["w"]).sum() / gw) if gw else 1.4,
        "away_avg": float((df["ftag"] * df["w"]).sum() / gw) if gw else 1.1,
        "by_league": base,
    }
    return factors, baselines


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _outcome_probs(lam_home: float, lam_away: float):
    ph = [_poisson_pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    pa = [_poisson_pmf(j, lam_away) for j in range(MAX_GOALS + 1)]
    p_home = p_draw = p_away = 0.0
    best_p, best_score = -1.0, (0, 0)
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if p > best_p:
                best_p, best_score = p, (i, j)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total, best_score


class ClubModel:
    """Recency-weighted Poisson model over the top-5 league match history."""

    def __init__(self, matches: pd.DataFrame | None = None, halflife: float = HALFLIFE_YEARS):
        self.matches = matches if matches is not None else load_club_matches()
        self.ok = self.matches is not None and len(self.matches) > 0
        if self.ok:
            asof = self.matches["date"].max()
            self.factors, self.baselines = _fit_factors(self.matches, asof, halflife)
            self._index = {normalize_team_name(t): t for t in self.factors}
        else:
            self.factors, self.baselines, self._index = {}, {}, {}

    def resolve(self, name: str):
        """Map a free-text club name to a known club key."""
        q = normalize_team_name(name)
        if not q:
            return None
        if q in self._index:
            return self._index[q]
        cand = [orig for norm, orig in self._index.items() if q in norm or norm in q]
        if cand:
            # Prefer the most-data team among loose matches.
            return max(cand, key=lambda t: self.factors[t]["weight"])
        return None

    def expected_goals(self, home: str, away: str):
        fh, fa = self.factors[home], self.factors[away]
        bl = self.baselines
        home_avg = bl["home_avg"]
        away_avg = bl["away_avg"]
        lam_home = home_avg * fh["atk"] * fa["def"]
        lam_away = away_avg * fa["atk"] * fh["def"]
        return max(lam_home, 1e-3), max(lam_away, 1e-3)

    def predict(self, home_name: str, away_name: str):
        if not self.ok:
            return None
        h = self.resolve(home_name)
        a = self.resolve(away_name)
        if h is None or a is None:
            return {"error": True, "home_found": h is not None, "away_found": a is not None}
        lam_h, lam_a = self.expected_goals(h, a)
        p_home, p_draw, p_away, best = _outcome_probs(lam_h, lam_a)
        return {
            "home": h, "away": a,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "exp_home": lam_h, "exp_away": lam_a,
            "likely_score": best,
        }

    def backtest(self, split: str = "2024-07-01") -> dict:
        """Train on matches before `split`, test on/after it; compare to the bookmaker."""
        if not self.ok:
            return {}
        cut = pd.Timestamp(split)
        train = self.matches[self.matches["date"] < cut]
        test = self.matches[self.matches["date"] >= cut]
        factors, baselines = _fit_factors(train, cut - pd.Timedelta(days=1))
        idx = {normalize_team_name(t): t for t in factors}

        model_ok = book_ok = total = book_total = 0
        for r in test.itertuples(index=False):
            h, a = factors.get(r.home), factors.get(r.away)
            if h is None or a is None:
                continue
            lam_h = baselines["home_avg"] * h["atk"] * a["def"]
            lam_a = baselines["away_avg"] * a["atk"] * h["def"]
            p_home, p_draw, p_away, _ = _outcome_probs(lam_h, lam_a)
            pick = "H" if p_home >= max(p_draw, p_away) else "A" if p_away >= p_draw else "D"
            total += 1
            if pick == r.ftr:
                model_ok += 1
            oh, od, oa = getattr(r, "odds_home", None), getattr(r, "odds_draw", None), getattr(r, "odds_away", None)
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                book_pick = "H" if oh <= min(od, oa) else "A" if oa <= od else "D"
                book_total += 1
                if book_pick == r.ftr:
                    book_ok += 1
        return {
            "test_matches": total,
            "model_acc": model_ok / total if total else 0.0,
            "book_acc": book_ok / book_total if book_total else 0.0,
        }


if __name__ == "__main__":
    m = ClubModel()
    print(f"[club_model] clubs: {len(m.factors)} | matches: {len(m.matches)}")
    bt = m.backtest()
    print(f"[club_model] backtest (since 2024-07): {bt['test_matches']} matches | "
          f"model {bt['model_acc']:.0%} vs bookmaker {bt['book_acc']:.0%}")
    for h, a in [("Man City", "Arsenal"), ("Real Madrid", "Barcelona"), ("Bayern Munich", "Dortmund")]:
        p = m.predict(h, a)
        print(f"\n{h} vs {a}: {p['home']} {p['p_home']:.0%} | draw {p['p_draw']:.0%} | "
              f"{p['away']} {p['p_away']:.0%} | xG {p['exp_home']:.2f}-{p['exp_away']:.2f} "
              f"| likely {p['likely_score'][0]}-{p['likely_score'][1]}")
