"""
match_predictor.py
Runtime wrapper around the trained Logistic Regression match predictor.

Loads predictor_model.pkl (built by train_predictor.py) and turns a head-to-head
question into a calibrated outcome distribution, a consistent scoreline, and the
real driving features for the reasoning section. National-team focused; the
strength table covers every nation in the player data, with the 48 World Cup 2026
teams using their official squads.
"""

from __future__ import annotations

import math
import pickle

import numpy as np

from team_strength import (FEATURES, canonical_nation, match_diff, goal_feats)

MODEL_PKL = "predictor_model.pkl"
_MAX_G = 7


def _poisson_dist(xg_a: float, xg_b: float, max_g: int = _MAX_G) -> dict:
    dist = {}
    for h in range(max_g + 1):
        ph = math.exp(-xg_a) * xg_a ** h / math.factorial(h)
        for a in range(max_g + 1):
            pa = math.exp(-xg_b) * xg_b ** a / math.factorial(a)
            dist[(h, a)] = ph * pa
    return dist


def _scoreline_for_outcome(xg_a: float, xg_b: float, outcome: str) -> tuple[int, int]:
    """Most probable scoreline CONSISTENT with the predicted outcome — so the
    headline result and the scoreline never contradict each other."""
    dist = _poisson_dist(xg_a, xg_b)
    def ok(h, a):
        if outcome == "win":  return h > a
        if outcome == "loss": return h < a
        return h == a
    cands = [(k, p) for k, p in dist.items() if ok(*k)]
    if not cands:
        return (1, 1)
    return max(cands, key=lambda kp: kp[1])[0]


class MatchPredictor:
    def __init__(self, path: str = MODEL_PKL):
        with open(path, "rb") as f:
            b = pickle.load(f)
        self.clf = b["clf"]
        self.goal = b["goal"]
        self.table = b["strength_table"]
        self.classes = b["classes"]
        self.trained_at = b.get("trained_at", "")
        self.n_train = b.get("n_train", 0)
        self._idx = set(self.table.index)

    # ── feature lookup ────────────────────────────────────────────────────────
    def has_team(self, team: str) -> bool:
        return canonical_nation(team, self._idx) in self._idx

    def _feat(self, team: str):
        key = canonical_nation(team, self._idx)
        if key is None or key not in self._idx:
            return None, team, None
        row = self.table.loc[key]
        vec = np.array([float(row[f]) for f in FEATURES], dtype=float)
        return vec, key, row

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, team1: str, team2: str) -> dict | None:
        fa, na, ra = self._feat(team1)
        fb, nb, rb = self._feat(team2)
        if fa is None or fb is None:
            return None

        proba = self.clf.predict_proba([match_diff(fa, fb)])[0]
        cls = self.classes
        p_win = float(proba[cls.index("H")])
        p_draw = float(proba[cls.index("D")])
        p_loss = float(proba[cls.index("A")])

        xg_a = float(max(0.2, self.goal.predict([goal_feats(fa, fb)])[0]))
        xg_b = float(max(0.2, self.goal.predict([goal_feats(fb, fa)])[0]))

        best = max((("win", p_win), ("draw", p_draw), ("loss", p_loss)), key=lambda kv: kv[1])
        outcome = best[0]
        score = _scoreline_for_outcome(xg_a, xg_b, outcome)
        if outcome == "win":
            label = f"{team1} win"
        elif outcome == "loss":
            label = f"{team2} win"
        else:
            label = "Draw"

        return {
            "team_a": team1, "team_b": team2,
            "resolved_a": na, "resolved_b": nb,
            "p_win": p_win, "p_draw": p_draw, "p_loss": p_loss,
            "xg_a": xg_a, "xg_b": xg_b,
            "score": (int(score[0]), int(score[1])),
            "outcome": outcome, "outcome_label": label,
            "confidence": self._confidence(p_win, p_draw, p_loss),
            "factors": self._factors(team1, team2, ra, rb),
            "trained_at": self.trained_at, "n_train": self.n_train,
        }

    @staticmethod
    def _confidence(p_win, p_draw, p_loss) -> str:
        top = max(p_win, p_draw, p_loss)
        gap = top - sorted([p_win, p_draw, p_loss])[-2]
        if gap > 0.25: return "High"
        if gap > 0.12: return "Medium"
        return "Low"

    @staticmethod
    def _factors(a, b, ra, rb) -> list[str]:
        """Human-readable driving features for the reasoning section."""
        if ra is None or rb is None:
            return []
        f = []
        va = math.expm1(ra["value_xi_log"]); vb = math.expm1(rb["value_xi_log"])
        if vb > 0 and va > 0:
            ratio = va / vb
            if ratio >= 1.15:
                f.append(f"{a}'s starting-XI market value is {ratio:.1f}× {b}'s")
            elif ratio <= 0.87:
                f.append(f"{b}'s starting-XI market value is {1/ratio:.1f}× {a}'s")
        dr = ra["rating_mean"] - rb["rating_mean"]
        if abs(dr) >= 0.8:
            hi, lo = (a, b) if dr > 0 else (b, a)
            f.append(f"{hi} has the higher average squad rating (+{abs(dr):.1f})")
        da = ra["attack"] - rb["defense"]
        db = rb["attack"] - ra["defense"]
        if da - db >= 2:
            f.append(f"{a}'s attack outmatches {b}'s defense")
        elif db - da >= 2:
            f.append(f"{b}'s attack outmatches {a}'s defense")
        if not f:
            f.append("Squad-strength metrics are close — expect a tight match")
        return f[:4]


def format_prediction(p: dict) -> str:
    """Markdown card for a single head-to-head prediction (shared by the tools)."""
    a, b = p["team_a"], p["team_b"]
    sh, sa = p["score"]
    lines = [
        f"**Prediction: {a} vs {b}** (neutral ground)\n",
        f"Likely result: **{p['outcome_label']}** — predicted score {a} {sh}–{sa} {b}",
        f"Probabilities: {a} {round(p['p_win']*100)}% | draw {round(p['p_draw']*100)}% | "
        f"{b} {round(p['p_loss']*100)}%",
        f"Expected goals: {a} {p['xg_a']:.2f} — {b} {p['xg_b']:.2f} | confidence: **{p['confidence']}**",
        "",
        "**Why:**",
    ]
    for fct in p["factors"]:
        lines.append(f"- {fct}")
    lines.append("\n🔍 Method: Logistic Regression on squad-strength features (starting-XI market "
                 "value, average squad rating, attack vs defense), trained on historical "
                 "international match results.")
    return "\n".join(lines)
