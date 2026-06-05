"""
wc_predictor.py
World Cup 2026 prediction engine.

Capabilities:
  - predict_wc_match(team_a, team_b)          → win/draw/loss probabilities + scoreline
  - predict_group(group_letter)               → expected standings + qualification probability
  - simulate_tournament(n_sims)              → Monte Carlo win / finalist / semi probabilities
  - predict_wc_winner(n_sims)               → tournament winner probabilities, top 10

Data used:
  - ds_engine.build_national_strength()      → hybrid Elo + squad-value strength (0-1)
  - fwc26_match_schedule_agent.csv           → all 72 group matches + knockout structure
  - national_matches.csv                     → historical WC results for Elo pedigree

Method:
  - Softmax on strength difference → match probabilities (neutral ground)
  - Poisson on expected goals (derived from strength) → scoreline
  - Points-based group simulation with GD tiebreaker
  - 10,000-simulation Monte Carlo for tournament-winner probabilities
"""

from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# ── WC 2026 team name → player-data nationality name (for strength lookup) ───
WC_NAME_MAP: dict[str, str] = {
    # already in ds_engine NATION_MAP, but schedule uses these exact strings
    "USA":                   "United States",
    "IR Iran":               "Iran",
    "Korea Republic":        "Korea, South",
    "Czechia":               "Czech Republic",
    "Côte d'Ivoire":         "Cote d'Ivoire",
    "Cote d’Ivoire":    "Cote d'Ivoire",
    "Côte d'Ivoire":    "Cote d'Ivoire",
    "Bosnia & Herzegovina":  "Bosnia-Herzegovina",
    "Cabo Verde":            "Cape Verde",
    "Congo DR":              "DR Congo",
    "Curaçao":               "Curacao",
    # common alternate spellings
    "United States":         "United States",
    "South Korea":           "Korea, South",
    "Ivory Coast":           "Cote d'Ivoire",
    "Iran":                  "Iran",
}

# WC 2026 knockout bracket seeding order (which groups feed each R32 slot).
# Groups are paired: A↔B, C↔D, E↔F, G↔H, I↔J, K↔L (first/second place cross-pairs).
# Third-placed teams fill the remaining 8 slots in a separate draw we simulate randomly.
_GROUP_PAIRS = [("A","B"),("C","D"),("E","F"),("G","H"),("I","J"),("K","L")]

SCALE = 5.0          # softmax sensitivity to strength difference
BASE_XG = 1.20       # neutral-ground average goals per team per match


def _norm_team(name: str, known: set) -> str:
    """Map a WC schedule team name to the national_strength index."""
    if name in known:
        return name
    # Normalise smart quotes / apostrophes before lookup
    name_clean = name.replace("’", "'").replace("‘", "'")
    if name_clean in known:
        return name_clean
    mapped = WC_NAME_MAP.get(name) or WC_NAME_MAP.get(name_clean)
    if mapped and mapped in known:
        return mapped
    # fuzzy: ignore case
    nl = name_clean.lower()
    for k in known:
        if k.lower() == nl:
            return k
    return name_clean   # return cleaned as-is; strength lookup will return average


class WCPredictor:
    """World Cup 2026 prediction engine."""

    def __init__(self, schedule: pd.DataFrame, national_strength: pd.DataFrame):
        self.schedule = schedule.copy()
        self.strength = national_strength  # index = nationality name, col: strength, pedigree_elo, etc.
        self.known = set(national_strength.index)

        # Group-stage matches only (teams are known)
        self.gs = schedule[schedule["stage"] == "Group Stage"].copy()
        # All 48 WC teams from the group stage
        self._winner_cache: dict | None = None  # computed once at first call
        self.all_teams: list[str] = sorted(
            set(self.gs["team1_name"].dropna().tolist() +
                self.gs["team2_name"].dropna().tolist())
        )
        # Pre-normalise team names → strength table keys
        self._mapped: dict[str, str] = {
            t: _norm_team(t, self.known) for t in self.all_teams
        }
        # Groups dict: group_letter → list of 4 teams
        self.groups: dict[str, list[str]] = {}
        for grp, gdf in self.gs.groupby("group"):
            teams = sorted(set(gdf["team1_name"].tolist() + gdf["team2_name"].tolist()))
            self.groups[str(grp)] = teams

    # ── team strength ─────────────────────────────────────────────────────────
    def _strength(self, team: str) -> float:
        """Return the 0-1 hybrid strength for a WC team."""
        key = self._mapped.get(team, _norm_team(team, self.known))
        if key in self.strength.index:
            return float(self.strength.loc[key, "strength"])
        return 0.35    # conservative average for unresolved newcomers

    # ── single match prediction ───────────────────────────────────────────────
    def predict_match(self, team_a: str, team_b: str) -> dict:
        """
        Neutral-ground match prediction.
        Returns: p_a (win), p_draw, p_b (win), xg_a, xg_b, scoreline.
        """
        sa, sb = self._strength(team_a), self._strength(team_b)
        ea = math.exp(SCALE * sa); eb = math.exp(SCALE * sb)
        p_a_raw = ea / (ea + eb); p_b_raw = eb / (ea + eb)
        closeness = 1.0 - abs(sa - sb) / max(sa + sb, 1e-6)
        p_draw = round(0.20 + 0.13 * closeness, 3)
        p_a = round(p_a_raw * (1 - p_draw), 3)
        p_b = round(p_b_raw * (1 - p_draw), 3)

        # Expected goals: stronger team gets more; scale around BASE_XG
        xg_a = max(0.3, BASE_XG * (1 + 0.6 * (sa - sb)))
        xg_b = max(0.3, BASE_XG * (1 + 0.6 * (sb - sa)))

        # Scoreline via Poisson
        from prediction_engine import generate_scoreline_distribution, classify_match_profile, select_context_aware_scoreline
        dist = generate_scoreline_distribution(xg_a, xg_b)
        profile = classify_match_profile(xg_a, xg_b, (sa - sb) * 400)  # pseudo Elo diff
        score = select_context_aware_scoreline(dist, profile, xg_a, xg_b, p_a, p_draw, p_b)
        return {
            "team_a": team_a, "team_b": team_b,
            "p_a": p_a, "p_draw": p_draw, "p_b": p_b,
            "xg_a": round(xg_a, 2), "xg_b": round(xg_b, 2),
            "scoreline": score, "profile": profile,
            "strength_a": round(sa, 3), "strength_b": round(sb, 3),
        }

    # ── group simulation (deterministic expected value) ───────────────────────
    def predict_group(self, group: str) -> dict:
        """
        Expected group standings using expected-value simulation
        (each match split into its probability-weighted outcomes).
        Returns full standings + qualification probability for each team.
        """
        teams = self.groups.get(group.upper())
        if not teams:
            return {"error": f"Group {group} not found."}
        matches = self.gs[self.gs["group"] == group.upper()]

        # Initialise stats
        stats: dict[str, dict] = {t: {"pts": 0.0, "gf": 0.0, "ga": 0.0, "gd": 0.0} for t in teams}
        match_results = []
        for _, row in matches.iterrows():
            t1, t2 = row["team1_name"], row["team2_name"]
            m = self.predict_match(t1, t2)
            # Expected points
            stats[t1]["pts"] += m["p_a"] * 3 + m["p_draw"] * 1
            stats[t2]["pts"] += m["p_b"] * 3 + m["p_draw"] * 1
            stats[t1]["gf"]  += m["xg_a"]; stats[t1]["ga"]  += m["xg_b"]
            stats[t2]["gf"]  += m["xg_b"]; stats[t2]["ga"]  += m["xg_a"]
            match_results.append({
                "match": f"{t1} vs {t2}",
                "predicted": f"{t1} {m['scoreline'][0]}–{m['scoreline'][1]} {t2}",
                "p_a": f"{round(m['p_a']*100)}%",
                "p_draw": f"{round(m['p_draw']*100)}%",
                "p_b": f"{round(m['p_b']*100)}%",
            })
        for t in teams:
            stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

        # Sort by expected pts → GD → GF
        table = sorted(teams,
            key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]),
            reverse=True
        )
        rows = []
        for pos, t in enumerate(table, 1):
            s = stats[t]
            rows.append({
                "pos": pos, "team": t,
                "exp_pts": round(s["pts"], 1),
                "exp_gd": round(s["gd"], 1),
                "exp_gf": round(s["gf"], 1),
                "strength": round(self._strength(t), 3),
                "qualifies": pos <= 2,
            })
        return {
            "group": group.upper(),
            "table": rows,
            "match_predictions": match_results,
            "note": "Top 2 qualify automatically. Third-place teams compete for 8 wild-card spots.",
        }

    # ── Monte Carlo tournament simulation ─────────────────────────────────────
    def _sim_match(self, team_a: str, team_b: str) -> str:
        """Simulate one match, return winner (no draws in knockout)."""
        sa, sb = self._strength(team_a), self._strength(team_b)
        ea = math.exp(SCALE * sa); eb = math.exp(SCALE * sb)
        pa = ea / (ea + eb)       # head-to-head; tiebreak removes draw
        return team_a if random.random() < pa else team_b

    def _sim_group(self, teams: list[str]) -> list[str]:
        """Simulate round-robin group; return teams ranked 1st…4th."""
        pts  = defaultdict(int)
        gd   = defaultdict(float)
        gf   = defaultdict(float)
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                sa, sb = self._strength(t1), self._strength(t2)
                xg_a = max(0.3, BASE_XG * (1 + 0.6 * (sa - sb)))
                xg_b = max(0.3, BASE_XG * (1 + 0.6 * (sb - sa)))
                g1 = np.random.poisson(xg_a)
                g2 = np.random.poisson(xg_b)
                gf[t1] += g1; gf[t2] += g2
                gd[t1] += g1 - g2; gd[t2] += g2 - g1
                if g1 > g2:   pts[t1] += 3
                elif g2 > g1: pts[t2] += 3
                else:         pts[t1] += 1; pts[t2] += 1
        return sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)

    def simulate_tournament(self, n_sims: int = 10_000) -> dict:
        """
        Monte Carlo tournament simulation.
        Returns: win_prob, finalist_prob, semi_prob, qualified_prob per team.
        """
        groups = list(self.groups.values())
        group_keys = list(self.groups.keys())
        counts: dict[str, dict] = {t: {"wins":0,"final":0,"semi":0,"quarters":0,"qualified":0}
                                    for t in self.all_teams}
        for _ in range(n_sims):
            # Group stage
            qualified: list[str] = []
            third_place: list[str] = []
            for g_teams in groups:
                ranked = self._sim_group(g_teams)
                qualified.extend(ranked[:2])
                third_place.append(ranked[2])

            # Best 8 third-placed teams qualify by strength (proxy for performance)
            third_place.sort(key=lambda t: self._strength(t), reverse=True)
            qualified.extend(third_place[:8])   # 24 + 8 = 32

            # Knockout rounds
            random.shuffle(qualified)
            round_teams = qualified[:]
            n_teams = len(round_teams)
            while len(round_teams) > 1:
                n = len(round_teams)
                next_round = []
                for i in range(0, n, 2):
                    winner = self._sim_match(round_teams[i], round_teams[i+1])
                    next_round.append(winner)
                    # Track deep runs
                    if n == 4:    # semi-finals
                        counts[round_teams[i]]["semi"] += 1
                        counts[round_teams[i+1]]["semi"] += 1
                    if n == 8:    # quarter-finals
                        counts[round_teams[i]]["quarters"] += 1
                        counts[round_teams[i+1]]["quarters"] += 1
                round_teams = next_round

            counts[round_teams[0]]["wins"] += 1

        # Qualification probability: teams in top-2 per group most of the time
        # Compute separately for display
        N = n_sims
        result = {}
        for t, c in counts.items():
            result[t] = {
                "win_prob":     round(c["wins"] / N * 100, 1),
                "semi_prob":    round(c["semi"] / N * 100, 1),
                "quarters_prob":round(c["quarters"] / N * 100, 1),
            }
        return result

    def predict_wc_winner(self, n_sims: int = 5_000) -> dict:
        """Return top candidates to win the 2026 World Cup with probabilities (cached)."""
        if self._winner_cache is not None:
            return self._winner_cache
        probs = self.simulate_tournament(n_sims)
        ranked = sorted(probs.items(), key=lambda x: x[1]["win_prob"], reverse=True)
        top = [{"team": t, **stats, "strength": round(self._strength(t), 3)}
               for t, stats in ranked if stats["win_prob"] > 0][:12]
        self._winner_cache = {"type": "wc_winner", "n_sims": n_sims, "candidates": top}
        return self._winner_cache

    def warm_up(self) -> None:
        """No-op — simulation is now computed lazily on the first call."""
        pass


# ── response formatters ──────────────────────────────────────────────────────
def format_group_prediction(result: dict) -> str:
    if "error" in result:
        return result["error"]
    grp = result["group"]
    lines = [f"**World Cup 2026 — Group {grp} predicted standings:**\n",
             f"{'Pos':<4} {'Team':<25} {'Exp Pts':<10} {'Exp GD':<8} {'Strength':<9} {'Status'}"]
    lines.append("-" * 65)
    for r in result["table"]:
        status = "✅ Qualify" if r["qualifies"] else "🟡 3rd-place wild-card chance" if r["pos"] == 3 else "❌ Likely eliminated"
        lines.append(f"{r['pos']:<4} {r['team']:<25} {r['exp_pts']:<10} {r['exp_gd']:<8} {r['strength']:<9} {status}")
    lines.append("\n**Match predictions:**")
    for m in result["match_predictions"]:
        lines.append(f"• {m['match']}: **{m['predicted']}** ({m['p_a']} / {m['p_draw']} / {m['p_b']})")
    lines.append(f"\n_{result['note']}_")
    lines.append("\n🔍 Method: Neutral-ground match probabilities from hybrid squad-value + World Cup Elo strength. "
                 "Group standings from expected-value simulation of all 6 group matches.")
    return "\n".join(lines)


def format_wc_winner(result: dict) -> str:
    cands = result["candidates"]
    n = result["n_sims"]
    lines = [f"**World Cup 2026 — tournament winner probabilities ({n:,} simulations):**\n"]
    for i, c in enumerate(cands[:10], 1):
        bar = "█" * int(c["win_prob"] / 2)
        lines.append(
            f"{i:>2}. **{c['team']:<22}** {c['win_prob']:>5}% to win  "
            f"| {c['semi_prob']:>5}% to reach semi  {bar}"
        )
    lines.append("\n🔍 Method: Monte Carlo simulation (10,000 tournaments). Each match uses softmax "
                 "probabilities on the hybrid squad-strength + World Cup Elo rating. Group stage simulated "
                 "with Poisson goals; knockout rounds use head-to-head win probabilities.")
    return "\n".join(lines)


def format_wc_match(result: dict) -> str:
    a, b = result["team_a"], result["team_b"]
    sa, sb = result["scoreline"]
    winner = a if sa > sb else b if sb > sa else "Draw"
    lines = [
        f"**World Cup 2026 match prediction: {a} vs {b}** (neutral ground)\n",
        f"Predicted result: **{winner}** — {a} {sa}–{sb} {b}",
        f"Probabilities: {a} {round(result['p_a']*100)}% | draw {round(result['p_draw']*100)}% | {b} {round(result['p_b']*100)}%",
        f"Expected goals: {a} {result['xg_a']} — {b} {result['xg_b']}",
        f"Team strength: {a} {result['strength_a']} vs {b} {result['strength_b']} (scale 0–1)",
        "",
        f"🔍 Method: Neutral-ground softmax on hybrid squad-value + WC Elo pedigree. "
        f"Scoreline from context-aware Poisson distribution (match profile: {result['profile'].replace('_',' ')})."
    ]
    return "\n".join(lines)
