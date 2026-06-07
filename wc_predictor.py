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

import re
import math
import random
import unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict


# ── squad name-matching helpers ───────────────────────────────────────────────
def _strip_accents(s: str) -> str:
    nf = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nf if not unicodedata.combining(c))


def _team_key(s: str) -> str:
    """Normalise a team name so squad and schedule spellings collapse to one key.
    'Bosnia & Herzegovina' / 'Bosnia And Herzegovina' → 'bosnia and herzegovina'."""
    s = _strip_accents(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _name_toks(s: str) -> list[str]:
    """Ordered, accent-stripped name tokens (len >= 2, so initials like 'J.' drop)."""
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if len(t) >= 2]


def _first_name_match(a: str, b: str) -> bool:
    """True if two first names match exactly or one is a prefix of the other
    (handles 'Nico' vs 'Nicolás', 'Alex' vs 'Alexander')."""
    if a == b:
        return True
    return len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a))

# ── WC 2026 team name → player-data nationality name (for strength lookup) ───
WC_NAME_MAP: dict[str, str] = {
    # already in ds_engine NATION_MAP, but schedule uses these exact strings
    "USA":                   "United States",
    "IR Iran":               "Iran",
    "Korea Republic":        "Korea, South",
    "Czechia":               "Czech Republic",
    "Côte d'Ivoire":         "Cote d'Ivoire",
    "Cote d’Ivoire":         "Cote d'Ivoire",
    "Côte d’Ivoire":         "Cote d'Ivoire",
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
        self._sim_cache: dict | None = None     # full per-team simulation results
        self._squad_index: dict | None = None   # team_key → list of squad attacker descriptors
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
        Returns: win_prob, semi_prob, quarters_prob, r16_prob, r32_prob per team.
        """
        groups = list(self.groups.values())
        counts: dict[str, dict] = {
            t: {"wins": 0, "semi": 0, "quarters": 0, "r16": 0, "r32": 0}
            for t in self.all_teams
        }
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
            while len(round_teams) > 1:
                n = len(round_teams)
                next_round = []
                for i in range(0, n, 2):
                    t1, t2 = round_teams[i], round_teams[i + 1]
                    winner = self._sim_match(t1, t2)
                    next_round.append(winner)
                    if n == 32:
                        counts[t1]["r32"] += 1
                        counts[t2]["r32"] += 1
                    elif n == 16:
                        counts[t1]["r16"] += 1
                        counts[t2]["r16"] += 1
                    elif n == 8:
                        counts[t1]["quarters"] += 1
                        counts[t2]["quarters"] += 1
                    elif n == 4:
                        counts[t1]["semi"] += 1
                        counts[t2]["semi"] += 1
                round_teams = next_round

            counts[round_teams[0]]["wins"] += 1

        N = n_sims
        result = {}
        for t, c in counts.items():
            result[t] = {
                "win_prob":      round(c["wins"]    / N * 100, 1),
                "semi_prob":     round(c["semi"]    / N * 100, 1),
                "quarters_prob": round(c["quarters"]/ N * 100, 1),
                "r16_prob":      round(c["r16"]     / N * 100, 1),
                "r32_prob":      round(c["r32"]     / N * 100, 1),
            }
        self._sim_cache = result
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

    def _build_squad_index(self, squads_df: "pd.DataFrame") -> dict:
        """Index the official 2026 squads by normalised team name.
        Only forwards/midfielders are kept as match targets (top-scorer candidates),
        which also removes goalkeeper/defender name collisions."""
        index: dict[str, list[dict]] = {}
        for _, r in squads_df.iterrows():
            if str(r.get("position", "")).upper() not in ("FW", "MF"):
                continue
            fn = _name_toks(r.get("first_names", ""))
            ln = _name_toks(r.get("last_names", ""))
            sh = _name_toks(r.get("name_on_shirt", ""))
            index.setdefault(_team_key(r.get("team", "")), []).append({
                "name":      str(r.get("player_name", "")),
                "first":     fn[0] if fn else "",
                "surnames":  set(ln),
                "shirt":     " ".join(sh),
            })
        return index

    @staticmethod
    def _match_squad(db_name: str, members: list[dict]) -> str | None:
        """Return the squad player_name a database player maps to, or None.
        Multi-token names match on (surname ∈ squad surnames) + (first-name match);
        single-token names (Brazilian mononyms) must equal a shirt or first name."""
        db = _name_toks(db_name)
        if not db:
            return None
        if len(db) == 1:
            for m in members:
                if db[0] == m["shirt"] or db[0] == m["first"]:
                    return m["name"]
            return None
        db_first, db_sur = db[0], db[-1]
        for m in members:
            if db_sur in m["surnames"] and _first_name_match(db_first, m["first"]):
                return m["name"]
        joined = " ".join(db)
        for m in members:
            if joined == m["shirt"]:
                return m["name"]
        return None

    def predict_wc_top_scorer(self, players_df: "pd.DataFrame",
                              squads_df: "pd.DataFrame | None" = None, n: int = 10) -> dict:
        """
        Project top Golden Boot candidates: player goal rate × expected WC games played.
        Expected games = 3 (group) + P(R32) + P(R16) + P(QF) + P(SF) + 2·P(win).
        When the official squad list is supplied, only players actually called up are
        considered (so retired/uncalled players like Benzema are excluded).
        """
        import pandas as _pd
        from prediction_engine import ATTACKER_SUBS

        if self._sim_cache is None:
            self.simulate_tournament(5_000)
        sim = self._sim_cache

        if squads_df is not None and self._squad_index is None:
            self._squad_index = self._build_squad_index(squads_df)
        squad_index = self._squad_index

        attack_subs = ATTACKER_SUBS | {"Centre-Forward"}
        has_minutes = "minutes_played" in players_df.columns

        rows = []
        for team in self.all_teams:
            probs = sim.get(team, {})
            e_games = (
                3
                + probs.get("r32_prob", 0) / 100
                + probs.get("r16_prob", 0) / 100
                + probs.get("quarters_prob", 0) / 100
                + probs.get("semi_prob", 0) / 100
                + probs.get("win_prob", 0) / 100 * 2
            )

            nat = self._mapped.get(team, team)
            nation_df = players_df[
                (players_df["nationality"] == nat) &
                players_df["sub_position"].isin(attack_subs)
            ].copy()
            if nation_df.empty:
                continue

            for col in ("goals_per90", "fc_shooting", "fc_overall", "goals", "minutes_played"):
                if col in nation_df.columns:
                    nation_df[col] = _pd.to_numeric(nation_df[col], errors="coerce").fillna(0)

            # The stored goals_per90 is unreliable (capped/defaulted to ~0.59 for top
            # scorers), so recompute the real rate from goals & minutes where possible.
            stored_g90 = nation_df.get("goals_per90", _pd.Series(0.0, index=nation_df.index))
            if has_minutes:
                mins = nation_df["minutes_played"]
                real_g90 = (nation_df["goals"] / mins.where(mins > 0, np.nan) * 90.0)
                g90 = real_g90.where(mins >= 300, stored_g90).fillna(0.0)
            else:
                g90 = stored_g90
            g90 = g90.clip(lower=0.0, upper=1.2)

            shooting = nation_df.get("fc_shooting", _pd.Series(70.0, index=nation_df.index))
            nation_df["wc_g90"] = g90
            nation_df["wc_proj"] = g90 * e_games * (shooting / 80.0)
            nation_df = nation_df.sort_values("wc_proj", ascending=False)

            members = squad_index.get(_team_key(team)) if squad_index is not None else None
            taken: set[str] = set()    # one DB row per squad player

            for _, r in nation_df.iterrows():
                if members is not None:
                    matched = self._match_squad(str(r.get("player_name", "")), members)
                    if matched is None or matched in taken:
                        continue
                    taken.add(matched)
                rows.append({
                    "player":         str(r.get("player_name", "Unknown")),
                    "team":           team,
                    "nationality":    nat,
                    "sub_position":   str(r.get("sub_position", "")),
                    "goals_per90":    round(float(r.get("wc_g90", 0)), 2),
                    "shooting":       int(r.get("fc_shooting", 0)),
                    "overall":        int(r.get("fc_overall", 0)),
                    "expected_games": round(e_games, 1),
                    "wc_proj_goals":  round(float(r.get("wc_proj", 0)), 2),
                })
                # Cap candidates per team so one nation can't flood the board.
                if sum(1 for x in rows if x["team"] == team) >= 4:
                    break

        rows.sort(key=lambda x: x["wc_proj_goals"], reverse=True)
        note = ("wc_proj_goals = goals_per90 × expected_games_played × shooting_quality_factor"
                + ("; candidates restricted to official 2026 squads" if squad_index else ""))
        return {"type": "wc_top_scorer", "candidates": rows[:n], "note": note}

    def warm_up(self) -> None:
        """Pre-compute the tournament simulation at startup so the first call is instant."""
        self.predict_wc_winner(5_000)


# ── response formatters ──────────────────────────────────────────────────────
def format_group_prediction(result: dict) -> str:
    if "error" in result:
        return result["error"]
    grp = result["group"]
    lines = [f"**Prediction: World Cup 2026 — Group {grp} predicted standings:**\n",
             f"{'Pos':<4} {'Team':<25} {'Exp Pts':<10} {'Exp GD':<8} {'Strength':<9} {'Status'}"]
    lines.append("-" * 65)
    for r in result["table"]:
        status = "✅ Qualify" if r["qualifies"] else "🟡 3rd-place wild-card chance" if r["pos"] == 3 else "❌ Likely eliminated"
        lines.append(f"{r['pos']:<4} {r['team']:<25} {r['exp_pts']:<10} {r['exp_gd']:<8} {r['strength']:<9} {status}")
    lines.append("\n**Match predictions:**")
    for m in result["match_predictions"]:
        lines.append(f"• {m['match']}: **{m['predicted']}** ({m['p_a']} / {m['p_draw']} / {m['p_b']})")
    lines.append(f"\n_{result['note']}_")
    lines.append("\n🔍 Method: Per-match win/draw/loss via a logistic (softmax) function on hybrid "
                 "squad-value + Elo strength; expected goals via Poisson. Group standings from an "
                 "expected-value simulation of all 6 group matches.")
    return "\n".join(lines)


def format_wc_winner(result: dict) -> str:
    cands = result["candidates"]
    n = result["n_sims"]
    lines = [f"**Prediction: World Cup 2026 — tournament winner probabilities ({n:,} simulations):**\n"]
    for i, c in enumerate(cands[:10], 1):
        bar = "█" * int(c["win_prob"] / 2)
        lines.append(
            f"{i:>2}. **{c['team']:<22}** {c['win_prob']:>5}% to win  "
            f"| {c['semi_prob']:>5}% to reach semi  {bar}"
        )
    lines.append(f"\n🔍 Method: Monte Carlo tournament simulation ({n:,} runs). Group goals via "
                 "Poisson distribution; knockout match outcomes via a logistic (softmax) function on "
                 "hybrid team strength = squad market value blended with Elo from historical WC results.")
    return "\n".join(lines)


def format_wc_top_scorer(result: dict) -> str:
    cands = result["candidates"]
    lines = ["**Prediction: World Cup 2026 — Golden Boot candidates (projected goals):**\n",
             f"{'#':<4} {'Player':<24} {'Team':<22} {'Pos':<20} {'G/90':<7} {'SHT':<6} {'xGames':<8} {'Proj WC Goals'}"]
    lines.append("-" * 95)
    for i, c in enumerate(cands, 1):
        lines.append(
            f"{i:<4} {c['player']:<24} {c['team']:<22} {c['sub_position']:<20} "
            f"{c['goals_per90']:<7} {c['shooting']:<6} {c['expected_games']:<8} {c['wc_proj_goals']:.1f}"
        )
    lines.append(f"\n_{result['note']}_")
    lines.append("\n🔍 Method: Player goal-rate (goals/90) from FBref + FC26 data × expected WC "
                 "games (Monte Carlo R32/R16/QF/SF/Final stage probabilities) × shooting quality. "
                 "Candidates restricted to official 2026 squads; ranked by projected tournament goals.")
    from viz import embed_viz
    viz = {"type": "ranking", "title": "World Cup 2026 — Golden Boot (projected)", "unit": "goals",
           "items": [{"name": c["player"], "value": round(float(c["wc_proj_goals"]), 1), "sub": c["team"]}
                     for c in cands[:6]]}
    return embed_viz("\n".join(lines), viz)


def format_wc_match(result: dict) -> str:
    a, b = result["team_a"], result["team_b"]
    sa, sb = result["scoreline"]
    winner = a if sa > sb else b if sb > sa else "Draw"
    lines = [
        f"**Prediction: World Cup 2026 match prediction: {a} vs {b}** (neutral ground)\n",
        f"Predicted result: **{winner}** — {a} {sa}–{sb} {b}",
        f"Probabilities: {a} {round(result['p_a']*100)}% | draw {round(result['p_draw']*100)}% | {b} {round(result['p_b']*100)}%",
        f"Expected goals: {a} {result['xg_a']} — {b} {result['xg_b']}",
        f"Team strength: {a} {result['strength_a']} vs {b} {result['strength_b']} (scale 0–1)",
        "",
        f"🔍 Method: Win/draw/loss via a logistic (softmax) function on hybrid team strength = squad "
        f"market value blended with Elo from historical WC results. Scoreline from context-aware "
        f"Poisson distribution (match profile: {result['profile'].replace('_',' ')})."
    ]
    return "\n".join(lines)
