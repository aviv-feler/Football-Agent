"""
train_predictor.py
Train the FOOTBOT match-outcome predictor (course model: Logistic Regression).

Pipeline:
  1. Build per-nation squad-strength features (team_strength.TeamStrength) from the
     Kaggle player data + the official World Cup 2026 squads.
  2. Build a training set from historical international results (national_matches.csv):
     features = strength DIFFERENCES between the two teams, label = H / D / A.
  3. Train a multiclass Logistic Regression for the outcome and a Poisson regression
     for expected goals (scoreline).
  4. Save everything to predictor_model.pkl and print validation predictions.

Run:  python train_predictor.py
"""

from __future__ import annotations

import pickle
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from team_strength import (TeamStrength, FEATURES, DIFF_FEATURES, GOAL_FEATURES,
                           match_diff, goal_feats, PLAYERS_CSV, SQUADS_CSV)

NATIONAL_MATCHES_CSV = "data/national_matches.csv"
MODEL_PKL = "predictor_model.pkl"


def build_training_set(matches: pd.DataFrame, ts: TeamStrength):
    """Return X_diff, y, and per-side goal data from historical matches."""
    X, y = [], []
    Xg, yg = [], []          # goals model: one sample per team per match
    skipped = 0
    table_idx = set(ts.table.index)
    for r in matches.itertuples(index=False):
        home, away = r.home, r.away
        if home not in table_idx or away not in table_idx:
            skipped += 1
            continue
        fa = ts.table.loc[home, FEATURES].astype(float).values
        fb = ts.table.loc[away, FEATURES].astype(float).values
        X.append(match_diff(fa, fb))
        y.append(r.result)                      # 'H' / 'D' / 'A'
        # goals model (both directions)
        if not (np.isnan(r.home_goals) or np.isnan(r.away_goals)):
            Xg.append(goal_feats(fa, fb)); yg.append(float(r.home_goals))
            Xg.append(goal_feats(fb, fa)); yg.append(float(r.away_goals))
    return (np.array(X, float), np.array(y), np.array(Xg, float),
            np.array(yg, float), skipped)


def main():
    print("[train] Loading data...", flush=True)
    players = pd.read_csv(PLAYERS_CSV, low_memory=False)
    squads = pd.read_csv(SQUADS_CSV)
    matches = pd.read_csv(NATIONAL_MATCHES_CSV)
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")

    print("[train] Building squad-strength features...", flush=True)
    ts = TeamStrength(players, squads)
    n_squad = (ts.table["source"] == "squad").sum()
    print(f"[train] Strength table: {len(ts.table)} nations "
          f"({n_squad} from official 2026 squads, rest from player pool).", flush=True)

    X, y, Xg, yg, skipped = build_training_set(matches, ts)
    print(f"[train] Training matches: {len(X)} (skipped {skipped} with unknown teams). "
          f"Goal samples: {len(Xg)}.", flush=True)
    print(f"[train] Label balance: {dict(pd.Series(y).value_counts())}", flush=True)

    # ── outcome model: multiclass Logistic Regression ────────────────────────
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.7,
                                           class_weight="balanced"))
    clf.fit(X, y)
    try:
        cv = cross_val_score(clf, X, y, cv=5).mean()
        print(f"[train] LogReg 5-fold accuracy: {cv:.1%} "
              f"(baseline majority {pd.Series(y).value_counts(normalize=True).max():.1%})", flush=True)
    except Exception as e:
        print(f"[train] CV skipped: {e}", flush=True)

    # ── goals model: Poisson regression ──────────────────────────────────────
    goal = make_pipeline(StandardScaler(),
                         PoissonRegressor(alpha=0.01, max_iter=1000))
    goal.fit(Xg, yg)

    bundle = {
        "clf": clf,
        "goal": goal,
        "strength_table": ts.table,
        "features": FEATURES,
        "diff_features": DIFF_FEATURES,
        "goal_features": GOAL_FEATURES,
        "classes": list(clf.named_steps["logisticregression"].classes_),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X)),
        "n_nations": int(len(ts.table)),
    }
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] Saved {MODEL_PKL} (classes={bundle['classes']}).", flush=True)

    # ── validation: differentiated, realistic predictions? ───────────────────
    from match_predictor import MatchPredictor
    mp = MatchPredictor(MODEL_PKL)
    print("\n[train] === Validation predictions ===")
    for a, b in [("Brazil", "Argentina"), ("France", "Canada"),
                 ("Spain", "Morocco"), ("Germany", "Japan")]:
        p = mp.predict(a, b)
        print(f"  {a} vs {b}: {a} {p['p_win']:.0%} / draw {p['p_draw']:.0%} / "
              f"{b} {p['p_loss']:.0%}  ->  {p['outcome_label']}  "
              f"score {p['score'][0]}-{p['score'][1]}  (xG {p['xg_a']:.2f}-{p['xg_b']:.2f})")


if __name__ == "__main__":
    main()
