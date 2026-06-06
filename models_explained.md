# ScoutAI - Data Science Models Explained

This document is the academic explanation layer for the project. It should stay
aligned with the actual code and with the course models used in the live demo.

## 1. Feature Matrix

Each player is represented as a numeric feature vector. The final feature set will
be rebuilt after the new CSV files and Football Manager dataset are uploaded.

Expected feature groups:

- Real performance stats: goals per 90, assists per 90, minutes, appearances,
  cards, and other available match-performance columns.
- Player profile stats: age, height, market value, international caps.
- Football Manager attributes: pace, acceleration, finishing, passing, technique,
  tackling, positioning, decisions, work rate, current ability, potential ability,
  and other available attribute columns.

Missing numeric values should be filled by position-group median. All modeling
features must be normalized with `StandardScaler`.

Formula:

```text
z = (x - mean) / standard_deviation
```

This is critical because otherwise large-scale features such as market value or
current ability can dominate smaller-scale features such as goals per 90.

## 2. Similar Players - Cosine Similarity

Tool: `find_similar_players`

Players are compared as normalized numeric vectors.

Formula:

```text
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
```

Why this model: cosine similarity compares the direction of the player profile,
which is useful for style/attribute similarity. The player name is used only for
lookup, not for semantic text similarity.

Expected method line:

```text
Method: Cosine similarity on normalized numeric player features.
```

## 3. Scout Recommendations - Content-Based Filtering

Tool: `scout_players`

The natural-language query is parsed into filters such as position, age, league,
nationality, budget, or role. After hard filters, candidates are ranked by
similarity to an ideal target vector.

Why this model: we recommend players based on their feature profile, not on user
ratings. Collaborative filtering is not appropriate unless we later add user/team
preference histories.

Expected method line:

```text
Method: Content-based filtering with cosine similarity on normalized features.
```

## 4. Player Archetypes - K-Means Clustering

Tool: `get_player_archetype`

K-Means groups players into role/profile clusters using the normalized feature
matrix.

Objective:

```text
minimize sum of squared distances from each point to its assigned centroid
```

The number of clusters should be selected with the elbow method by testing several
values of k and choosing the point where inertia improvement begins to flatten.

Expected method line:

```text
Method: K-Means clustering (k=N, selected by the elbow method).
```

## 5. Anomaly Detection - Z-Score Inside Cluster

Tool: `detect_anomalies`

For each player, compare the player's normalized feature values against the mean
and standard deviation of the player's K-Means cluster.

Formula:

```text
z_cluster = (player_feature - cluster_mean) / cluster_standard_deviation
```

Players with large absolute deviations on key features are flagged as
overperformers or underperformers.

Expected method line:

```text
Method: Z-score deviation from K-Means cluster centroid.
```

## 6. Categorical Comparison - Jaccard Similarity

Tool: `compare_players_jaccard`

Each player is converted into a set of categorical traits, such as position,
sub-position, nationality, league, foot, age bucket, value tier, and archetype.

Formula:

```text
J(A, B) = |A intersection B| / |A union B|
```

Why this model: Jaccard is appropriate for comparing sets of categorical traits,
while cosine is appropriate for numeric vectors.

Expected method line:

```text
Method: Jaccard similarity on categorical trait sets.
```

## 7. Match Prediction — Logistic Regression on Squad Strength

Tools: `predict_match`, `predict_wc_match` (and the landing-page featured widget).
Code: `team_strength.py`, `train_predictor.py`, `match_predictor.py`.

### Why we changed the model

The previous predictor leaned on *recent form* pulled from the live API. Most
national teams — especially smaller ones — have too few recent matches, so the
form features came back empty/similar and almost every match collapsed to a 1–1
draw. Only extreme mismatches produced a real result. We replaced this with a
**multiclass Logistic Regression** (a core course model) whose primary signal is
**squad strength**, which we have for every team.

### Step A — Team strength features (`team_strength.py`)

For each nation we compute interpretable strength features from the player data
(`players_clean.csv`). For the 48 World Cup 2026 teams we use the **official
called-up squad** (`world_cup_2026_squads.csv`) joined to the player data by name
(~83% match; the high-value players that drive strength match best, and gaps fall
back to position defaults). Other nations use their strongest players in the pool.

Per-team features (`FEATURES`):

- `value_xi_log` — log of the **starting-XI market value** (sum of the top-11 by value).
- `value_mean_log` — log of the mean market value of the top-23 (squad depth of value).
- `rating_mean` — average EA FC rating of the best 16 players.
- `attack` — mean shooting of the top-5 attackers.
- `defense` — mean defending of the top-5 defenders.

### Step B — Training set + model (`train_predictor.py`)

Training labels come from historical international results
(`national_matches.csv`, 250 usable World Cup matches, neutral ground). For each
match the features are the **differences** between the two teams (`DIFF_FEATURES`):

```text
d_value_xi   = A.value_xi_log   - B.value_xi_log
d_value_mean = A.value_mean_log - B.value_mean_log
d_rating     = A.rating_mean    - B.rating_mean
d_att_def    = A.attack         - B.defense
d_def_att    = A.defense        - B.attack
```

Pipeline: `StandardScaler` → `LogisticRegression(multi_class, class_weight="balanced")`,
labels `H / D / A`. A separate `PoissonRegressor` predicts each team's expected
goals (own attack/value vs opponent defense + rating edge) for the scoreline.
The bundle is saved to `predictor_model.pkl` and loaded once at startup.

5-fold CV accuracy ≈ 47% vs a 42% majority baseline (3-class, high-variance domain).

### Step C — Prediction (`match_predictor.py`)

`predict(team1, team2)` builds both strength vectors, takes the differences, runs
the Logistic Regression for `P(win) / P(draw) / P(loss)`, and the Poisson model for
expected goals. The scoreline is the most probable one **consistent with the
predicted outcome**, so the headline result and the score never contradict (and a
draw outcome always shows an equal scoreline → fixes the "1–1 but winner = X" bug).
The reasoning cites the real driving features (value ratio, rating gap, attack vs
defense).

### Step D — Missing data

Every nation in the player pool resolves to a strength vector; truly unknown teams
fall back to neutral defaults and the answer says so. The model never collapses to
a blanket 1–1 — predictions are differentiated by real strength gaps.

### Validation (printed by `python train_predictor.py`)

```text
Brazil vs Argentina : 27% / 41% / 31%  -> Draw        1-1   (close, slight edge Argentina)
France vs Canada    : 53% / 29% / 18%  -> France win  1-0
Spain  vs Morocco   : 38% / 33% / 28%  -> Spain win   1-0   (favored, not a blowout)
Germany vs Japan    : 47% / 33% / 21%  -> Germany win 1-0   (Japan competitive)
```

### Method line

```text
🔍 Method: Logistic Regression on squad-strength features (starting-XI market value,
average squad rating, attack vs defense), trained on historical international results.
```

## Query To Model Map

| User query | Tool | Course model |
| --- | --- | --- |
| "Find players similar to X" | `find_similar_players` | Cosine similarity |
| "Find/top/best players by criteria" | `scout_players` | Content-based filtering |
| "What archetype is X?" | `get_player_archetype` | K-Means clustering |
| "Find anomalies" | `detect_anomalies` | Z-score |
| "Compare X and Y" | `compare_players_jaccard` | Jaccard similarity |
| "Predict X vs Y" | `predict_match` / `predict_wc_match` | Logistic Regression (squad strength) |
| "Who is in X's squad?" | `get_national_squad` | Official 2026 squad lookup (data, not model) |
