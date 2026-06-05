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

## 7. Match Prediction

Tool: `predict_match`

The current implementation uses a softmax-style squad-strength model. After the
new data is organized, this should be revisited and either documented honestly as
softmax squad-strength prediction or replaced with a simple logistic-regression
model trained on historical match features.

Possible features:

- Recent win rate from `football-data.org`.
- Average goals scored.
- Average goals conceded.
- Squad market value.
- Squad FM current ability / potential ability aggregates.

Expected method line:

```text
Method: Logistic regression or softmax model on team-form and squad-strength features.
```

## Query To Model Map

| User query | Tool | Course model |
| --- | --- | --- |
| "Find players similar to X" | `find_similar_players` | Cosine similarity |
| "Find/top/best players by criteria" | `scout_players` | Content-based filtering |
| "What archetype is X?" | `get_player_archetype` | K-Means clustering |
| "Find anomalies" | `detect_anomalies` | Z-score |
| "Compare X and Y" | `compare_players_jaccard` | Jaccard similarity |
| "Predict X vs Y" | `predict_match` | Logistic/softmax prediction |ך
