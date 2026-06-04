# ScoutAI - Football Scout & Predictor Agent

ScoutAI is a football data-science web application with a chatbot interface.
Users can ask football questions in English or Hebrew, while the internal data
pipeline and model explanations stay in English for consistency and easier grading.

The current priority is rebuilding the data layer correctly before expanding the
agent.

## Current Goal

Build one clean, defensible player database from multiple sources:

- `football-data.org` API for current competitions, standings, fixtures, and results.
- Kaggle/Transfermarkt-style CSV files for player identity, club, nationality,
  market value, appearances, and real performance stats.
- Football Manager CSV exports for scouting-style player attributes such as pace,
  finishing, passing, tackling, decisions, current ability, and potential ability.

Football Manager data is especially useful for attribute-based player comparison,
but it should be documented as an expert/scouting attribute dataset rather than as
official match-event data.

## Data-Science Requirements

The project should use course models clearly and defensibly:

| Course model | Project usage |
| --- | --- |
| Cosine similarity | Similar players from normalized numeric player vectors |
| Euclidean distance | Optional distance view in the same feature space |
| K-Means clustering | Player archetypes/role profiles |
| Content-based recommendation | Scout recommendations after filters |
| Z-score | Anomaly detection inside each K-Means cluster |
| Jaccard similarity | Categorical trait comparison |
| Logistic/softmax prediction | Match prediction from team strength or recent-form features |

Every agent tool should end with a `Method:` line explaining the exact model used.

## Planned Data Pipeline

1. Inspect every uploaded CSV/XLSX schema before modeling.
2. Normalize player/team names and create source-priority rules.
3. Match player records across real-world CSVs, Football Manager exports, and API context.
4. Build a single `players_master.csv`.
5. Engineer numeric features such as per-90 stats, market-value transforms, and FM attributes.
6. Fill missing numeric values by position-group median.
7. Scale all model features with `StandardScaler`.
8. Build `player_features.npy`, K-Means clusters, archetype labels, and metadata.
9. Validate results with direct tool tests before connecting the chatbot.

## Runtime App

```bash
pip install -r requirements.txt
python app.py
```

Local URL:

```text
http://127.0.0.1:5000
```

Environment variables:

```text
GEMINI_API_KEY=...
FOOTBALL_DATA_API_KEY=...
```

Optional debugging flag:

```text
SCOUTAI_ENABLE_SMART_QA=1
```

By default, the app routes common football questions directly to deterministic
data-science tools. The older `football_qa.py` smart intent pipeline is disabled
unless this flag is enabled, because it can mix ranking logic and make demo
answers harder to explain.

## Important Note

Do not polish the agent before the data is stable. The next serious milestone is
data collection, cleaning, matching, and model preparation.
