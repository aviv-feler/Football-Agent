# ScoutAI - Football Scout & Predictor Agent

ScoutAI is a Flask web app with a LangChain/Gemini agent for football scouting,
player comparison, match prediction, live standings, and FIFA World Cup 2026 schedule lookup.

The project is built around precomputed data-science artifacts in `data/` and exposes them
through agent tools. The LLM is used for routing and explanation, while player facts and
rankings come from the local tools.

## Features

- Player scouting from natural-language criteria.
- Similar-player search using normalized numeric performance vectors.
- Player archetypes using K-Means clustering.
- Anomaly detection using cluster-relative Z-scores.
- Categorical player comparison using Jaccard similarity.
- National-team match prediction using squad-strength features.
- Live standings and fixtures via football-data.org.
- FIFA World Cup 2026 fixture lookup from the included schedule CSV.

## Data Science Methods

| Method | Used for |
| --- | --- |
| K-Means clustering | Player archetypes / roles |
| Cosine similarity | Similar players and content-based scouting |
| Jaccard similarity | Categorical trait comparison |
| Cluster Z-score | Overperformer / underperformer detection |
| Squad-strength softmax | National-team match prediction |

## Agent Tools

1. `find_similar_players` - cosine similarity on normalized numeric player features.
2. `scout_players` - content-based filtering from parsed scouting criteria.
3. `get_player_archetype` - K-Means archetype and cluster explanation.
4. `detect_anomalies` - Z-score deviation from K-Means cluster statistics.
5. `compare_players_jaccard` - Jaccard similarity on categorical trait sets.
6. `predict_match` - national-team prediction from aggregated squad strength.
7. `get_live_standings` - live standings/fixtures from football-data.org.
8. `world_cup_info` - World Cup 2026 fixture lookup from the local CSV.

Every tool response is expected to include a `Method:` line so the UI and course grading
can show which data-science method was used.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Create a local `.env` file with:

```text
GEMINI_API_KEY=...
FOOTBALL_DATA_API_KEY=...
```

`FOOTBALL_DATA_API_KEY` is only needed for live standings. The local player tools and
World Cup schedule work from the included data files.

## Tool Smoke Test

```bash
python test_tools.py
```

Note: `test_tools.py` includes a live standings call, so that part needs network access
and a valid `FOOTBALL_DATA_API_KEY`.

## Data Files

The raw Kaggle datasets are not included because they are large. The repo includes the
runtime artifacts the app needs:

- `data/players_clean.csv`
- `data/player_features.npy`
- `data/feature_meta.json`
- `data/fwc26_match_schedule_agent.csv`

To rebuild the player artifacts from raw data, place the raw CSVs in `data/` and run:

```bash
python data_prep.py
```

## Deployment

Render configuration lives in `render.yaml`. Configure these environment variables in
the hosting dashboard:

- `GEMINI_API_KEY`
- `FOOTBALL_DATA_API_KEY`

The app also exposes `/healthz` for basic hosting health checks.
