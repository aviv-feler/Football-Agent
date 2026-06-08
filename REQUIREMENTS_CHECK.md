# FOOTBOT — Assignment Requirements Audit (honest)

_Audited against the actual code on branch `main`. Status legend: ✅ DONE · ⚠️ PARTIAL · ❌ MISSING._

> **Headline for the demo:** the data-science core is genuinely there and real
> (Jaccard, K-Means, Z-score anomaly, content-based scouting, weighted-similarity,
> and a **genuine Logistic Regression** match model). The two things that can bite
> you in the demo are (1) several *prediction* models are really Poisson / RandomForest /
> GradientBoosting / Monte-Carlo **relabeled as "Logistic Regression"** in the pills, and
> (2) there is currently **no off-domain refusal**. Details below.

---

## Core requirements

### 1. Free natural-language agent (Hebrew + English) — ✅ DONE
- LLM-first agent: `agent.py` `ScoutAgent.invoke` → `_run_tool_loop` lets the LLM (OpenAI
  GPT-5.5) read the question, choose tools, and narrate. Not a menu.
- Language handling: `_detect_language` (`agent.py`) detects Hebrew by Unicode range and adds a
  language directive; verified live — Hebrew and English both answer in-language.

### 2. At least 2 external data sources — ✅ DONE (well more than 2)
Runtime sources actually used:
1. **football-data.org API** (live) — `tools/get_live_standings.py`, `tools/get_top_scorers.py`,
   `app.py /api/wc-match`, `live_tables.py /api/league-tables`.
2. **Transfermarkt/Kaggle player data** → `data/players_clean.csv`, `data/player_profiles.csv`,
   `data/players.csv` (club/league/market value/age).
3. **FBref 2025-26 stats** (merged into `players_clean.csv`, `stats_source='fbref_2025_26'`).
4. **EA FC26 attributes** (`fc_pace/shooting/passing/...` columns in `players_clean.csv`).
5. **Official FIFA World Cup 2026 squads** → `data/world_cup_2026_squads.csv`.
6. **Historical match results** → `data/national_matches.csv`, `data/club_matches.csv`, `data/games.csv`.
7. **Precomputed DS artifacts** → `data/player_features.npy`, `data/feature_meta.json`.

### 3. Similarity model (real, not the LLM) — ✅ DONE
- **Jaccard — ✅ active & real.** `tools/compare_players_jaccard.py` → `ds_engine.jaccard()` on
  categorical trait sets (`ds_engine.trait_set`: position, sub_position, nationality, foot, league,
  age_bucket, value_tier, archetype). Example: *"Compare Mbappé and Vinicius Jr."*
- **Cosine similarity — ✅ active & real.** `tools/scouting_tools.py` `find_similar_player` →
  `scouting.calculate_weighted_cosine`: `cos = Σ w·a·b / (sqrt(Σ w·a²) · sqrt(Σ w·b²))` on
  normalized performance attributes + per-90 features, within the same position group. The
  pill shown to the user is "Cosine Similarity". Example: *"Find players similar to Bellingham."*
- **Weighted Euclidean — ✅ active & real (different feature).** The other scouting paths
  (`find_replacement`, `search_by_profile`, `find_wonderkids`) rank candidates against a
  synthetic ideal-target vector via `calculate_weighted_similarity`
  (`dist = sqrt(Σ w·(C−t)²)`, `sim = 1/(1+dist)`). These show the "Content-Based Filtering"
  pill because that's exactly what they do (content-based scout against a target profile).

### 4. Clustering (K-Means, real) — ✅ DONE (computed offline, used live)
- K-Means is run in the **data-prep pipeline** (`data_prep.py:298/382/455`, `data_manager.py:555`,
  `sklearn.cluster.KMeans`), producing the `cluster` + `archetype` columns in `players_clean.csv`
  and centroids in `feature_meta.json`.
- Used live in answers: `tools/get_player_archetype.py` (reports the player's archetype/role) and
  `ds_engine` per-cluster mean/std for anomaly detection. Example: *"What type of player is Mbappé?"*
- ⚠️ Honest note: the algorithm runs in prep, not per request (standard practice) — be ready to say so.

### 5. Recommendations / content-based scouting (real) — ✅ DONE
- `tools/scouting_tools.py`: `search_by_profile`, `find_wonderkids`, `find_replacement` →
  `scouting.py` weighted-similarity + multi-factor ranking (potential, current ability, age fit,
  data reliability). Real, not the LLM. Example: *"Find a young creative attacking midfielder with high potential."*

### 6. Anomaly detection (Z-score, real, exposed) — ✅ DONE
- `tools/detect_anomalies.py` → `ds_engine.zscores()` (`ds_engine.py:120`):
  `z = (X − cluster_mean) / cluster_std`, flags |z|>2 over/under-performers. Wired in `build_agent`
  and reachable by the user. Example: *"Show overperformers in the Premier League."*

### 7. NLP / LLM — ✅ DONE
- OpenAI via `langchain_openai.ChatOpenAI` (`agent.py`), with a model chain (`MODEL_CHAIN`,
  default `gpt-5.5` → `gpt-5-mini` → `gpt-4o-mini`) rotated on rate-limit/quota/bad-model
  errors, tool-calling + final NL generation in the user's language.

### 8. Domain-limited agent — ❌ MISSING (refusal) / ✅ (domain knowledge)
- Rich football-domain grounding: ✅ (all tools + datasets are football-only).
- **No off-domain refusal.** The current `SYSTEM_PROMPT` has **no rule** to decline non-football
  questions — and actually says *"Never say 'I cannot answer'"* (`agent.py:76`). The old hard guard
  ("I can only answer questions based on our football database") was removed in the FOOTBOT refactor.
  → It will likely answer "capital of France"-type questions. **Gap to fix before demo.**

### 9. Professor can run it — ✅ deployable / ⚠️ confirm it's actually live
- `render.yaml` present (gunicorn `app:app`, `OPENAI_API_KEY` + `OPENAI_MODEL` + `FOOTBALL_DATA_API_KEY` env vars);
  runtime data committed; `/healthz` endpoint exists. So it **deploys**.
- ⚠️ I cannot confirm a **live public URL** is currently up — verify the Render service is deployed
  and reachable, and that both API keys are set there.

### 10. Every answer shows the model used; pills only course algorithms (no Elo/Poisson) — ⚠️ PARTIAL (honesty risk)
- **Pills shown are course-approved only.** `index.html` `PILL_MAP` only emits labels from:
  Cosine Similarity, Jaccard Similarity, K-Means Clustering, Content-Based Filtering,
  Anomaly Detection (Z-Score), Logistic Regression, Collaborative Filtering, Live API. The raw
  `🔍 Method:` line is **stripped** from the visible bubble (`appendMsg`) and replaced by pills.
- **But this is achieved by RELABELING non-course models.** `PILL_MAP` maps
  `poisson|softmax|elo|random forest|gradient boost|monte carlo → "Logistic Regression"`. Under the hood:
  - `predict_club_match_score` → **RandomForest + Gradient Boosting + Poisson** (`prediction_engine.py:635`).
  - `predict_wc_winner/group` → **Monte Carlo + softmax + Poisson** (`wc_predictor.py`).
  - `predict_match` (club branch) → **Poisson** (`tools/predict_match.py:42,120`).
  - `predict_top_scorer` / `predict_player_goals` → **RandomForest regressor**.
  - Only `predict_match` (national) and `predict_wc_match` are a **genuine Logistic Regression** (`match_predictor.py`).
- Risks: (a) the relabeling is **misleading** if the professor inspects the code; (b) the `🔍 Method:`
  text (containing "Poisson"/"RandomForest"/"Monte Carlo") is still present in the `/chat` JSON
  `response` field even though it's hidden in the bubble. **So "no Elo/Poisson" is true for the visible
  pills, but false for the actual models and the raw payload.**

---

## Per-model confirmation

| Model | Course algorithm | Data columns / tables | Real or faked | Example query | Live? |
|---|---|---|---|---|---|
| Jaccard | Jaccard similarity | trait sets from `players_clean` (position, nationality, foot, league, age_bucket, value_tier, archetype) | **Real** | "Compare Mbappé and Vinícius" | ✅ |
| Similar players | **Cosine** (weighted) | `fc_pace/shooting/passing/dribbling/defending/physic` + per-90 (normalized) | **Real** | "Players similar to Bellingham" | ✅ |
| Content-based scout (target) | (weighted Euclidean) | same features, vs synthetic ideal-target vector | **Real** | "Young high-potential CB" | ✅ |
| K-Means | K-Means clustering | feature vectors → `cluster`/`archetype` + `feature_meta.json` centroids | **Real (offline prep)** | "What type of player is Haaland?" | ✅ (consumed) |
| Anomaly | Z-score in cluster | `X`, cluster mean/std | **Real** | "Overperformers in Serie A" | ✅ |
| Content-based scout | Content-based filtering | FC26 attrs + per-90 + potential/age | **Real** | "Young high-potential CB" | ✅ |
| Match outcome | **Logistic Regression** | squad-strength diffs from `world_cup_2026_squads.csv` + `players_clean.csv`, trained on `national_matches.csv` | **Real** | "Predict Spain vs Morocco" | ✅ |
| Club scoreline | (RF + GB + Poisson) | `club_matches.csv` | Real model, **not a course algo** | "Score: Chelsea vs Arsenal" | ✅ (mislabeled) |
| WC winner / top scorer | (Monte Carlo / softmax / RF) | `national_matches.csv`, squads, player goals | Real model, **not a course algo** | "Who wins the World Cup?" | ✅ (mislabeled) |
| NL understanding/gen | LLM (OpenAI GPT-5.5) | — | **Real** | any question | ✅ |

---

## Summary table

| # | Requirement | Status | Where in code | Notes |
|---|---|---|---|---|
| 1 | Free NL agent (He/En) | ✅ | `agent.py` invoke / `_detect_language` | Verified live both languages |
| 2 | ≥2 external sources | ✅ | `tools/get_live_standings.py` + `data/*.csv` | API + Transfermarkt + FBref + FC26 + squads |
| 3 | Similarity (real) | ✅ | `compare_players_jaccard.py`, `scouting.calculate_weighted_cosine` | Jaccard + Cosine (similar players) + Euclidean (target scout) — all live & real |
| 4 | Clustering (real) | ✅ | `data_prep.py` KMeans → `get_player_archetype.py` | Computed offline, used live |
| 5 | Recommendations | ✅ | `scouting.py` search/wonderkids | Real weighted similarity |
| 6 | Anomaly detection | ✅ | `detect_anomalies.py` / `ds_engine.zscores` | Z-score vs cluster, exposed |
| 7 | NLP / LLM | ✅ | `agent.py` (OpenAI GPT-5.5) | Tool-calling + generation |
| 8 | Domain-limited | ❌/⚠️ | `SYSTEM_PROMPT` | Rich domain ✅, **no off-domain refusal** |
| 9 | Runnable on Render | ✅/⚠️ | `render.yaml`, `/healthz` | Deployable; **confirm live URL** |
| 10 | Method pills, no Elo/Poisson | ⚠️ | `index.html` PILL_MAP | Pills are course-only, but **non-course models relabeled "Logistic Regression"** |

---

## Gaps to fix before the demo (ranked)

1. **(HIGH) Pill honesty / "no Elo/Poisson".** Decide one of:
   (a) demo only the genuine-course-model features (match LR, Jaccard, similarity, K-Means archetype,
   anomaly, content-based scout); or (b) replace the club/WC predictors with real course models; or
   (c) relabel pills honestly. **At minimum, strip the `🔍 Method:` text from the `/chat` payload** so
   "Poisson/RandomForest/Monte Carlo" can never leak, and don't claim "Logistic Regression" for the
   Poisson/RF predictors.
2. **(HIGH) Add an off-domain refusal** to `SYSTEM_PROMPT` (politely decline non-football questions) —
   requirement #8. Quick prompt change.
3. ✅ **DONE — Cosine similarity is now the active "similar players" metric** via
   `scouting.calculate_weighted_cosine`. The Method line says "Role-weighted cosine similarity
   between player vectors..." and the pill reads "Cosine Similarity".
4. **(MEDIUM) Confirm the live Render deployment** (public URL up, both API keys set) — requirement #9.
5. **(LOW) Be ready to explain** that K-Means runs in the data-prep pipeline (offline) and its output
   drives archetype + anomaly answers.
