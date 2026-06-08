# ScoutAI — Technical Architecture Audit

**Honest, code-level audit of what the agent actually does right now.**
Written by reading the real source, not the intended design. Date: 2026-06-05.

---

## TL;DR — the honest verdict (read this first)

| Capability | What actually computes it | Verdict |
|---|---|---|
| Find similar players | `cosine` similarity on 16 normalized numeric features (`tools/find_similar_players.py` → `ds_engine.DSEngine`) | ✅ **REAL MODEL** |
| Scout / filter queries | Hard filters + content-based `cosine` to an ideal vector + market-value blend (`tools/scout_players.py`) | ✅ **REAL MODEL** |
| Player archetype | K-Means cluster + centroid traits (`tools/get_player_archetype.py`, clusters from `data_prep.py`) | ✅ **REAL MODEL** |
| Anomaly detection | Z-score vs cluster centroid, \|z\|>2 (`tools/detect_anomalies.py`) | ✅ **REAL MODEL** |
| Compare players | Jaccard on categorical traits (`tools/compare_players_jaccard.py`) | ✅ **REAL MODEL** |
| Predict club match | Recency-weighted Poisson goals model (`club_model.py`) | ✅ **REAL MODEL** |
| Predict national/WC match | Hybrid squad-value + World Cup Elo (`ds_engine.build_national_strength` + `tools/predict_match.py`) | ✅ **REAL MODEL** |
| Live standings / scorers | football-data.org REST API (`tools/get_live_standings.py`) | ✅ **REAL API**, key is configured |

**The LLM (OpenAI GPT-5.5) does NOT invent any of the numbers, rankings, or probabilities.**
Every figure in those answers is computed by Python/scikit-learn/your own code. The LLM's job
is (a) deciding which tool to call and (b) wording the final sentence — and for many queries it
is bypassed entirely (see §1).

> **UPDATE 2026-06-05 — most of the "ugly bits" below were FIXED.** The agent was
> restructured to be **LLM-first**: `ScoutAgent.invoke()` now always lets the LLM reason
> and choose tools (so it decides on its own to call the live football-data.org API), and
> `_direct_tool_answer` is now only a **fallback** used when the LLM is unavailable (quota).
> A new live tool `get_top_scorers` (football-data.org `/scorers`) was added, the
> `compare_players_jaccard` tool now includes a side-by-side stat block, the heavy unused
> `FootballQAPipeline` is no longer built at startup, and the dead `embeddings.npy` /
> `future_match_probabilities_baseline.csv` files were deleted. The list below is kept for
> history; the ✅/⚠️ marks the current state.

**Things you should know before the demo (the ugly bits):**

1. ✅ **FIXED-IN-PRACTICE — `football_qa.py` (≈1,480 lines) is still disabled and now not even
   imported at runtime.** It is no longer constructed (saved a big TF-IDF startup cost). The
   file remains on disk (used only by `test_football_qa.py`); it does not run in the app.
2. ✅ **FIXED — the LLM now handles every query.** It reasons, calls tools (function-calling),
   and writes the answer. The keyword router only runs if the LLM hits its quota.
3. ✅ **FIXED — Hebrew answers.** Because the LLM is now primary, answers come back in the
   user's language. (Only the rare quota-fallback path returns raw English tool text.)
4. ✅ **FIXED — dead files deleted** (`embeddings.npy`, `future_match_probabilities_baseline.csv`).
   ⚠️ `data_manager._read_xlsx_sheet()` + the `Football_clubs_players_full.xlsx` constant are
   still present but unused (kept only so the dead `football_qa.py` still imports).
5. ⚠️ **Still true — the OpenAI account can hit a rate limit / quota (HTTP 429)**; the
   code rotates to the configured fallback models on 429 (and on 404/unsupported-param
   errors), and now also falls back to the deterministic tool router so the agent keeps
   answering even when the LLM is gone.
6. ⚠️ **Known data-quality caveat — K-Means archetype labels are coarse (only 4 global
   clusters).** A midfielder like Pedri can be labelled "Goalscorer / Poacher". Similarity
   percentages and stats are correct; only the archetype *name* is imprecise. Worth refining
   (more/position-aware clusters) before leaning on archetype wording in the demo.

---

## 1. Agent pipeline — the full path of a query

### Entry point
- **Frontend:** `templates/index.html` (line ~413) does `fetch('/chat', { method POST, body
  { message, session_id } })`. `session_id` is a UUID kept in `localStorage`.
- **Backend route:** `app.py` → `@app.route("/chat", methods=["POST"])` function `chat()`
  (line ~35). It reads `message` + `session_id`, then calls `_agent.invoke(user_msg,
  session_id=session_id)` under a global `threading.RLock` (`_agent_lock`). The agent is a
  single shared `ScoutAgent` built once at startup by `build_agent()`.
- There is also `@app.route("/reset")` → clears that session's history, and `/healthz`.

### What `ScoutAgent.invoke()` does (`agent.py`, ~line 305)
Numbered flow for a single message:

```
1. User types in browser
   → POST /chat  (app.py: chat())
2. chat() → ScoutAgent.invoke(user_input, session_id)   [agent.py]
3. _detect_language(user_input)   → "Hebrew" or "English" (Unicode range check)
4. direct = _direct_tool_answer(user_input)             ← KEYWORD ROUTER, runs FIRST
       • lowercases the text and keyword-matches:
         similar/like/דומה        → find_similar_players tool
         archetype/profile/ארכיטיפ → get_player_archetype tool
         anomal/overperform/חריג   → detect_anomalies tool
         compare/jaccard/השווה     → compare_players_jaccard tool
         predict/who wins/נבא      → predict_match tool   (+ regex to split "A vs B")
         standings/table/טבלה      → get_live_standings tool
         world cup/group/fixture   → world_cup_info tool
         best/top/find/striker...  → scout_players tool   (_looks_like_scout_query)
       • If matched: calls tool.invoke(...) and RETURNS THE TOOL'S RAW STRING.
         >>> The LLM is NOT called. Answer is English, includes the 🔍 Method line. <<<
5. (Smart-QA pipeline) — SKIPPED. Guarded by env SCOUTAI_ENABLE_SMART_QA (default off).
6. If step 4 did not match → LLM TOOL-CALLING LOOP:
       a. Build messages = [system_prompt, session_history, language_directive, user_msg]
       b. _call_llm(messages): invoke an OpenAI (GPT-5.5) model that has the tools bound via
          llm.bind_tools(tools)  (LangChain function-calling). On HTTP 429 / quota / 404 /
          unsupported-param it rotates to the next model in MODEL_CHAIN.
       c. If the LLM returns tool_calls → run each tool, append ToolMessage(results),
          loop again (max 5 iterations).
       d. When the LLM returns text with no tool calls → that text is the final answer.
          (Fallback: if the text is empty, it retries _direct_tool_answer.)
7. _remember(session_id, user_input, answer)  → per-session history (max 6 turns).
8. Return answer string → app.py jsonifies {response, session_id} → browser renders it.
```

**Key truths about routing:**
- There are **two live routers**: (A) the deterministic keyword router
  `_direct_tool_answer` (runs first, no LLM), and (B) the OpenAI LLM function-calling (only if A
  misses). For your typical demo queries ("players similar to X", "predict A vs B",
  "best strikers in La Liga", "compare X and Y"), **router A catches them** and returns the
  tool output directly. The LLM is used mainly for free-form/unmatched questions.
- Both routers call **the same tool functions** with the same real models. So the numbers
  are identical either way; the only difference is whether the LLM rewrites the wording.
- On the LLM path, the system prompt (`SYSTEM_PROMPT` in `agent.py`) hard-instructs it to
  call tools and never answer player/stat questions from memory, and to keep the `🔍 Method:`
  line. This is a *prompt instruction, not an enforced guarantee* — a free-form question that
  the LLM chooses to answer without a tool can still be ungrounded.

---

## 2. football-data.org API — when is it actually called?

- **Live path:** only **one** tool calls the API in normal operation:
  `tools/get_live_standings.py` → `get_live_standings(competition)`.
  - Endpoints hit: `GET /v4/competitions/{ID}/standings`, and if that's empty it falls back
    to `GET /v4/competitions/{ID}/matches?status=SCHEDULED&limit=10`.
  - `{ID}` comes from `COMPETITION_MAP` (premier league→PL, la liga→PD, bundesliga→BL1,
    serie a→SA, ligue 1→FL1, champions league→CL, world cup→WC, euro→EC, …).
  - Returns: standings table rows (position, team, P/W/D/L, GF/GA, points) or upcoming
    fixtures.
- **Is the key real?** Yes. `.env` (or the Render dashboard) provides `FOOTBALL_DATA_API_KEY`
  and `OPENAI_API_KEY`, loaded via `load_dotenv()` in `app.py`. The tool reads
  `os.getenv("FOOTBALL_DATA_API_KEY")`; if it were missing it returns a graceful "not
  configured" message instead of crashing.
- **When does a live call happen?** Only when the user asks about standings / league table
  (router A keyword `standings`/`table`/`טבלה`, or the LLM choosing the tool). Similar-player,
  scouting, prediction, archetype, anomaly, and compare queries make **no** API call — they
  use local data.
- **Caveat for the demo:** asking for **World Cup 2026 standings** via this tool will likely
  return nothing useful — that competition has no standings yet — and it falls back to
  fixtures. World Cup *schedule* questions are served by `world_cup_info` from the local CSV,
  not the API.
- **Second API client exists but is DEAD:** `football_qa.py` has its own `FootballDataClient`
  (scorers + standings). It only runs inside the disabled smart-QA pipeline, so in practice
  it never executes.

---

## 3. LLM — which model exactly?

- **Provider:** OpenAI, via `langchain-openai`'s `ChatOpenAI` (no Google/Gemini anywhere).
- **Configured in `agent.py`**, `MODEL_CHAIN` (~line 191) is built from env vars at import:
  ```
  _PRIMARY_MODEL   = os.getenv("OPENAI_MODEL")          or "gpt-5.5"
  _FALLBACK_MODELS = os.getenv("OPENAI_FALLBACK_MODEL")  or "gpt-5-mini,gpt-4o-mini"
  MODEL_CHAIN      = [primary, *fallbacks]   # de-duplicated, order preserved
  ```
  So by default `MODEL_CHAIN == ["gpt-5.5", "gpt-5-mini", "gpt-4o-mini"]`. `render.yaml` sets
  `OPENAI_MODEL=gpt-5.5`.
- **How each model is wrapped** (`build_agent()`): `ChatOpenAI(model=..., api_key=...,
  max_retries=0)` then `.bind_tools(tools)`. GPT-5 / o-series are reasoning models, so instead
  of `temperature` they get `reasoning_effort` (default `"low"`, override with
  `OPENAI_REASONING_EFFORT`) — keeping effort low cut narration latency from ~15s to ~4–6s.
  Non-reasoning models (e.g. `gpt-4o-mini`) get `temperature=0.3` instead.
- **Rotation:** `_call_llm()` tries the current model and rotates to the next one in the chain
  on rate-limit/quota (`429`, `rate_limit`, `insufficient_quota`), transient server errors
  (`500/502/503`, `overloaded`, `unavailable`), or **bad-model/param errors** (`404`,
  `model_not_found`, `does not exist`, `unsupported value`, `reasoning_effort`). That last
  group means a stale or renamed primary string falls back to a valid model instead of
  hard-failing the request.
- **The LLM's role:** strictly (1) decide which tool to call (function-calling) on the LLM
  path, and (2) phrase the final natural-language answer / translate it to the user's
  language. **It does not generate the data, numbers, rankings, or probabilities** — those
  come from the tools. On the keyword (`_direct_tool_answer`) path, the LLM is not involved at
  all.

---

## 4. Data Science models — are they real? (the important part)

### A) Finding similar players — ✅ REAL MODEL
- **Live code:** `tools/find_similar_players.py` → uses `ds_engine.DSEngine.cosine()`.
- **Algorithm:** cosine similarity between the target player's vector and candidate vectors,
  **restricted to the same broad position**, top 5.
- **Features (16):** from `data/player_features.npy` (built by `data_prep.py`), names in
  `data/feature_meta.json`:
  `goals_per90, assists_per90, ga_per90, cards_per90, minutes_played_log, appearances_log,
  age, height_in_cm, market_value_log, international_caps_log, fc_pace, fc_shooting,
  fc_passing, fc_dribbling, fc_defending, fc_physic`.
- **Normalization:** `StandardScaler` (z-scores) fit in `data_prep.py`; skewed counts use
  `log1p` first; missing values filled by **position-group median** (not zero).
- **Note:** there is a *second*, more elaborate similar-players function
  (`football_qa.findSimilarPlayers`, which blends cosine + Jaccard + market value + cluster
  bonus). **It is part of the disabled pipeline and does not run.** The live tool is the
  simpler cosine one above.

### B) Predicting match results — ✅ REAL MODEL (two of them)
- **Club vs club** (`club_model.py`, `ClubModel`): a **recency-weighted Poisson goals model**.
  Each club gets attack/defence factors (weighted by recency, half-life 2 years, normalized
  within its league) + a global home-advantage baseline → two Poisson distributions → a score
  matrix → P(home/draw/away), expected goals, most-likely scoreline.
  - Data: `data/club_matches.csv` (17,936 top-5-league matches, 10 seasons), built from the
    `data/<league>/` football-data.co.uk CSVs by `build_club_data.py`.
  - Validation: backtest in `ClubModel.backtest()` — model ~50% vs bookmaker ~53% on 3,181
    held-out matches.
- **National / World Cup** (`tools/predict_match.py` + `ds_engine.build_national_strength`):
  **hybrid** = current squad market value **+** walk-forward World Cup **Elo** pedigree, fed
  to a softmax over the strength difference.
  - Data: player market values (squad strength) + `data/national_matches.csv` (253 WC matches
    2010/2014/2018/2022, from `build_national_data.py`).
  - Teams with no WC history fall back to squad strength only.
- **No trained model file (.pkl) is saved.** Both predictors are **computed on the fly at
  startup** from the CSVs (Poisson factors and Elo ratings are recomputed each boot). The
  K-Means clusters/scaler *are* persisted (`feature_meta.json` + `player_features.npy`).
- **`predict_match` routing:** if both names resolve to known clubs → Poisson model; if both
  resolve to nations → hybrid model; otherwise a graceful "couldn't build a prediction"
  message. The LLM does not compute any of these probabilities.
- **Honest caveats:** cross-league club matches (e.g. Man City vs Bayern) are approximate
  (factors are league-relative); national history is only 4 World Cups; the LLM is not the
  predictor in any case.

### C) Scout / filter queries — ✅ REAL MODEL
- **Live code:** `tools/scout_players.py`.
- **NL → filters:** done in **Python with regex/keyword maps**, *not* by the LLM. It parses
  position keywords, age phrases ("under 23", "young", "veteran"), region/continent and
  nationality (incl. Hebrew aliases), league, and World Cup mentions, applying hard filters.
- **Ranking:** builds an "ideal" target vector by emphasizing relevant features, then
  `cosine_to_vector` similarity **blended 50/50 with a market-value quality score**
  (`0.5*cosine + 0.5*log market value`). The market-value blend exists because pure cosine
  returned obscure players. Output top 5 with the `🔍 Method:` line.
- The LLM only rewords this if the query reaches the LLM path; the ranking itself is the model.

### Also real (not in your three but graded):
- **Archetype** (`tools/get_player_archetype.py`): K-Means cluster membership + centroid's
  strongest features, k chosen by the elbow method in `data_prep.py`. Cluster labels in
  `feature_meta.json`.
- **Anomalies** (`tools/detect_anomalies.py`): per-feature **z-score vs the player's K-Means
  cluster centroid**, flags \|z\|>2, restricted to players with ≥3000 minutes.
- **Compare** (`tools/compare_players_jaccard.py`): **Jaccard** similarity on categorical
  trait sets (position, sub-position, nationality, foot, league, age bucket, value tier,
  archetype).

**Bottom line for the professor:** every core capability is a real algorithm on real data.
The only "faking" risk is the narrow case where a free-form question slips past the keyword
router *and* the LLM answers it without calling a tool — then you get LLM prose, not model
output. For the scripted demo queries, that won't happen.

---

## 5. Data sources & parameters — the full map

### Files loaded at startup (`agent.load_resources` / `build_agent`)
| File | Loaded by | Contents | Used? |
|---|---|---|---|
| `data/players_clean.csv` | `ds_engine.load_engine` | 47,701 players, 16 features + metadata (cluster, archetype, value_tier…) | ✅ live |
| `data/player_features.npy` | `ds_engine.load_engine` | 47,701 × 16 normalized matrix (row-aligned to players_clean) | ✅ live |
| `data/feature_meta.json` | `ds_engine.load_engine` | feature names, scaler mean/scale, k, centroids, archetype labels | ✅ live |
| `data/player_profiles.csv` | `data_manager.load_player_profiles` | richer per-player profile (current-season scores, fc_* attrs, potential) | ✅ live |
| `data/national_matches.csv` | `ds_engine.build_national_strength` | 253 WC matches (2010–2022) for Elo | ✅ live |
| `data/club_matches.csv` | `club_model.ClubModel` | 17,936 top-5 league matches for Poisson | ✅ live |
| `data/fwc26_match_schedule_agent.csv` | `agent.load_resources` | 104 World Cup 2026 fixtures | ✅ live |
| `data/data_source_map.json` | written by `data_manager.write_source_map` | provenance doc (not read at runtime) | ℹ️ output only |
| `data/players.csv` (Transfermarkt) | `data_prep.py`, `data_manager` profile build | raw identity/market value backbone | ⚙️ build-time |
| `data/players_data-2025_2026.csv` (FBref) | `data_manager.load_current_player_stats` | current-season top-5 stats | ⚙️ build/profiles |
| `data/FC26_20250921.csv` | `data_prep.load_fc26` | EA FC26 attributes + potential | ⚙️ build-time |
| `data/appearances.csv` | `data_prep`, `data_manager` | career per-player aggregates | ⚙️ build-time |
| `data/WorldCup_2018_2014_2010_2022.xlsx` | `build_national_data.py` | raw WC results | ⚙️ build-time |
| `data/<league folders>/*.csv` | `build_club_data.py` | raw league results+odds | ⚙️ build-time |
| `data/embeddings.npy` (73 MB) | — | old sentence-embeddings | ❌ **DEAD, not loaded** |
| `data/future_match_probabilities_baseline.csv` | — | old baseline output | ❌ **DEAD, not loaded** |
| `data/games.csv`, `appearances.csv`, `game_lineups.csv`, `transfers.csv` | partly build-time / kept | Transfermarkt extras | mostly unused at runtime |

### Tool → Data Source → Columns → Algorithm
| Tool (live) | Data source | Key columns / params | Model / algorithm |
|---|---|---|---|
| `find_similar_players` | `player_features.npy` + `players_clean.csv` | 16 normalized features; same `position` group | **Cosine similarity** (top 5) |
| `scout_players` | `player_features.npy` + `players_clean.csv` | position, age, nationality/region, league, WC; feature emphasis + `market_value_in_eur` | **Content-based: cosine to ideal vector + value blend**, regex/keyword filters |
| `get_player_archetype` | `feature_meta.json` + `players_clean.csv` | `cluster`, centroids, feature_names | **K-Means** (elbow-chosen k) |
| `detect_anomalies` | `player_features.npy` + `players_clean.csv` | features vs cluster centroid; `minutes_played`≥3000 | **Z-score** (\|z\|>2) |
| `compare_players_jaccard` | `players_clean.csv` | position, sub_position, nationality, foot, league, age_bucket, value_tier, archetype | **Jaccard** on trait sets |
| `predict_match` (club) | `club_matches.csv` | goals for/against, recency weight, home flag, odds (validation) | **Recency-weighted Poisson** |
| `predict_match` (national) | player values + `national_matches.csv` | squad market value, WC Elo, n_wc_matches | **Hybrid squad-value + Elo → softmax** |
| `get_live_standings` | football-data.org API | competition → PL/PD/BL1/SA/FL1/CL/WC… | **Live REST API** |
| `world_cup_info` | `fwc26_match_schedule_agent.csv` | team/group/stage/city/date | **CSV lookup/filter** |

---

## Appendix — dead / misleading code to clean up (not breaking, but be aware)
- `football_qa.py` (whole file): disabled unless `SCOUTAI_ENABLE_SMART_QA=1`. If you ever
  enable it, note its `_current_ranking_context` still prints
  *"Current-season player ranking from …Football_clubs_players_full.xlsx"* even though
  `rank_current_players` now reads FBref — a **misleading debug label**, not a data bug.
- `data_manager._read_xlsx_sheet()` and the `FOOTBALL_WORKBOOK_FILE` constant: leftover from
  the deleted xlsx; unused by the live path.
- `data/embeddings.npy` (73 MB) and `data/future_match_probabilities_baseline.csv`: safe to
  delete — no code loads them.
- The `_direct_tool_answer` path returns English tool output even for Hebrew questions
  (no translation). If bilingual answers matter for the demo, route those through the LLM
  path or translate the tool output.
