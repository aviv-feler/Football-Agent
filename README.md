# ⚽ ScoutAI — Football Scout & Predictor Agent

סוכן AI לכדורגל שרץ על ממשק web. המשתמש משוחח בשפה חופשית (עברית/אנגלית)
ומקבל תשובות מבוססות-נתונים על שחקנים, חיזויי משחקים ומידע על מונדיאל 2026.

נבנה כפרויקט בקורס **Data Science**.

## יכולות

- **Scout Mode** — חיפוש שחקנים לפי קריטריונים ומציאת שחקנים דומים
- **Predictor Mode** — חיזוי תוצאות משחקי מונדיאל 2026

## שיטות Data Science

| שיטה | שימוש |
|------|-------|
| **K-Means clustering** | צמצום מועמדים לאשכול ההתנהגותי של השחקן |
| **TF-IDF + cosine** | דמיון על מסמך קטגוריאלי לכל שחקן |
| **Jaccard similarity** | דמיון בין קבוצות התגיות של שחקנים |
| **IsolationForest** | זיהוי שחקנים חריגים |
| **National strength** | חוזק נבחרות נגזר מצבירת נתוני השחקנים |

## כלי הסוכן (LangChain tools)

1. `find_similar_players` — שחקנים דומים (cluster + Jaccard + TF-IDF)
2. `scout_players` — סקאוטינג לפי קריטריונים בשפה טבעית
3. `detect_anomalies` — זיהוי חריגות
4. `predict_match` — חיזוי משחקי נבחרות
5. `get_live_standings` — טבלאות חיות (football-data.org)
6. `world_cup_info` — לוח משחקי מונדיאל 2026

## הרצה מקומית

```bash
pip install -r requirements.txt
# צור קובץ .env עם המפתחות (ראה .env.example):
#   GEMINI_API_KEY=...
#   FOOTBALL_DATA_API_KEY=...
python app.py        # http://127.0.0.1:5000
```

בדיקת הכלים (ללא צורך במכסת LLM):
```bash
python test_tools.py
```

## נתונים

הנתונים הגולמיים מ-Kaggle כבדים מדי ל-GitHub ואינם כלולים ב-repo:
- `maso0dahmed/football-players-data`
- `davidcariboo/player-scores`

ה-repo כולל את **`data/players_clean.csv`** המעובד (שממנו נבנים TF-IDF/clustering
בזמן ריצה) ואת **`data/fwc26_match_schedule_agent.csv`** (לוח המונדיאל).
ליצירת `players_clean.csv` מחדש מהנתונים הגולמיים:

```bash
pip install sentence-transformers==3.0.1
python data_prep.py
```

## פריסה (Render.com)

ראה `render.yaml`. הגדר את `GEMINI_API_KEY` ו-`FOOTBALL_DATA_API_KEY` כמשתני סביבה.

## הערה על מכסת Gemini

פרויקט חדש ב-Gemini API מקבל מכסת free-tier מצומצמת (≈20 בקשות ליום למודל).
הסוכן מסובב בין כמה מודלים (`MODEL_CHAIN` ב-`agent.py`) כדי למקסם את הקיבולת היומית.
