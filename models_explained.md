# ScoutAI — Data Science Models Explained

מסמך אקדמי המתאר את מודלי ה-Data Science שבהם משתמש כל כלי בסוכן, הנוסחה,
והסיבה לבחירה. זהו הבסיס להסבר בהגנה ("איזה מודל שימש בכל שאילתה וכיצד").

---

## 0. בסיס: וקטור הפיצ'רים (Feature Engineering + Normalization)

לכל שחקן נבנה **וקטור ביצועים מספרי** מ-10 פיצ'רים (`data_prep.py`):

| פיצ'ר | תיאור |
|-------|-------|
| `goals_per90`, `assists_per90`, `ga_per90` | תפוקה התקפית מנורמלת ל-90 דקות |
| `cards_per90` | משמעת |
| `minutes_played_log`, `appearances_log` | היקף משחק / ניסיון (log) |
| `age`, `height_in_cm` | פרופיל פיזי |
| `market_value_log` | שווי שוק (log) — איכות עדכנית |
| `international_caps_log` | ותק בנבחרת |

**טיפול בחוסרים:** מילוי לפי **חציון קבוצת-העמדה** (לא 0).
**נרמול:** `StandardScaler` (z-score) — **קריטי**. בלי נרמול, שווי השוק (מיליונים)
משתלט על גולים (חד-ספרתי) והדמיון מאבד משמעות. זו הייתה הסיבה לתוצאות הגרועות בגרסה
הקודמת (שהשתמשה ב-embeddings סמנטיים של טקסט).

הנוסחה: \( z_i = (x_i - \mu_i) / \sigma_i \). הפלט נשמר ב-`player_features.npy`.

---

## 1. find_similar_players → **Cosine Similarity**

מודדים דמיון בין שני שחקנים כווקטורים מספריים:
\[ \text{cos}(\vec{a},\vec{b}) = \frac{\vec{a}\cdot\vec{b}}{\lVert\vec{a}\rVert\,\lVert\vec{b}\rVert} \]
- מסננים מראש לאותה קבוצת-עמדה (לא מחזירים שוער כדומה לחלוץ).
- **למה Cosine ולא Euclidean?** Cosine מודד דמיון *בכיוון הפרופיל* (סגנון יחסי) ופחות
  רגיש לעוצמה מוחלטת — מתאים ל"שחקן דומה בסגנון". Euclidean זמין גם הוא במנוע.

## 2. scout_players → **Content-Based Recommendation**

המלצה מבוססת-תוכן: בונים **וקטור-מטרה אידיאלי** מהקריטריון (דגש על goals/assists/value
לפי הניסוח), מסננים סינון קשיח (עמדה/גיל/אזור), ומדרגים בשילוב:
\[ \text{score} = 0.5\cdot\text{cosine}(\text{candidate}, \text{target}) + 0.5\cdot\text{quality} \]
כאשר quality = שווי שוק מנורמל (log). השילוב מונע החזרת שחקנים אלמונים ש"מצביעים לכיוון"
הנכון אך חסרי איכות. Gemini משמש רק כשכבת ה-NLP שמחלצת את הקריטריון.
*(חלופה: Item-Item Collaborative Filtering — לא מומשה כי אין נתוני דירוג משתמשים.)*

## 3. get_player_archetype → **K-Means Clustering**

מקבצים שחקנים ל-k ארכיטיפים (תפקידים) על הווקטור המנורמל.
- **בחירת k:** שיטת המרפק (elbow) — מריצים k=2..10, מודדים inertia, ובוחרים את הנקודה
  עם המרחק המקסימלי מהקו הישר שמחבר את הקצוות. נבחר **k=4**.
- כל אשכול מקבל שם ארכיטיפ לפי הפיצ'ר המבדיל ביותר במרכז (centroid).
- ארכיטיפים: High-minutes regular · Goalscorer/Poacher · Fringe/limited minutes · Elite high-value.

## 4. detect_anomalies → **Z-score from Cluster Centroid**

לכל שחקן (עם ≥3000 דקות) מחשבים z מול **ממוצע וסטיית התקן של האשכול שלו**:
\[ z_i = (x_i - \mu^{(c)}_i) / \sigma^{(c)}_i \]
שחקן עם \(|z|>2\) על פיצ'ר מפתח מסומן כ-overperformer (z חיובי) או underperformer (z שלילי).

## 5. compare_players_jaccard → **Jaccard Similarity**

בונים לכל שחקן **קבוצת תכונות קטגוריאליות**: {עמדה, תת-עמדה, לאום, רגל, ליגה,
קבוצת-גיל, דרגת-שווי, ארכיטיפ}, ומחשבים:
\[ J(A,B) = \frac{|A \cap B|}{|A \cup B|} \]
מתאים להשוואה מבוססת-קבוצות (תכונות בדידות), בניגוד ל-Cosine שמתאים לווקטורים רציפים.

## 6. predict_match → **Squad-Strength Logistic (Softmax)**

חוזק נבחרת נגזר מצבירת נתוני השחקנים (שווי סגל ממוצע + עומק). ההסתברויות:
\[ P(\text{team}_1) = \frac{e^{k\cdot s_1}}{e^{k\cdot s_1}+e^{k\cdot s_2}} \]
עם הקצאת הסתברות-תיקו לפי קרבת החוזק. (football-data.org משמש לטבלאות חיות בכלי נפרד.)

---

## סיכום מיפוי שאילתה → מודל

| שאילתת משתמש | כלי | מודל DS |
|--------------|-----|---------|
| "שחקנים דומים ל-X" | find_similar_players | Cosine similarity |
| "מצא/הכי טובים..." | scout_players | Content-based (cosine + quality) |
| "איזה סוג שחקן / ארכיטיפ" | get_player_archetype | K-Means (elbow) |
| "שחקנים חריגים" | detect_anomalies | Z-score מול centroid |
| "השווה X ל-Y" | compare_players_jaccard | Jaccard |
| "מי ינצח X נגד Y" | predict_match | Softmax על חוזק סגל |
