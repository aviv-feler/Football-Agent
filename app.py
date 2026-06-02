"""
app.py
שרת Flask של ScoutAI - נקודת הכניסה הראשית
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()  # טוען .env בפיתוח מקומי

from agent import load_resources, build_agent

app = Flask(__name__)

# ── טעינת משאבים פעם אחת ב-startup ───────────────────────────────────────────
print("[app] מאתחל ScoutAI…", flush=True)
_engine, _national_strength, _schedule = load_resources()
_agent = build_agent(_engine, _national_strength, _schedule)
print("[app] ScoutAI מוכן לקבל שאלות.", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"error": "הודעה ריקה"}), 400

    try:
        response = _agent.invoke(user_msg)
    except Exception as e:
        print(f"[app] שגיאה בעת הרצת agent: {e}", flush=True)
        response = (
            "מצטער, אירעה שגיאה בעיבוד הבקשה שלך. "
            "נסה לנסח מחדש את השאלה."
        )

    return jsonify({"response": response})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
