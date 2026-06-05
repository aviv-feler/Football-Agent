"""
UI-only preview server — serves templates/index.html + static assets WITHOUT
loading the ML / agent stack, so you can iterate on the frontend instantly.
The /chat route returns a canned demo reply so the chat flow is fully clickable.

Run:  python tools/preview_ui.py    ->  http://127.0.0.1:5050
(This is a dev helper only; the real app is app.py.)
"""
import os
from flask import Flask, render_template, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__,
            template_folder=os.path.join(ROOT, "templates"),
            static_folder=os.path.join(ROOT, "static"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    demo = (
        "**Prediction: Chelsea 2 - 1 Manchester United**\n"
        "- **Confidence:** Medium\n"
        "- **Match profile:** Balanced\n"
        "- **Key factors:** Home advantage, recent form, attacking strength\n\n"
        "_(UI preview — this is a canned demo reply, the real agent is in app.py.)_"
    )
    return jsonify({"response": demo, "session_id": "preview"})


@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"ok": True, "session_id": "preview"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
