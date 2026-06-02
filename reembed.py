"""
reembed.py
מחשב מחדש את ה-embeddings מתוך players_clean.csv הקיים,
תוך שימוש בפרופיל הטקסט החדש (מבוסס סגנון, ללא שם).
מהיר יותר מהרצת data_prep.py המלא כי לא מבצע מיזוג/clustering מחדש.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from profile_utils import build_profile_text

DATA_CSV        = "data/players_clean.csv"
EMBEDDINGS_FILE = "data/embeddings.npy"
MODEL_NAME      = "all-MiniLM-L6-v2"


def main():
    print("[reembed] טוען players_clean.csv...", flush=True)
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"[reembed] {len(df)} שחקנים", flush=True)

    print("[reembed] בונה פרופילי טקסט חדשים (מבוססי סגנון)...", flush=True)
    df["profile_text"] = df.apply(build_profile_text, axis=1)
    print(f"[reembed] דוגמה: {df['profile_text'].iloc[0]}", flush=True)

    # שמירת ה-profile_text המעודכן חזרה ל-CSV
    df.to_csv(DATA_CSV, index=False)

    print(f"[reembed] מחשב embeddings עם {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(df["profile_text"].tolist(), show_progress_bar=True, batch_size=128)
    np.save(EMBEDDINGS_FILE, emb)
    print(f"[reembed] נשמר {emb.shape} ל-{EMBEDDINGS_FILE}", flush=True)
    print("[reembed] הושלם.", flush=True)


if __name__ == "__main__":
    main()
