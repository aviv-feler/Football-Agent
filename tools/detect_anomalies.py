"""
tools/detect_anomalies.py
כלי לזיהוי שחקנים חריגים (overperfomers / underperformers) באמצעות IsolationForest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from langchain.tools import tool

NUMERIC_COLS = ["goals", "assists", "minutes_played", "market_value_in_eur", "age"]


def make_detect_anomalies_tool(df: pd.DataFrame):
    """Factory – יוצר את הכלי עם גישה לנתונים."""

    @tool
    def detect_anomalies(filter_by: str = "") -> str:
        """
        Detect anomalous football players - those who significantly over-perform or
        under-perform compared to similar players in their cluster.
        Optionally filter by position (e.g. 'striker', 'GK') or nationality.
        Input is an optional filter string, or empty string for all players.
        """
        subset = df.copy()

        # פילטר אופציונלי
        if filter_by.strip():
            f = filter_by.strip().lower()
            pos_mask = subset.get("position", pd.Series(dtype=str)).str.lower().str.contains(f, na=False)
            nat_mask = subset.get("nationality", pd.Series(dtype=str)).str.lower().str.contains(f, na=False)
            combined = pos_mask | nat_mask
            if combined.any():
                subset = subset[combined]

        # וידוא עמודות
        available = [c for c in NUMERIC_COLS if c in subset.columns]
        if len(available) < 2:
            return "אין מספיק נתונים מספריים לניתוח חריגות."

        data = subset[available].fillna(0)

        # IsolationForest
        iso = IsolationForest(contamination=0.1, random_state=42)
        subset = subset.copy()
        subset["anomaly_score"] = iso.fit_predict(data)
        subset["raw_score"] = iso.decision_function(data)

        anomalies = subset[subset["anomaly_score"] == -1].copy()
        anomalies = anomalies.sort_values("raw_score")  # השליליים ביותר = החריגים ביותר

        if anomalies.empty:
            return "לא נמצאו שחקנים חריגים בקבוצה המסוננת."

        lines = [f"**שחקנים חריגים שזוהו{' בקטגוריה: ' + filter_by if filter_by else ''}:**\n"]
        for rank, (_, row) in enumerate(anomalies.head(8).iterrows(), 1):
            name    = row.get("player_name", "N/A")
            pos     = row.get("position", "N/A")
            club    = row.get("club", "N/A")
            goals   = int(row.get("goals", 0))
            assists = int(row.get("assists", 0))
            age     = int(row.get("age", 0))
            cluster = int(row.get("cluster", -1))
            score   = round(row["raw_score"], 3)

            # ניתוח כיוון החריגות
            cluster_subset = df[df["cluster"] == cluster] if "cluster" in df.columns else df
            avg_goals   = cluster_subset["goals"].mean()   if "goals"   in cluster_subset else 0
            avg_assists = cluster_subset["assists"].mean() if "assists" in cluster_subset else 0

            if goals > avg_goals * 1.5 or assists > avg_assists * 1.5:
                anomaly_type = "מצטיין מעל הממוצע (overperformer)"
            elif goals < avg_goals * 0.5 and assists < avg_assists * 0.5:
                anomaly_type = "ביצועים מתחת לציפייה (underperformer)"
            else:
                anomaly_type = "חריגות סטטיסטית"

            lines.append(
                f"{rank}. **{name}** ({pos} | {club} | גיל {age})\n"
                f"   {anomaly_type} | גולים: {goals} | בישולים: {assists} "
                f"| ציון חריגות: {score}"
            )

        return "\n".join(lines)

    return detect_anomalies
