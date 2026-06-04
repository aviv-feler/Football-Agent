"""
tools/detect_anomalies.py
TOOL 4 — זיהוי שחקנים חריגים.
שיטה: Z-score של כל פיצ'ר מול מרכז האשכול (K-Means) של השחקן.
שחקן עם |z| > 2 על פיצ'ר מפתח מסומן כ-overperformer / underperformer.
"""

import numpy as np
from langchain.tools import tool

_KEY_FEATS = ["goals_per90", "assists_per90", "ga_per90", "minutes_played_log", "market_value_log"]
_Z_THRESHOLD = 2.0


def make_detect_anomalies_tool(engine):
    df = engine.df

    @tool
    def detect_anomalies(filter_by: str = "") -> str:
        """
        Detect statistically anomalous players — those whose stats deviate strongly
        (|z| > 2) from their K-Means cluster centroid: overperformers or underperformers.
        Optionally filter by position (e.g. 'Attack') or nationality. Empty = all players.
        Input is an optional filter string.
        """
        f = filter_by.strip().lower()
        sub = df
        if f:
            mask = (df["position"].astype(str).str.lower().str.contains(f, na=False) |
                    df["nationality"].astype(str).str.lower().str.contains(f, na=False))
            if mask.any():
                sub = df[mask]

        # סורקים מועמדים מבוססים (≥3000 דקות ≈ עונה+) אחרת per-90 רועש מאוד
        sub = sub[sub["minutes_played"] >= 3000]
        if sub.empty:
            return (
                "אין מספיק שחקנים עם זמן משחק לניתוח חריגות."
                "\n\n🔍 Method: Z-score deviation from K-Means cluster centroid (|z|>2)."
            )

        findings = []
        for il in (df.index.get_loc(i) for i in sub.index):
            z = engine.zscores(il)
            # הפיצ'ר עם |z| המקסימלי מבין פיצ'רי המפתח
            worst = max(_KEY_FEATS, key=lambda k: abs(z.get(k, 0)))
            zval = z.get(worst, 0)
            if abs(zval) >= _Z_THRESHOLD:
                findings.append((il, worst, zval))

        if not findings:
            return (
                f"לא נמצאו שחקנים חריגים{' עבור ' + filter_by if filter_by else ''} (|z|>2)."
                "\n\n🔍 Method: Z-score deviation from K-Means cluster centroid (|z|>2)."
            )

        findings.sort(key=lambda t: abs(t[2]), reverse=True)
        lines = [f"**שחקנים חריגים{' — ' + filter_by if filter_by else ''}** "
                 f"(סטייה מעל 2σ ממרכז האשכול):\n"]
        for rank, (il, feat, zval) in enumerate(findings[:8], 1):
            r = df.iloc[il]
            kind = "מצטיין (overperformer)" if zval > 0 else "מתחת לציפייה (underperformer)"
            lines.append(
                f"{rank}. **{r['player_name']}** ({r.get('position','?')} | {r.get('club','?')}) "
                f"— {kind} ב-{feat}: z={zval:+.1f} | ארכיטיפ: {r.get('archetype','?')} | "
                f"גולים: {int(r.get('goals',0))} | בישולים: {int(r.get('assists',0))}"
            )
        lines.append("\n🔍 Method: Z-score deviation from K-Means cluster centroid (|z|>2).")
        return "\n".join(lines)

    return detect_anomalies
