"""
TOOL 4 - Detect anomalous players.
Method: Z-score of each feature relative to the player's K-Means cluster.
Players with |z| > 2 on a key feature are flagged as over/underperformers.
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

        # Use established players only; per-90 values are noisy with low minutes.
        sub = sub[sub["minutes_played"] >= 3000]
        if sub.empty:
            return "Not enough players with meaningful minutes for anomaly analysis."

        findings = []
        for il in (df.index.get_loc(i) for i in sub.index):
            z = engine.zscores(il)
            # Key feature with the largest absolute z-score.
            worst = max(_KEY_FEATS, key=lambda k: abs(z.get(k, 0)))
            zval = z.get(worst, 0)
            if abs(zval) >= _Z_THRESHOLD:
                findings.append((il, worst, zval))

        if not findings:
            suffix = f" for {filter_by}" if filter_by else ""
            return f"No anomalous players were found{suffix} (|z|>2)."

        findings.sort(key=lambda t: abs(t[2]), reverse=True)
        lines = [f"**Anomalous players{' - ' + filter_by if filter_by else ''}** "
                 f"(deviation above 2 standard deviations from cluster profile):\n"]
        for rank, (il, feat, zval) in enumerate(findings[:8], 1):
            r = df.iloc[il]
            kind = "overperformer" if zval > 0 else "underperformer"
            lines.append(
                f"{rank}. **{r['player_name']}** ({r.get('position','?')} | {r.get('club','?')}) "
                f"- {kind} in {feat}: z={zval:+.1f} | archetype: {r.get('archetype','?')} | "
                f"goals: {int(r.get('goals',0))} | assists: {int(r.get('assists',0))}"
            )
        lines.append("\n🔍 Method: Z-score deviation from K-Means cluster centroid (|z|>2).")
        return "\n".join(lines)

    return detect_anomalies
