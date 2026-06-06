"""
viz.py
Out-of-band visual payloads for in-chat cards.

Bot answers are rewritten by the LLM, so we can't reliably parse numbers out of the
prose. Instead, viz-producing tools append a hidden marker carrying a JSON payload;
the agent extracts it from the tool result (the LLM never sees it) and the /chat
response returns it as a separate `viz` field that the frontend renders as a card.
"""

from __future__ import annotations

import re
import json

_OPEN, _CLOSE = "<<VIZ>>", "<<ENDVIZ>>"
_RE = re.compile(re.escape(_OPEN) + r"(.*?)" + re.escape(_CLOSE), re.DOTALL)


def embed_viz(text: str, viz: dict | None) -> str:
    """Append a hidden viz payload to a tool's text output."""
    if not viz:
        return text
    return f"{text}\n{_OPEN}{json.dumps(viz, ensure_ascii=False)}{_CLOSE}"


def split_viz(text: str):
    """Return (clean_text, viz_dict_or_None). Safe on text with no marker."""
    if not text or _OPEN not in text:
        return text, None
    m = _RE.search(text)
    if not m:
        return text, None
    try:
        viz = json.loads(m.group(1))
    except Exception:
        viz = None
    clean = _RE.sub("", text).strip()
    return clean, viz
