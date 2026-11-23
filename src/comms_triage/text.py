"""Small text helpers shared by the index, the Slack cards and reply drafts."""
from __future__ import annotations


def shorten(text: str, limit: int = 90) -> str:
    """Shorten to at most `limit` characters, cutting at a word boundary and adding an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:.-") + "…"
