"""Rule based extraction of urgency, deadline and requested assets. Always available, no API needed."""
from __future__ import annotations

import re
from datetime import datetime

from comms_triage.extract.schema import ExtractedFields

WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
TIME = r"(?:\s*(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:am|pm))"

# Ordered from most to least specific; the first match wins.
DEADLINE_PATTERNS = [
    r"\b(?:by|before|at) \d{1,2}(?::\d{2})?\s*(?:am|pm)(?: today| tomorrow| tmrw)?\b",
    r"\bin (?:\d+|one|two|three|a few) (?:hours?|minutes?|days?|weeks?)\b",
    rf"\b(?:next |this )?{WEEKDAYS}(?: morning| afternoon)?{TIME}?\b",
    rf"\b{MONTHS} \d{{1,2}}(?:st|nd|rd|th)?\b",
    rf"\b{MONTHS}\b(?= *(?:[.,!?]|$))",
    r"\bthe \d{1,2}(?:st|nd|rd|th)\b",
    r"\bthe week of the \d{1,2}(?:st|nd|rd|th)\b",
    rf"\btoday{TIME}?\b",
    rf"\b(?:tomorrow|tmrw)(?: morning| afternoon| night)?{TIME}?\b",
    r"\btonight\b",
    r"\bthis (?:afternoon|morning|evening)\b",
    r"\b(?:end of|by end of) (?:next week|this week|the week|day|month|the month|next month|quarter)\b",
    r"\b(?:next|this) (?:week|month|quarter)\b",
    r"\bend of (?:day|month)\b",
    r"\basap\b|\bas soon as possible\b",
    rf"\b(?:by|before|at) \d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)\b",
]

HIGH_URGENCY = [
    r"\basap\b", r"\burgent(?:ly)?\b", r"\bas soon as possible\b", r"\bright away\b", r"\bimmediately\b",
    r"\btoday\b", r"\btonight\b", r"\bthis (?:afternoon|morning|evening)\b", r"\bin (?:\d+|one|two|three|a few) (?:hours?|minutes?)\b",
    r"\bon deadline\b", r"\bspreading\b", r"\bviral\b", r"\bend of day\b", r"\bby end of day\b",
    r"\b(?:tomorrow|tmrw) (?:morning|at|\d)", r"\bimpersonat", r"\boutage\b.*\b(?:reply|statement|respond)",
    r"\blive hit\b", r"\bbroken\b", r"\bfiles? at\b",
]
MEDIUM_URGENCY = [
    rf"\b{WEEKDAYS}\b", rf"\b{MONTHS} \d", r"\bthe \d{1,2}(?:st|nd|rd|th)\b", r"\bnext week\b", r"\bthis week\b",
    r"\bdeadline\b", r"\bdue\b", r"\btomorrow\b", r"\btmrw\b", r"\bin \d+ days\b", r"\bsoon\b", r"\bbefore\b",
]

ASSETS = {
    "press release": r"press release",
    "statement": r"\bstatement\b",
    "quote": r"\bquote\b",
    "comment": r"\bcomment\b",
    "interview": r"\binterview|\bpodcast\b|\bfireside\b|\btv\b",
    "media list": r"media list",
    "briefing": r"\bbrief(?:ing)?\b",
    # "blog post" is not a social post; "post the blog on LinkedIn" still is (via "linkedin").
    "social post": r"(?<!blog )\bpost\b|\b(?:tweet|repost|reshare|retweet|linkedin|instagram|tiktok|reel)\b",
    "thread": r"\bthread\b",
    "video": r"\bvideo\b|\breel\b|\blivestream\b",
    "banner": r"\bbanner\b",
    "slides": r"\bslides?\b|\bdeck\b|\bpresentation\b",
    "infographic": r"\binfographic|\bchart\b",
    "diagram": r"\bdiagram|\bflowchart\b",
    "icons": r"\bicons?\b|\bemoji\b",
    "graphic": r"\bgraphic|\bimage\b|\bthumbnail\b|\bgif\b|\billustration|\bposter\b|\bartwork\b",
    "logo": r"\blogo\b",
    "newsletter item": r"\bnewsletter\b",
    "email": r"\bemail\b",
    "wiki update": r"\b(?:company )?hub\b|\bwiki\b|\bintranet\b",
    "announcement": r"\bannounce(?:ment)?\b",
    "faq": r"\bfaq\b",
    "talking points": r"talking points",
    "speech": r"\bspeech\b|\bremarks\b|\bkeynote\b|\bspeaker notes\b|\btestimony\b",
    "letter": r"\bletter\b|\bnote from\b|\bmemo\b",
    "bio": r"\bbio\b|\bheadshot\b",
    # Only count a blog as the asset when someone is asked to write it, not when it needs a banner.
    "article": r"\barticle\b|\bop-?ed\b|\b(?:write|draft|ghostwrite|writing)\b[^.?!]*\bblog\b",
}


def find_deadline(text: str) -> str | None:
    lower = text.lower()
    for pattern in DEADLINE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return match.group(0).strip()
    return None


def resolve_deadline(phrase: str | None, now: datetime | None = None) -> str | None:
    """Best effort: turn 'Friday' or 'tomorrow 3pm' into an ISO date. Returns None if unsure."""
    if not phrase or phrase in ("asap", "as soon as possible"):
        return None
    try:
        import dateparser
    except ImportError:
        return None
    cleaned = re.sub(r"\btmrw\b", "tomorrow", phrase)
    cleaned = re.sub(r"\b(?:next|this|the week of|by|before|end of|the)\b", " ", cleaned).strip()
    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", cleaned)
    now = now or datetime.now()
    if not re.search(r"\d|day|tomorrow|tonight|morning|afternoon|evening|hour|minute|" + MONTHS, cleaned):
        return None  # "next week", "end of month": too vague to pin to a date
    if re.fullmatch(MONTHS, cleaned):
        return None  # a month alone ("in May") has no day
    parsed = dateparser.parse(cleaned, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now})
    if parsed is None:
        return None
    day = parsed.date()
    if day < now.date():
        # "the 15th" on the 19th means next month; "March 12" in November means next year.
        if re.fullmatch(r"\d{1,2}", cleaned):
            month = now.month % 12 + 1
            year = now.year + (now.month == 12)
            try:
                day = day.replace(year=year, month=month)
            except ValueError:
                return None
        else:
            try:
                day = day.replace(year=day.year + 1)
            except ValueError:
                return None
    return day.isoformat()


def urgency_of(text: str) -> str:
    lower = text.lower()
    if any(re.search(p, lower) for p in HIGH_URGENCY):
        return "high"
    if any(re.search(p, lower) for p in MEDIUM_URGENCY):
        return "medium"
    return "low"


def assets_of(text: str) -> list[str]:
    lower = text.lower()
    return [name for name, pattern in ASSETS.items() if re.search(pattern, lower)]


def summarize(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    first = re.split(r"(?<=[.!?])\s", text)[0]
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def extract_rules(text: str, now: datetime | None = None) -> ExtractedFields:
    deadline = find_deadline(text)
    return ExtractedFields(urgency=urgency_of(text), deadline=deadline,
                           deadline_date=resolve_deadline(deadline, now), assets=assets_of(text),
                           summary=summarize(text), method="rules")
