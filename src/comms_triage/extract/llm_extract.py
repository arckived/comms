"""LLM based field extraction and reply drafting, each with a rule based fallback."""
from __future__ import annotations

import logging
from datetime import datetime

from pydantic import ValidationError

from comms_triage.extract.rules import extract_rules, resolve_deadline
from comms_triage.extract.schema import ExtractedFields
from comms_triage.llm import LLMClient, LLMError

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = """You extract structured fields from requests sent to a company's communications team.
Reply with ONLY a JSON object, no prose, with these keys:
  "urgency":  "low", "medium" or "high"  (high = needed today, within hours, or a live issue)
  "deadline": the deadline phrase exactly as written in the request, or null if none
  "assets":   list of short lowercase names of what is being asked for, e.g. ["press release", "banner"]
  "summary":  one sentence, max 20 words, describing the request
Do not invent a deadline that is not in the text."""

DRAFT_SYSTEM = """You write the first reply to a colleague who sent a request to the communications team.
Be brief and friendly (2 to 4 sentences). Confirm which team will handle it, restate what they need and the
deadline if known, and ask for any missing information. If related past work is provided, mention the most
relevant item by its title. Do not promise outcomes or dates the team has not agreed to. Plain text only."""


def extract_fields(text: str, llm: LLMClient | None = None, now: datetime | None = None) -> ExtractedFields:
    rules = extract_rules(text, now)
    if llm is None or not llm.enabled:
        return rules
    try:
        data = llm.chat_json(EXTRACT_SYSTEM, text, temperature=0.0, max_tokens=300)
        fields = ExtractedFields(**{k: data.get(k) for k in ("urgency", "deadline", "assets", "summary")},
                                 method="llm")
    except (LLMError, ValidationError, TypeError) as exc:
        log.warning("LLM extraction failed, using rules: %s", exc)
        return rules
    # Guard against hallucinated deadlines: the phrase must actually appear in the request.
    if fields.deadline and fields.deadline.lower() not in text.lower():
        fields.deadline = rules.deadline
    fields.deadline_date = resolve_deadline(fields.deadline.lower() if fields.deadline else None, now)
    if not fields.summary:
        fields.summary = rules.summary
    return fields


def template_reply(team: str, fields: ExtractedFields, auto: bool, related: list[dict]) -> str:
    if not auto:
        return ("Thanks for your request! We're confirming which team owns this and will get back to you shortly. "
                "If you can, add the deadline and what you need (for example a post, a quote or a graphic).")
    parts = [f"Thanks! This is with the {team} team."]
    if fields.assets:
        parts.append(f"We noted you need: {', '.join(fields.assets[:3])}.")
    if fields.deadline:
        parts.append(f"Deadline noted as \"{fields.deadline}\".")
    else:
        parts.append("Could you share your deadline?")
    if related:
        parts.append(f"Related: {related[0]['title']}.")
    return " ".join(parts)


def draft_reply(text: str, team: str, fields: ExtractedFields, auto: bool, related: list[dict],
                llm: LLMClient | None = None) -> str:
    fallback = template_reply(team, fields, auto, related)
    if llm is None or not llm.enabled:
        return fallback
    context = "\n".join(f"- {r['title']} ({r.get('url') or r.get('kind')})" for r in related[:3]) or "none"
    prompt = (f"Request: {text}\nOwning team: {team if auto else 'not decided yet, under review'}\n"
              f"Deadline: {fields.deadline or 'unknown'}\nRelated past work:\n{context}")
    try:
        return llm.chat(DRAFT_SYSTEM, prompt, temperature=0.4, max_tokens=200).strip()
    except LLMError as exc:
        log.warning("LLM drafting failed, using template: %s", exc)
        return fallback
