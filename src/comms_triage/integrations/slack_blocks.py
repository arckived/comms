"""Slack Block Kit messages. Pure functions (no Slack calls) so they are easy to test."""
from __future__ import annotations

from comms_triage.config import ROUTABLE_TEAMS

URGENCY_EMOJI = {"high": ":red_circle:", "medium": ":large_orange_circle:", "low": ":white_circle:"}


def _fields(req: dict) -> str:
    parts = [f"{URGENCY_EMOJI.get(req['urgency'], '')} *Urgency:* {req['urgency']}"]
    if req.get("deadline"):
        date = f" ({req['deadline_date']})" if req.get("deadline_date") else ""
        parts.append(f"*Deadline:* {req['deadline']}{date}")
    if req.get("assets"):
        parts.append(f"*Needs:* {', '.join(req['assets'][:4])}")
    return "   ".join(parts)


def _related(req: dict) -> list[dict]:
    items = []
    for r in (req.get("related") or [])[:3]:
        label = r["title"]
        if r.get("url"):
            label = f"<{r['url']}|{label}>"
        prefix = "Past request" if r["kind"] == "past_request" else "Announcement"
        items.append(f"• {prefix}: {label}")
    if not items:
        return []
    return [{"type": "context", "elements": [{"type": "mrkdwn", "text": "*Related*\n" + "\n".join(items)}]}]


def team_card(req: dict, permalink: str | None = None, assigned_by: str | None = None) -> list[dict]:
    who = f"<@{req['requester']}>" if req.get("requester") else "someone"
    how = (f"assigned by <@{assigned_by}>" if assigned_by
           else f"auto routed, {req['confidence']:.0%} confident")
    link = f"  <{permalink}|View original>" if permalink else ""
    return [
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": f"*New request for {req['final_team']}* from {who} ({how}){link}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f">{req['text'][:2800]}"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": _fields(req)}]},
        *_related(req),
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"*Suggested reply:* {req['draft_reply'][:1500]}"}]},
    ]


def review_card(req: dict, permalink: str | None = None) -> list[dict]:
    probs = req.get("probabilities") or {}
    top = ", ".join(f"{t} {p:.0%}" for t, p in list(probs.items())[:3])
    link = f"  <{permalink}|View original>" if permalink else ""
    buttons = [{"type": "button", "action_id": f"assign_{i}", "text": {"type": "plain_text", "text": team},
                "value": f"{req['id']}|{team}", **({"style": "primary"} if team == req["predicted_team"] else {})}
               for i, team in enumerate(ROUTABLE_TEAMS)]
    return [
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": f"*Needs a person to route* (request #{req['id']}){link}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f">{req['text'][:2800]}"}},
        {"type": "context", "elements": [{"type": "mrkdwn",
                                          "text": f"Model guess: *{req['predicted_team']}* "
                                                  f"({req['confidence']:.0%}, threshold {req['threshold']:.0%}). "
                                                  f"Top: {top}"}]},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": _fields(req)}]},
        {"type": "actions", "block_id": f"review_{req['id']}", "elements": buttons},
    ]


def resolved_review_card(req: dict, user_id: str) -> list[dict]:
    changed = "" if req["final_team"] == req["predicted_team"] else f" (model said {req['predicted_team']})"
    return [
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": f":white_check_mark: Request #{req['id']} assigned to "
                                             f"*{req['final_team']}* by <@{user_id}>{changed}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f">{req['text'][:2800]}"}},
    ]


def ack_text(req: dict) -> str:
    if req["route"] == "auto":
        return f":incoming_envelope: Sent to *{req['final_team']}*. {req['draft_reply']}"
    return f":hourglass_flowing_sand: {req['draft_reply']}"


def parse_action_value(value: str) -> tuple[int, str]:
    request_id, team = value.split("|", 1)
    if team not in ROUTABLE_TEAMS:
        raise ValueError(f"unknown team {team!r}")
    return int(request_id), team
