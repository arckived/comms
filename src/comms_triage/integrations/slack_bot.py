"""Slack bot (Socket Mode, so no public URL or ngrok is needed).

    python -m comms_triage.integrations.slack_bot

Needs SLACK_BOT_TOKEN (xoxb-...) and SLACK_APP_TOKEN (xapp-...) and a running API (API_URL).
The bot listens in the intake channel, sends each new message to the API, then:
* confident  -> posts a card in the owning team's channel and replies in the thread
* uncertain  -> posts a card with team buttons in the review channel; a click assigns it
"""
from __future__ import annotations

import json
import logging
import os
import re

import httpx

from comms_triage.config import get_settings
from comms_triage.integrations.slack_blocks import (ack_text, parse_action_value, resolved_review_card,
                                                    review_card, team_card)

log = logging.getLogger("slack_bot")

DEFAULT_TEAM_CHANNELS = {
    "PR": "team-pr", "Social": "team-social", "Design": "team-design",
    "Internal Comms": "team-internal-comms", "Exec Comms": "team-exec-comms",
}


def channel_config() -> dict:
    teams = dict(DEFAULT_TEAM_CHANNELS)
    if os.environ.get("SLACK_TEAM_CHANNELS"):
        teams.update(json.loads(os.environ["SLACK_TEAM_CHANNELS"]))
    return {
        "intake": os.environ.get("SLACK_INTAKE_CHANNEL", "comms-requests"),
        "review": os.environ.get("SLACK_REVIEW_CHANNEL", "triage-review"),
        "teams": teams,
    }


def resolve_channel_ids(client, names: list[str]) -> dict[str, str]:
    """Map channel names (without #) to IDs. IDs passed in are kept as they are."""
    wanted = {n.lstrip("#") for n in names}
    found = {n: n for n in wanted if n[:1] in ("C", "G") and n.isupper()}
    cursor = None
    while True:
        resp = client.conversations_list(types="public_channel,private_channel", limit=500, cursor=cursor,
                                         exclude_archived=True)
        for ch in resp["channels"]:
            if ch["name"] in wanted:
                found[ch["name"]] = ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    missing = wanted - set(found)
    if missing:
        log.warning("Channels not found (create them and /invite the bot): %s", ", ".join(sorted(missing)))
    return found


def build_app():
    from slack_bolt import App

    settings = get_settings()
    api = httpx.Client(base_url=settings.api_url, timeout=60)
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    cfg = channel_config()
    ids = resolve_channel_ids(app.client, [cfg["intake"], cfg["review"], *cfg["teams"].values()])
    intake_id = ids.get(cfg["intake"].lstrip("#"))
    review_id = ids.get(cfg["review"].lstrip("#"))
    team_ids = {team: ids.get(name.lstrip("#")) for team, name in cfg["teams"].items()}

    def permalink(client, channel, ts):
        try:
            return client.chat_getPermalink(channel=channel, message_ts=ts)["permalink"]
        except Exception:  # permalink is a nice to have
            return None

    def post_to_team(client, req, link, assigned_by=None):
        channel = team_ids.get(req["final_team"])
        if channel:
            client.chat_postMessage(channel=channel, text=f"New request for {req['final_team']}",
                                    blocks=team_card(req, link, assigned_by))
        else:
            log.warning("No channel for team %s", req["final_team"])

    @app.event("message")
    def on_message(event, client, logger):
        if event.get("channel") != intake_id:
            return
        if event.get("subtype") or event.get("bot_id") or event.get("thread_ts"):
            return  # ignore edits, bot posts and thread replies
        text = (event.get("text") or "").strip()
        if not text:
            return
        resp = api.post("/triage", json={"text": text, "source": "slack", "requester": event.get("user")})
        resp.raise_for_status()
        req = resp.json()
        api.post(f"/requests/{req['id']}/slack", json={"channel": event["channel"], "ts": event["ts"]})
        link = permalink(client, event["channel"], event["ts"])
        if req["route"] == "auto":
            post_to_team(client, req, link)
        elif review_id:
            client.chat_postMessage(channel=review_id, text=f"Request #{req['id']} needs routing",
                                    blocks=review_card(req, link))
        client.chat_postMessage(channel=event["channel"], thread_ts=event["ts"], text=ack_text(req))

    @app.action({"action_id": re.compile(r"^assign_\d+$")})
    def on_assign(ack, body, client, action, logger):
        ack()
        request_id, team = parse_action_value(action["value"])
        user = body["user"]["id"]
        resp = api.post(f"/requests/{request_id}/review", json={"final_team": team, "reviewer": user})
        resp.raise_for_status()
        req = resp.json()
        client.chat_update(channel=body["channel"]["id"], ts=body["message"]["ts"],
                           text=f"Request #{request_id} assigned to {team}",
                           blocks=resolved_review_card(req, user))
        link = permalink(client, req["slack_channel"], req["slack_ts"]) if req.get("slack_ts") else None
        post_to_team(client, req, link, assigned_by=user)
        if req.get("slack_channel") and req.get("slack_ts"):
            client.chat_postMessage(channel=req["slack_channel"], thread_ts=req["slack_ts"],
                                    text=f":incoming_envelope: Now with the *{team}* team.")

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    SocketModeHandler(build_app(), os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
