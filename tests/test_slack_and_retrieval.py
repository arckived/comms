import pytest

from comms_triage.config import ROUTABLE_TEAMS, PROJECT_ROOT
from comms_triage.data.io import load_jsonl
from comms_triage.integrations.slack_blocks import (ack_text, parse_action_value, resolved_review_card,
                                                    review_card, team_card)
from comms_triage.retrieve.index import KnowledgeIndex, build_entries, chunk_words

REQ = {
    "id": 7, "text": "Need a banner by Friday", "requester": "U123", "predicted_team": "Design",
    "final_team": "Design", "confidence": 0.93, "threshold": 0.9, "route": "auto", "urgency": "medium",
    "deadline": "friday", "deadline_date": "2026-11-20", "assets": ["banner"],
    "probabilities": {"Design": 0.93, "Social": 0.05, "PR": 0.02}, "draft_reply": "Thanks!",
    "related": [{"kind": "doc", "title": "Brand guidelines", "url": "https://example.com/hub/brand"}],
}


def test_review_card_has_a_button_per_team():
    blocks = review_card(REQ)
    actions = [b for b in blocks if b["type"] == "actions"][0]["elements"]
    assert [a["text"]["text"] for a in actions] == ROUTABLE_TEAMS
    assert len({a["action_id"] for a in actions}) == len(actions)
    assert parse_action_value(actions[2]["value"]) == (7, "Design")
    assert [a for a in actions if a.get("style") == "primary"][0]["text"]["text"] == "Design"


def test_cards_render_and_respect_slack_limits():
    for blocks in (team_card(REQ, "https://slack/link"), review_card(REQ), resolved_review_card(REQ, "U9")):
        assert 0 < len(blocks) <= 50
        for block in blocks:
            text = block.get("text", {}).get("text", "")
            assert len(text) <= 3000
    assert "<https://example.com/hub/brand|Brand guidelines>" in str(team_card(REQ))
    assert "Design" in ack_text(REQ)


def test_parse_action_value_rejects_unknown_team():
    with pytest.raises(ValueError):
        parse_action_value("3|Marketing")


def test_chunking_overlaps():
    words = " ".join(str(i) for i in range(300))
    chunks = chunk_words(words, size=120, overlap=30)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "90"


def test_retrieval_finds_the_right_announcement():
    docs = PROJECT_ROOT / "data" / "knowledge" / "sample_announcements.jsonl"
    kb = KnowledgeIndex.build(build_entries([docs], []), backend="tfidf")
    queries = load_jsonl(PROJECT_ROOT / "data" / "knowledge" / "retrieval_queries.jsonl")
    top3 = sum(q["relevant"][0] in [r["doc_id"] for r in kb.search(q["query"], k=3)] for q in queries)
    assert top3 / len(queries) >= 0.9


def test_shorten_cuts_at_a_word_and_adds_ellipsis():
    from comms_triage.text import shorten

    assert shorten("short text") == "short text"
    out = shorten("TechCrunch is writing about the funding round and wants background info plus a spokesperson", 80)
    assert out.endswith("…") and len(out) <= 80
    assert "spokesperso…" not in out and out[:-1].split()[-1] in "TechCrunch is writing about the funding round and wants background info plus a".split()


def test_kind_filter_finds_docs_even_when_past_requests_dominate():
    docs = PROJECT_ROOT / "data" / "knowledge" / "sample_announcements.jsonl"
    # 200 near identical past requests would fill any top-N candidate list on their own.
    past = [{"id": f"p{i}", "text": f"A reporter wants a comment on the outage, request {i}", "team": "PR"}
            for i in range(200)]
    kb = KnowledgeIndex.build(build_entries([docs], past), backend="tfidf")
    hits = kb.search("A reporter wants a comment on the outage", k=1, kind="doc")
    assert hits and hits[0]["kind"] == "doc"


def test_unclear_past_requests_are_not_indexed():
    entries = build_entries([], [{"id": "a", "text": "test", "team": "Unclear"},
                                 {"id": "b", "text": "Need a banner", "team": "Design"}])
    assert [e["doc_id"] for e in entries] == ["b"]
