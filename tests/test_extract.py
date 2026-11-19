import json
from datetime import datetime

import httpx
import pytest

from comms_triage.config import Settings
from comms_triage.extract.llm_extract import extract_fields
from comms_triage.extract.rules import extract_rules, resolve_deadline
from comms_triage.llm import LLMClient, LLMError, parse_json_object

NOW = datetime(2026, 11, 19, 10, 0)  # a Thursday


@pytest.mark.parametrize("text,urgency,deadline", [
    ("A reporter wants a comment on the outage by 4pm today", "high", "by 4pm today"),
    ("Need slides for my talk by Friday", "medium", "friday"),
    ("Can we get a LinkedIn post about the hackathon?", "low", None),
    ("make a graphic 4 the launch tmrw", "medium", "tmrw"),
    ("Draft remarks for the dinner on the 15th", "medium", "the 15th"),
    ("Need a statement ASAP", "high", "asap"),
])
def test_rules(text, urgency, deadline):
    fields = extract_rules(text, NOW)
    assert fields.urgency == urgency
    assert fields.deadline == deadline
    assert fields.method == "rules"


@pytest.mark.parametrize("phrase,expected", [
    ("friday", "2026-11-20"), ("tmrw", "2026-11-20"), ("the 15th", "2026-12-15"),
    ("the 25th", "2026-11-25"), ("march 12", "2027-03-12"), ("next week", None), ("asap", None), ("may", None),
])
def test_resolve_deadline(phrase, expected):
    assert resolve_deadline(phrase, NOW) == expected


def test_assets():
    assert "press release" in extract_rules("Please draft a press release", NOW).assets
    assert "banner" in extract_rules("Can design make a banner?", NOW).assets


def test_parse_json_object_handles_wrapped_replies():
    assert parse_json_object('Sure! ```json\n{"a": {"b": 1}}\n``` done') == {"a": {"b": 1}}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def _llm(handler) -> LLMClient:
    settings = Settings(llm_base_url="https://llm.test/v1", llm_model="test-model", llm_api_key="k")
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def _reply(content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    return handler


def test_llm_extraction_is_used_when_valid():
    llm = _llm(_reply(json.dumps({"urgency": "HIGH", "deadline": "Friday", "assets": ["Quote", "bio"],
                                  "summary": "Quote for analyst report"})))
    fields = extract_fields("Can we get a quote approved by Friday?", llm, NOW)
    assert fields.method == "llm"
    assert fields.urgency == "high"
    assert fields.assets == ["quote", "bio"]
    assert fields.deadline_date == "2026-11-20"


def test_llm_hallucinated_deadline_is_dropped():
    llm = _llm(_reply('{"urgency": "low", "deadline": "next Monday", "assets": [], "summary": "x"}'))
    fields = extract_fields("Could we do a LinkedIn post about the team?", llm, NOW)
    assert fields.deadline is None


def test_llm_failure_falls_back_to_rules():
    llm = _llm(lambda request: httpx.Response(429, json={"error": "rate limited"}))
    fields = extract_fields("Need a statement ASAP", llm, NOW)
    assert fields.method == "rules"
    assert fields.urgency == "high"
