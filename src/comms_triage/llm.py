"""A tiny client for any OpenAI compatible chat completions endpoint.

Works with (set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL):
* Groq:                  https://api.groq.com/openai/v1
* Cloudflare Workers AI: https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1
* Ollama (local):        http://localhost:11434/v1   (no key needed)
* OpenAI:                https://api.openai.com/v1

Every caller has a rule based fallback, so the app keeps working with no LLM configured
or when the provider is down or rate limited.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from comms_triage.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
        if not self.enabled:
            raise LLMError("LLM is not configured")
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=self.settings.llm_timeout, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

    def chat_json(self, system: str, user: str, **kwargs) -> dict:
        text = self.chat(system, user, **kwargs)
        return parse_json_object(text)


def parse_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply (models often wrap it in prose or ```)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise LLMError("No JSON object in LLM reply")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Invalid JSON in LLM reply: {exc}") from exc
    raise LLMError("Unbalanced JSON in LLM reply")
