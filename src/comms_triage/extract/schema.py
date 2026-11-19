from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExtractedFields(BaseModel):
    urgency: Literal["low", "medium", "high"] = "low"
    deadline: str | None = None          # the phrase as written, e.g. "Friday", "today 3pm"
    deadline_date: str | None = None     # ISO date if it could be resolved
    assets: list[str] = Field(default_factory=list)
    summary: str = ""
    method: Literal["llm", "rules"] = "rules"

    @field_validator("assets", mode="before")
    @classmethod
    def _clean_assets(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        return [str(v).strip().lower() for v in value if str(v).strip()][:6]

    @field_validator("urgency", mode="before")
    @classmethod
    def _clean_urgency(cls, value):
        value = str(value or "low").strip().lower()
        return value if value in ("low", "medium", "high") else "low"

    @field_validator("deadline", mode="before")
    @classmethod
    def _clean_deadline(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return None if value.lower() in ("", "none", "null", "n/a", "unknown") else value
