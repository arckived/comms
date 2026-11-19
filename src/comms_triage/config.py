"""All settings come from environment variables, so the same code runs locally, in Docker and in CI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEAMS = ["PR", "Social", "Design", "Internal Comms", "Exec Comms", "Unclear"]
# "Unclear" is never routed automatically: it always goes to the review queue.
ROUTABLE_TEAMS = [t for t in TEAMS if t != "Unclear"]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", str(PROJECT_ROOT / "artifacts" / "robust"))))
    database_url: str = field(default_factory=lambda: _env(
        "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'artifacts' / 'triage.db'}"))

    # "sbert" uses a sentence-transformers model; "tfidf" works fully offline.
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sbert"))
    sbert_model: str = field(default_factory=lambda: _env("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # Overrides the threshold chosen during training, if set.
    confidence_threshold: float | None = field(default_factory=lambda: (
        float(_env("CONFIDENCE_THRESHOLD")) if _env("CONFIDENCE_THRESHOLD") else None))

    # Any OpenAI compatible endpoint: Groq, Cloudflare Workers AI, Ollama, OpenAI...
    llm_base_url: str | None = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str | None = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "20")))

    knowledge_files: list[Path] = field(default_factory=lambda: [
        Path(p) for p in (_env("KNOWLEDGE_FILES") or "").split(",") if p.strip()])

    api_url: str = field(default_factory=lambda: _env("API_URL", "http://localhost:8000"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    def resolved_knowledge_files(self) -> list[Path]:
        if self.knowledge_files:
            return self.knowledge_files
        return sorted((self.data_dir / "knowledge").glob("*announcements*.jsonl")) + sorted(
            (self.data_dir / "knowledge").glob("rss*.jsonl"))


def get_settings() -> Settings:
    return Settings()
