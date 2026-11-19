from __future__ import annotations

import os

import pytest

# Tests always run offline with the TF-IDF backend and never call a real LLM.
os.environ["EMBEDDING_BACKEND"] = "tfidf"
for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "CONFIDENCE_THRESHOLD", "KNOWLEDGE_FILES"):
    os.environ.pop(var, None)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Train a small model once for the whole test session."""
    from comms_triage.classify.train import main

    out = tmp_path_factory.mktemp("model")
    main(["--backend", "tfidf", "--out", str(out), "--per-seed", "3", "--folds", "3",
          "--candidates", "tfidf_lr,emb_svm"])
    return out


@pytest.fixture()
def service(model_dir, tmp_path, monkeypatch):
    from comms_triage.config import Settings
    from comms_triage.db import Database
    from comms_triage.service import TriageService

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    settings = Settings()
    return TriageService(settings, db=Database(f"sqlite:///{tmp_path / 'test.db'}"))
