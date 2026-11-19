"""The triage pipeline used by the API (and through it by Slack and the dashboard).

classify -> route by confidence -> extract fields -> retrieve related work -> draft reply -> store
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import joblib

from comms_triage.config import ROUTABLE_TEAMS, TEAMS, Settings, get_settings
from comms_triage.db import Database, Request, utcnow
from comms_triage.extract.llm_extract import draft_reply, extract_fields
from comms_triage.llm import LLMClient
from comms_triage.retrieve.index import KnowledgeIndex, build_default

log = logging.getLogger(__name__)


class TriageService:
    def __init__(self, settings: Settings | None = None, db: Database | None = None,
                 llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        artifact = joblib.load(Path(self.settings.model_dir) / "model.joblib")
        self.classifier = artifact["classifier"]
        self.model_name = artifact.get("best_model")
        self.backend = artifact.get("backend")
        self.threshold = (self.settings.confidence_threshold
                          if self.settings.confidence_threshold is not None else artifact["threshold"])
        kb_dir = Path(self.settings.model_dir) / "kb"
        if (kb_dir / "index.faiss").exists():
            self.kb = KnowledgeIndex.load(kb_dir)
        else:
            log.info("No saved knowledge index, building one")
            self.kb = build_default(self.settings, self.backend)
            self.kb.save(kb_dir)
        self.db = db or Database(self.settings.database_url)
        self.llm = llm or LLMClient(self.settings)

    # ---- core -----------------------------------------------------------
    def classify(self, text: str) -> dict:
        probs = self.classifier.predict_proba([text])[0]
        classes = list(self.classifier.classes_)
        ranked = sorted(zip(classes, probs), key=lambda x: -x[1])
        team, conf = ranked[0]
        auto = conf >= self.threshold and team != "Unclear"
        return {"team": team, "confidence": float(conf), "route": "auto" if auto else "review",
                "probabilities": {c: round(float(p), 4) for c, p in ranked}}

    def triage(self, text: str, source: str = "api", requester: str | None = None) -> Request:
        text = text.strip()
        decision = self.classify(text)
        fields = extract_fields(text, self.llm)
        related = self.kb.search(text, k=3, min_score=0.15)
        auto = decision["route"] == "auto"
        reply = draft_reply(text, decision["team"], fields, auto, related, self.llm)
        row = Request(
            source=source, requester=requester, text=text,
            predicted_team=decision["team"], confidence=decision["confidence"],
            probabilities=decision["probabilities"], threshold=self.threshold, route=decision["route"],
            status="routed" if auto else "pending_review",
            final_team=decision["team"] if auto else None,
            urgency=fields.urgency, deadline=fields.deadline, deadline_date=fields.deadline_date,
            assets=fields.assets, summary=fields.summary, extraction_method=fields.method,
            related=related, draft_reply=reply,
        )
        return self.db.add(row)

    def review(self, request_id: int, final_team: str, reviewer: str | None = None) -> Request | None:
        if final_team not in ROUTABLE_TEAMS:
            raise ValueError(f"final_team must be one of {ROUTABLE_TEAMS}")
        return self.db.update(request_id, final_team=final_team, reviewed_by=reviewer,
                              reviewed_at=utcnow(), status="resolved")

    def attach_slack(self, request_id: int, channel: str, ts: str) -> Request | None:
        return self.db.update(request_id, slack_channel=channel, slack_ts=ts)

    # ---- reporting ------------------------------------------------------
    def stats(self) -> dict:
        rows = self.db.list(limit=100000)
        total = len(rows)
        reviewed = [r for r in rows if r.reviewed_at is not None]
        auto = [r for r in rows if r.route == "auto"]
        corrections = [r for r in reviewed if r.final_team != r.predicted_team]
        auto_overridden = [r for r in reviewed if r.route == "auto" and r.final_team != r.predicted_team]
        minutes = [(r.reviewed_at - r.created_at).total_seconds() / 60 for r in reviewed]
        by_team = Counter((r.final_team or r.predicted_team) for r in rows)
        return {
            "total": total,
            "auto_routed": len(auto),
            "auto_rate": round(len(auto) / total, 3) if total else None,
            "pending_review": sum(r.status == "pending_review" for r in rows),
            "reviewed": len(reviewed),
            "review_changed_team": len(corrections),
            "auto_routed_then_overridden": len(auto_overridden),
            "median_minutes_to_review": round(sorted(minutes)[len(minutes) // 2], 1) if minutes else None,
            "by_team": {t: by_team.get(t, 0) for t in TEAMS},
            "by_urgency": dict(Counter(r.urgency for r in rows)),
            "by_source": dict(Counter(r.source for r in rows)),
            "threshold": self.threshold,
            "model": self.model_name,
            "backend": self.backend,
            "llm_enabled": self.llm.enabled,
        }

    def feedback_rows(self) -> list[dict]:
        """Human decisions as labeled data for the next training run (train.py --feedback)."""
        return [{"id": f"fb{r.id}", "text": r.text, "team": r.final_team, "source": "feedback",
                 "predicted_team": r.predicted_team}
                for r in self.db.list(limit=100000) if r.reviewed_at is not None and r.final_team]
