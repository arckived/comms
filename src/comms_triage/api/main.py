"""FastAPI service.

    uvicorn comms_triage.api.main:app --reload

POST /triage                  classify, extract, retrieve and store a request
POST /requests/{id}/review    a person confirms or changes the team
GET  /requests                list requests (filters: status, team)
GET  /requests/{id}           one request
GET  /stats                   numbers for the dashboard
GET  /model                   training metrics of the loaded model
GET  /search?q=               search past requests and announcements
GET  /feedback                reviewed requests as JSONL, for retraining
GET  /health
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request as HttpRequest
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from comms_triage.config import ROUTABLE_TEAMS
from comms_triage.service import TriageService


class TriageIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    source: str = Field(default="api", max_length=20)
    requester: str | None = Field(default=None, max_length=120)


class ReviewIn(BaseModel):
    final_team: str
    reviewer: str | None = Field(default=None, max_length=120)


class SlackRefIn(BaseModel):
    channel: str
    ts: str


def create_app(service: TriageService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or TriageService()
        yield

    app = FastAPI(title="Comms Triage API", version="0.1.0", lifespan=lifespan)

    def svc(request: HttpRequest) -> TriageService:
        return request.app.state.service

    @app.get("/health")
    def health(s: TriageService = Depends(svc)):
        return {"status": "ok", "model": s.model_name, "backend": s.backend, "threshold": s.threshold,
                "llm_enabled": s.llm.enabled}

    @app.post("/triage")
    def triage(body: TriageIn, s: TriageService = Depends(svc)):
        if not body.text.strip():
            raise HTTPException(422, "text is empty")
        return s.triage(body.text, source=body.source, requester=body.requester).to_dict()

    @app.post("/requests/{request_id}/review")
    def review(request_id: int, body: ReviewIn, s: TriageService = Depends(svc)):
        if body.final_team not in ROUTABLE_TEAMS:
            raise HTTPException(422, f"final_team must be one of {ROUTABLE_TEAMS}")
        row = s.review(request_id, body.final_team, body.reviewer)
        if row is None:
            raise HTTPException(404, "request not found")
        return row.to_dict()

    @app.post("/requests/{request_id}/slack")
    def slack_ref(request_id: int, body: SlackRefIn, s: TriageService = Depends(svc)):
        row = s.attach_slack(request_id, body.channel, body.ts)
        if row is None:
            raise HTTPException(404, "request not found")
        return row.to_dict()

    @app.get("/requests")
    def list_requests(status: str | None = None, team: str | None = None,
                      limit: int = Query(100, ge=1, le=1000), s: TriageService = Depends(svc)):
        return [r.to_dict() for r in s.db.list(status=status, team=team, limit=limit)]

    @app.get("/requests/{request_id}")
    def get_request(request_id: int, s: TriageService = Depends(svc)):
        row = s.db.get(request_id)
        if row is None:
            raise HTTPException(404, "request not found")
        return row.to_dict()

    @app.get("/stats")
    def stats(s: TriageService = Depends(svc)):
        return s.stats()

    @app.get("/model")
    def model_info(s: TriageService = Depends(svc)):
        model_dir = Path(s.settings.model_dir)
        out = {}
        for name in ("metrics", "robustness", "retrieval", "extraction_rules", "extraction_llm"):
            path = model_dir / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text())
        return out

    @app.get("/search")
    def search(q: str = Query(min_length=1), k: int = Query(5, ge=1, le=20), kind: str | None = None,
               s: TriageService = Depends(svc)):
        return s.kb.search(q, k=k, kind=kind)

    @app.get("/feedback", response_class=PlainTextResponse)
    def feedback(s: TriageService = Depends(svc)):
        return "\n".join(json.dumps(r) for r in s.feedback_rows())

    return app


app = create_app()
