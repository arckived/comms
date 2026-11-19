"""Request log. Postgres in Docker, SQLite locally. Every prediction and every human decision is stored,
so review corrections become new labeled training data."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Request(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    source: Mapped[str] = mapped_column(String(20), default="api")       # web, slack, api
    requester: Mapped[str | None] = mapped_column(String(120), nullable=True)
    text: Mapped[str] = mapped_column(Text)

    predicted_team: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    threshold: Mapped[float] = mapped_column(Float)
    route: Mapped[str] = mapped_column(String(10))                       # auto or review
    status: Mapped[str] = mapped_column(String(20), index=True)          # routed, pending_review, resolved

    urgency: Mapped[str] = mapped_column(String(10), default="low")
    deadline: Mapped[str | None] = mapped_column(String(120), nullable=True)
    deadline_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    assets: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    extraction_method: Mapped[str] = mapped_column(String(10), default="rules")
    related: Mapped[list] = mapped_column(JSON, default=list)
    draft_reply: Mapped[str] = mapped_column(Text, default="")

    final_team: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    slack_channel: Mapped[str | None] = mapped_column(String(40), nullable=True)
    slack_ts: Mapped[str | None] = mapped_column(String(40), nullable=True)

    def to_dict(self) -> dict:
        out = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for key in ("created_at", "reviewed_at"):
            if out[key] is not None:
                out[key] = out[key].isoformat()
        return out


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            from pathlib import Path
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self.Session()

    def add(self, row: Request) -> Request:
        with self.session() as s:
            s.add(row)
            s.commit()
            return row

    def get(self, request_id: int) -> Request | None:
        with self.session() as s:
            return s.get(Request, request_id)

    def list(self, status: str | None = None, team: str | None = None, limit: int = 100) -> list[Request]:
        with self.session() as s:
            q = select(Request).order_by(Request.created_at.desc(), Request.id.desc()).limit(limit)
            if status:
                q = q.where(Request.status == status)
            if team:
                q = q.where((Request.final_team == team) |
                            ((Request.final_team.is_(None)) & (Request.predicted_team == team)))
            return list(s.scalars(q))

    def update(self, request_id: int, **fields) -> Request | None:
        with self.session() as s:
            row = s.get(Request, request_id)
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            s.commit()
            return row
