from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, TIMESTAMP, Column, MetaData, String, Table, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.domain.models import Event
from src.domain.repositories import EventRepository


def _get_database_url() -> str:
    """
    DATABASE_URL이 없으면 docker-compose 기본값을 기반으로 구성한다.

    - host: POSTGRES_HOST (default: localhost)
    - port: POSTGRES_PORT (default: 5432)
    - user: POSTGRES_USER (default: assistant)
    - pass: POSTGRES_PASSWORD (default: assistant)
    - db:   POSTGRES_DB (default: assistant)
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "assistant")
    password = os.getenv("POSTGRES_PASSWORD", "assistant")
    db = os.getenv("POSTGRES_DB", "assistant")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


class PostgresEventRepository(EventRepository):
    """
    PostgreSQL-backed repository (SQLAlchemy Core).

    MVP:
    - 테이블이 없으면 create_all로 생성
    - insert/list만 제공
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._database_url = database_url or _get_database_url()
        self._engine: Engine = create_engine(self._database_url, pool_pre_ping=True)
        self._metadata = MetaData()
        self._events = Table(
            "events",
            self._metadata,
            Column("id", String, primary_key=True),
            Column("kind", String, nullable=False, index=True),
            Column("payload", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
            Column("created_at", TIMESTAMP(timezone=True), nullable=False, index=True),
        )
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._metadata.create_all(self._engine)
            self._initialized = True
        except SQLAlchemyError as e:
            raise RuntimeError(
                "Failed to initialize Postgres schema. "
                f"database_url={self._database_url!r}. "
                "Run `docker-compose up -d` and/or set DATABASE_URL correctly."
            ) from e

    def insert(self, kind: str, payload: dict) -> Event:
        self._ensure_initialized()
        event_id = str(uuid.uuid4())
        now = datetime.utcnow()
        row = {
            "id": event_id,
            "kind": kind,
            "payload": dict(payload or {}),
            "created_at": now,
        }
        with self._engine.begin() as conn:
            conn.execute(self._events.insert().values(**row))
        return Event(id=event_id, kind=kind, payload=row["payload"], created_at=now)

    def list(self, limit: int = 50) -> List[Event]:
        self._ensure_initialized()
        lim = max(1, int(limit))
        stmt = select(self._events).order_by(self._events.c.created_at.desc()).limit(lim)
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            Event(
                id=str(r["id"]),
                kind=str(r["kind"]),
                payload=dict(r["payload"] or {}),
                created_at=r["created_at"],
            )
            for r in rows
        ]
