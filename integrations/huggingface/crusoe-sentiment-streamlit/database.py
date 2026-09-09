"""SQLAlchemy models and session (sync, for Streamlit)."""
import uuid
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime,
    ForeignKey, JSON, create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Session

from config import settings


class Base(DeclarativeBase):
    pass


# Use native UUID for PostgreSQL, String for SQLite
def _uuid_col(**kw):
    try:
        return Column(PG_UUID(as_uuid=True), default=uuid.uuid4, **kw)
    except Exception:
        return Column(String(36), default=lambda: str(uuid.uuid4()), **kw)


class Post(Base):
    __tablename__ = "posts"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(20), nullable=False)
    external_id = Column(String(255), unique=True, index=True)
    url = Column(Text)
    title = Column(Text)
    body = Column(Text)
    author = Column(String(255))
    published_at = Column(DateTime)
    collected_at = Column(DateTime, default=datetime.utcnow)
    raw_json = Column(JSON)
    analyses = relationship("Analysis", back_populates="post", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String(36), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    model_used = Column(String(100))
    sentiment = Column(String(20))
    score = Column(Float)
    summary = Column(Text)
    key_themes = Column(JSON)   # list stored as JSON (SQLite-compatible)
    entities = Column(JSON)
    confidence = Column(Float)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    post = relationship("Post", back_populates="analyses")


class Snapshot(Base):
    __tablename__ = "snapshots"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_at = Column(DateTime, default=datetime.utcnow, index=True)
    overall_score = Column(Float)
    overall_sentiment = Column(String(20))
    reddit_score = Column(Float)
    twitter_score = Column(Float)
    news_score = Column(Float)
    total_posts = Column(Integer)
    synthesis_report = Column(Text)
    top_themes = Column(JSON)


# ── Engine / session ──────────────────────────────────────────────────────────

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if "sqlite" in settings.db_url else {}
        _engine = create_engine(settings.db_url, connect_args=connect_args, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
    return _engine


SessionLocal = None


def get_session_factory():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return SessionLocal


@contextmanager
def db_session() -> Session:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
