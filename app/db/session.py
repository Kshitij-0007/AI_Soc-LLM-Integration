from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, hide_parameters=True)

SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False, future=True)


def get_db() -> Session:
    """Dependency to get a DB session."""
    return SessionLocal()


def healthcheck() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
