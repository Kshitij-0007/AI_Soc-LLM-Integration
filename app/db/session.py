from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

engine = create_engine(
    "postgresql+psycopg://postgres:postgres@localhost:5432/ai_soc",
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False, future=True)


def get_db() -> Session:
    """Dependency to get a DB session."""
    return SessionLocal()


def healthcheck() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
