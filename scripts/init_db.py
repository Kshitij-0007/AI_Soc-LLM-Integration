"""Initialize and migrate the database configured in .env."""

import sys
from pathlib import Path

from sqlalchemy import inspect, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.models import Base  # noqa: E402
from app.db.session import engine  # noqa: E402


def initialize(conn):
    conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('ai_soc_schema'))"))
    conn.exec_driver_sql((ROOT / "schema/ai_soc_schema_v2.sql").read_text("utf-8"))
    applied = set(conn.scalars(text("SELECT filename FROM schema_migrations")))
    for path in sorted((ROOT / "migrations").glob("*.sql")):
        if path.name not in applied:
            conn.execute(text(path.read_text("utf-8")))
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (filename, applied_at) "
                    "VALUES (:name, now())"
                ),
                {"name": path.name},
            )
    missing = set(Base.metadata.tables) - set(inspect(conn).get_table_names("public"))
    if missing:
        raise RuntimeError(f"Missing tables: {sorted(missing)}")


def main():
    print(f"Target database: {engine.url.database}")
    with engine.begin() as conn:
        initialize(conn)
    print("Verified all 12 application tables.")


if __name__ == "__main__":
    main()
