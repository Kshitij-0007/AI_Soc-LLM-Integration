import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import Session, sessionmaker

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/aisoc"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False, future=True)


def create_extension(conn) -> None:
    """Create required PostgreSQL extensions."""
    # uuid-ossp is needed for UUID generation in the schema
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))


def fix_unique_constraints(sql: str) -> str:
    """Replace partial unique indexes with regular unique constraints."""
    # Replace WHERE clauses in unique constraints
    sql = sql.replace(
        'UNIQUE (lane_id, is_active) WHERE is_active = true',
        'UNIQUE (lane_id, is_active)'
    )
    sql = sql.replace(
        'CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_model_per_lane ON model_versions(lane_id) WHERE is_active = true',
        'CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_model_per_lane ON model_versions(lane_id)'
    )
    return sql


def apply_schema(sql_path: Path) -> None:
    """Apply the v2 SQL schema file."""
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # Replace the extension creation line since we already created it
    sql = sql.replace(
        'CREATE EXTENSION IF NOT EXISTS "pg_catalog"."uuid-ossp"',
        "-- Extension already created above"
    )

    # Replace CREATE TYPE IF NOT EXISTS ... AS TEXT since types are already created
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS lane_id_enum AS TEXT',
        '-- lane_id_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS run_status_enum AS TEXT',
        '-- run_status_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS job_status_enum AS TEXT',
        '-- job_status_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS incident_status_enum AS TEXT',
        '-- incident_status_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS severity_enum AS TEXT',
        '-- severity_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS unified_class_enum AS TEXT',
        '-- unified_class_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS predicted_class_enum AS TEXT',
        '-- predicted_class_enum type already created'
    )
    sql = sql.replace(
        'CREATE TYPE IF NOT EXISTS tactic_enum AS TEXT',
        '-- tactic_enum type already created'
    )

    # Fix unique constraints
    sql = fix_unique_constraints(sql)

    with engine.connect() as conn:
        create_extension(conn)
        conn.execute(text(sql))
        conn.commit()


def apply_migrations(migrations_dir: Path) -> None:
    """Apply optional additive migration files in filename order."""
    if not migrations_dir.exists():
        return

    migration_files = sorted(migrations_dir.glob("*.sql"))
    applied = set()

    # Check which migrations have already been applied
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT filename FROM schema_migrations")
        ).fetchall()
        applied = {row.filename for row in rows}

    for mf in migration_files:
        filename = mf.name
        if filename in applied:
            continue
        with open(mf, "r", encoding="utf-8") as f:
            sql = f.read()
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO schema_migrations (filename, applied_at) VALUES (:fn, NOW())"),
                {"fn": filename},
            )
            conn.commit()


def verify_tables() -> None:
    """Verify that all 12 required tables exist; exit non-zero if any is missing."""
    required = {
        "lanes",
        "taxonomy_map",
        "runs",
        "ingestion_jobs",
        "raw_records",
        "rejected_records",
        "events",
        "model_versions",
        "predictions",
        "incidents",
        "incident_evidence",
        "reports",
    }

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        ).fetchall()
    existing = {row.tablename for row in result}

    missing = required - existing
    if missing:
        print(f"ERROR: Missing tables: {missing}", file=sys.stderr)
        sys.exit(1)

    print("OK: All 12 tables verified.")


def main() -> None:
    sql_path = PROJECT_DIR / "schema" / "ai_soc_schema_v2.sql"
    migrations_dir = PROJECT_DIR / "migrations"

    print("Target database: ai_soc")

    # Apply schema
    apply_schema(sql_path)

    # Apply migrations
    apply_migrations(migrations_dir)

    # Verify tables
    verify_tables()


if __name__ == "__main__":
    main()