import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/aisoc"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


def create_extension(conn) -> None:
    """Create required PostgreSQL extensions."""
    # uuid-ossp is needed for UUID generation in the schema
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))


def drop_all_tables() -> None:
    """Drop every table in the public schema, including old ones."""
    with engine.connect() as conn:
        inspector = sa_inspect(conn)
        tables = inspector.get_table_names(schema="public")
        for table in tables:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
        conn.commit()


def main() -> None:
    if "--yes" in sys.argv:
        answer = "yes"
    else:
        print("Target database: aisoc")
        answer = input(
            "This will DROP ALL TABLES in the public schema and re-apply the v2 schema. "
            "Are you sure? Type 'yes' to confirm: "
        )
    if answer.strip() != "yes":
        print("Aborted.")
        sys.exit(0)

    drop_all_tables()

    # Re-apply v2 schema
    sql_path = PROJECT_DIR / "schema" / "ai_soc_schema_v2.sql"
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with engine.connect() as conn:
        create_extension(conn)
        # Replace the extension creation line since we already created it
        sql = sql.replace(
            'CREATE EXTENSION IF NOT EXISTS "pg_catalog"."uuid-ossp"',
            "-- Extension already created above"
        )
        conn.execute(text(sql))
        conn.commit()

    # Ensure schema_migrations table exists
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ)"
        )
    )
    conn.commit()

    print("Schema v2 re-applied successfully.")


if __name__ == "__main__":
    main()