from sqlalchemy import inspect, text

from app.db.session import engine

if __name__ == "__main__":
    with engine.connect() as conn:
        print("Database:", conn.scalar(text("SELECT current_database()")))
        print("Tables in public:", inspect(conn).get_table_names(schema="public"))
