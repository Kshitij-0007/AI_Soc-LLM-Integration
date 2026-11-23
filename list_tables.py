from sqlalchemy import inspect

from app.db.session import engine

if __name__ == "__main__":
    print("Tables:", inspect(engine).get_table_names(schema="public"))
