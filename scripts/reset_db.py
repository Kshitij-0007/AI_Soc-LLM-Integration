"""Explicit, transactional reset of the configured database's public tables."""

import argparse

from sqlalchemy import inspect

if __package__:
    from .init_db import engine, initialize
else:
    from init_db import engine, initialize


def main():
    print(f"Target database: {engine.url.database}", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", required=True)
    parser.parse_args()
    with engine.begin() as conn:
        quote = conn.dialect.identifier_preparer.quote_identifier
        for name in inspect(conn).get_table_names(schema="public"):
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS public.{quote(name)} CASCADE")
        initialize(conn)


if __name__ == "__main__":
    main()
