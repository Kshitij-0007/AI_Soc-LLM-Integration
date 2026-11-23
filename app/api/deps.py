from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import SessionLocal


def request_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session


Database = Annotated[Session, Depends(request_session)]
