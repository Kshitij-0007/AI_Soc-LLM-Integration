from sqlalchemy.orm import Session

from app.db.bulk import bulk_insert_events

from ..models import Events


def insert_events(session: Session, records: list[dict]) -> None:
    """Bulk insert events."""
    if not records:
        return
    bulk_insert_events(session, records)


def get_events_by_job(
    session: Session, job_id: str
) -> list[Events]:
    return (
        session.query(Events)
        .filter(Events.job_id == job_id)
        .all()
    )
