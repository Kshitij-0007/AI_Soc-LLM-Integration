from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db.models import Base, Reports


def create_report(db: Session, run_id: str, status: str = "pending"):
    """Create a new report row."""
    from sqlalchemy.exc import IntegrityError
    try:
        row = Reports(
            run_id=run_id,
            status=status,
            report_json={},
            llm_provider=None,
            llm_model=None,
            prompt_version=None,
            used_fallback=False,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    except IntegrityError:
        db.rollback()
        # Report already exists for this run
        stmt = select(Reports).where(Reports.run_id == run_id)
        row = db.execute(stmt).scalar_one()
        return row


def get_report(db: Session, run_id: str):
    """Get a report by run_id."""
    stmt = select(Reports).where(Reports.run_id == run_id)
    row = db.execute(stmt).scalar_one_or_none()
    return row


def update_report(
    db: Session,
    report_id: str,
    status: str,
    report_json: dict | None = None,
    generated: datetime | None = None,
    file_path: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    prompt_version: str | None = None,
    used_fallback: bool | None = None,
):
    """Update a report row."""
    stmt = select(Reports).where(Reports.report_id == report_id)
    row = db.execute(stmt).scalar_one_or_none()
    if row:
        if status is not None:
            row.status = status
        if report_json is not None:
            row.report_json = report_json
        if generated is not None:
            row.generated_at = generated
        if file_path is not None:
            row.file_path = file_path
        if llm_provider is not None:
            row.llm_provider = llm_provider
        if llm_model is not None:
            row.llm_model = llm_model
        if prompt_version is not None:
            row.prompt_version = prompt_version
        if used_fallback is not None:
            row.used_fallback = used_fallback
        db.commit()
        db.refresh(row)
    return row