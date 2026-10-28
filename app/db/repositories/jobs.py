from sqlalchemy.orm import Session

from ..models import IngestionJobs


def create_job(
    session: Session,
    run_id: str,
    lane_id: str,
    filename: str,
) -> IngestionJobs:
    job = IngestionJobs(run_id=run_id, lane_id=lane_id, filename=filename)
    session.add(job)
    session.flush()
    return job


def get_job(session: Session, job_id: str) -> IngestionJobs | None:
    return session.query(IngestionJobs).filter(IngestionJobs.job_id == job_id).first()


def update_job_status(
    session: Session,
    job_id: str,
    status: str,
    rows_total: int | None = None,
    rows_valid: int | None = None,
    rows_rejected: int | None = None,
    validation_report: dict | None = None,
    error: str | None = None,
) -> None:
    job = session.query(IngestionJobs).filter(IngestionJobs.job_id == job_id).first()
    if job:
        job.status = status
        if rows_total is not None:
            job.rows_total = rows_total
        if rows_valid is not None:
            job.rows_valid = rows_valid
        if rows_rejected is not None:
            job.rows_rejected = rows_rejected
        if validation_report is not None:
            job.validation_report = validation_report
        if error is not None:
            job.error = error
        if status in ("done", "failed"):
            job.completed_at = None  # will be set by caller


def get_jobs_by_run(
    session: Session, run_id: str
) -> list[IngestionJobs]:
    return (
        session.query(IngestionJobs)
        .filter(IngestionJobs.run_id == run_id)
        .order_by(IngestionJobs.created_at)
        .all()
    )
