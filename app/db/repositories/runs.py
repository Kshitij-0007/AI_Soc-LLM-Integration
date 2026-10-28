from sqlalchemy.orm import Session

from ..models import IngestionJobs, Runs


def create_run(session: Session, status: str = "queued") -> Runs:
    run = Runs(status=status)
    session.add(run)
    session.flush()
    return run


def get_run(session: Session, run_id: str) -> Runs | None:
    return session.query(Runs).filter(Runs.run_id == run_id).first()


def update_run_status(session: Session, run_id: str, status: str) -> None:
    run = session.query(Runs).filter(Runs.run_id == run_id).first()
    if run:
        run.status = status


def get_run_with_jobs(session: Session, run_id: str) -> dict | None:
    run = session.query(Runs).filter(Runs.run_id == run_id).first()
    if not run:
        return None
    jobs = (
        session.query(IngestionJobs)
        .filter(IngestionJobs.run_id == run_id)
        .all()
    )
    return {
        "run_id": str(run.run_id),
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "jobs": [
            {
                "job_id": str(j.job_id),
                "lane_id": j.lane_id,
                "filename": j.filename,
                "status": j.status,
                "rows_total": j.rows_total,
                "rows_valid": j.rows_valid,
                "rows_rejected": j.rows_rejected,
                "validation_report": j.validation_report,
                "error": j.error,
            }
            for j in jobs
        ],
    }
