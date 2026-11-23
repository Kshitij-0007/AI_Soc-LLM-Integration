from datetime import UTC, datetime

from app.db.repositories.jobs import get_jobs_by_run
from app.db.repositories.runs import lock_run
from app.orchestration.service import build_incidents


def check_finalizer(run_id, db_session):
    run = lock_run(db_session, run_id)
    if run is None or run.finished_at is not None:
        return
    jobs = get_jobs_by_run(db_session, run_id)
    if not jobs or any(job.status not in ("done", "failed") for job in jobs):
        return
    if any(job.status == "done" for job in jobs):
        build_incidents(run_id, db_session)
        run.status = "scored"
    else:
        run.status = "failed"
    run.finished_at = datetime.now(UTC)
