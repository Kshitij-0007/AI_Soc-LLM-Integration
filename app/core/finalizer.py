from __future__ import annotations

from datetime import UTC, datetime

from app.db.repositories.jobs import get_jobs_by_run
from app.db.repositories.runs import get_run, update_run_status


def check_finalizer(run_id: str, db_session) -> None:
    """When no job of the run is queued/running, build incidents and close the run.

    Called after every job completion. Every job failed -> run 'failed'.
    """
    jobs_list = get_jobs_by_run(db_session, run_id)
    if not jobs_list or not all(j.status in ("done", "failed") for j in jobs_list):
        return

    if all(j.status == "failed" for j in jobs_list):
        update_run_status(db_session, run_id, "failed")
    else:
        from app.orchestration.service import build_incidents

        build_incidents(run_id, db_session)
        update_run_status(db_session, run_id, "scored")

    run = get_run(db_session, run_id)
    if run is not None:
        run.finished_at = datetime.now(UTC)
