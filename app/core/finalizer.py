from __future__ import annotations

from app.db.repositories.jobs import get_jobs_by_run
from app.orchestration.service import build_incidents


def check_finalizer(run_id: str, db_session) -> None:
    """Check if all jobs of a run are done, and if so, build incidents and finalize.

    Called after each job completion.
    """
    # Get all jobs of this run
    jobs_list = get_jobs_by_run(run_id, db_session)

    # Check status of all jobs
    all_done = all(j.status in ("done", "failed") for j in jobs_list)
    any(j.status == "failed" for j in jobs_list)

    if not all_done:
        return  # Not all jobs completed yet

    # If every job failed, set run status to 'failed'
    if all(j.status == "failed" for j in jobs_list):
        from app.db.repositories.runs import update_run_status
        update_run_status(run_id, "failed", db_session)
        return

    # Otherwise, build incidents
    build_incidents(run_id, db_session)

    # Set run status to 'scored' and set finished_at
    from app.db.repositories.runs import update_run_status
    update_run_status(run_id, "scored", db_session)
