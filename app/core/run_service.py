"""Stage uploads atomically before handing committed jobs to lane workers."""

import re
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.config import settings
from app.core.finalizer import check_finalizer
from app.core.job_service import run_job
from app.core.lane_spec import LaneSpec
from app.db.repositories.jobs import create_job, update_job_status
from app.db.repositories.lanes import get_lane
from app.db.repositories.model_versions import get_active_model_per_lane
from app.db.repositories.runs import create_run
from app.db.session import SessionLocal
from app.execution.runner import LaneExecutor


class UploadTooLargeError(ValueError):
    pass


class LaneUnavailableError(ValueError):
    pass


def _save(stream: BinaryIO, path: Path) -> None:
    size = 0
    with path.open("xb") as dest:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise UploadTooLargeError(
                    f"Maximum upload size is {settings.MAX_UPLOAD_MB} MB"
                )
            dest.write(chunk)


def _work(job_id: str, path: Path, spec: LaneSpec, run_id: str) -> None:
    with SessionLocal() as session:
        run_job(job_id, path, spec.lane_id, spec, session, run_id)


def submit_run(
    session: Session,
    uploads: list[tuple[str, str, BinaryIO]],
    specs: dict[str, LaneSpec],
    pools: dict[str, LaneExecutor],
) -> dict:
    if not uploads:
        raise ValueError("At least one file is required")
    active = get_active_model_per_lane(session)
    for lane_id in {lane for lane, _, _ in uploads}:
        lane = get_lane(session, lane_id) if lane_id in specs else None
        if lane is None or not lane.is_enabled:
            raise ValueError(f"Unknown or disabled lane: {lane_id}")
        if lane_id not in active:
            raise LaneUnavailableError(f"No active model for lane {lane_id}")

    run_id = str(create_run(session).run_id)
    folder = settings.uploads_path / run_id
    folder.mkdir(parents=True)
    for section in ("raw", "train", "test"):
        (folder / section).mkdir()
    jobs = []
    paths = []
    try:
        for lane_id, name, stream in uploads:
            name = re.sub(r"[^\w.-]", "_", Path(name.replace("\\", "/")).name)
            name = name[:160].strip(".") or "upload.csv"
            job = create_job(session, run_id, lane_id, name)
            job_id = str(job.job_id)
            path = folder / "raw" / f"{job_id}_{name}"
            paths.append(path)
            _save(stream, path)
            jobs.append(
                {
                    "job_id": job_id,
                    "lane_id": lane_id,
                    "filename": name,
                    "status": "queued",
                }
            )
        session.commit()
    except Exception:
        session.rollback()
        for path in paths:
            path.unlink(missing_ok=True)
        for section in ("raw", "train", "test"):
            (folder / section).rmdir()
        folder.rmdir()
        raise

    for job, path in zip(jobs, paths, strict=True):
        lane_id = job["lane_id"]
        try:
            pools[lane_id].submit(
                job["job_id"], _work, job["job_id"], path, specs[lane_id], run_id
            )
        except RuntimeError:
            update_job_status(
                session, job["job_id"], "failed", error="Worker unavailable"
            )
            session.commit()
            check_finalizer(run_id, session)
            session.commit()
            job["status"] = "failed"
    return {"run_id": run_id, "jobs": jobs}
