import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.discovery import discover_lanes
from app.db.repositories.jobs import create_job
from app.db.repositories.runs import create_run, get_run_with_jobs
from app.db.session import get_db

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/runs")
def create_run(
    files: list[UploadFile] = File(...),
    lanes: list[str] = File(...),
    db: Session = Depends(get_db),
):
    """Create a new run with uploaded files.

    Each file is assigned a lane_id from the lanes list (same order).
    Returns run_id immediately for polling.
    """
    # Validate that all lane_ids exist and are enabled
    lanes_dict = discover_lanes()
    for lane_id in lanes:
        if lane_id not in lanes_dict:
            return {"error": f"Invalid lane_id: {lane_id}"}
        if not lanes_dict[lane_id].is_enabled:
            return {"error": f"Lane {lane_id} is disabled"}

    # Create run
    run_id = str(uuid.uuid4())
    create_run(db, run_id, "queued")

    # Save files and create ingestion jobs
    upload_dir = Path("data/uploads") / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    job_ids = []
    for i, file in enumerate(files):
        lane_id = lanes[i] if i < len(lanes) else lanes[0] if lanes else ""
        # Save file
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create ingestion job
        job_id = str(uuid.uuid4())
        create_job(db, run_id, lane_id, filename)

        # Submit job to lane executor
        from app.execution.runner import LaneExecutor
        executor = LaneExecutor(lane_id=lane_id)
        from app.core.job_service import run_job as rj
        executor.submit(int(job_id), rj, job_id, lane_id, db, run_id)

        job_ids.append(
            {
                "job_id": job_id,
                "lane_id": lane_id,
                "filename": filename,
                "status": "queued",
            }
        )

    return {"run_id": run_id, "jobs": job_ids}


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Get run status and job details."""
    result = get_run_with_jobs(run_id, db)
    if not result:
        return {"error": "Run not found"}
    return result


@router.get("/runs/{run_id}/results")
def get_results(
    run_id: str,
    lane: str = Query(default=""),
    min_conf: float = Query(default=0.0),
    limit: int = Query(default=100),
    db: Session = Depends(get_db),
):
    """Get prediction results for a run.

    Filters by lane (optional), minimum confidence, and result limit.
    """
    from sqlalchemy import select

    # Get predictions joined with events
    select(
        "entity_type",
        "entity_value",
        "window_start",
        "window_end",
        "event_count",
        "predicted_class",
        "confidence",
        "class_probabilities",
        "unified_class",
        "tactic",
    ).select_from(
        # Would join events and predictions tables
    ).where(
        # Would filter by run_id
    )

    if lane:
        # Would filter by lane
        pass

    results = []  # Placeholder

    # Apply limit
    results = results[:limit]

    return {"rows": results}


@router.get("/runs/{run_id}/incidents")
def get_incidents(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Get incidents for a run."""
    from app.db.repositories.incidents import get_incidents_by_run
    incidents = get_incidents_by_run(run_id, db)
    return {"incidents": incidents}
