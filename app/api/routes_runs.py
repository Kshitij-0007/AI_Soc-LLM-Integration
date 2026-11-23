from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.api.deps import Database
from app.core.run_service import LaneUnavailableError, UploadTooLargeError, submit_run
from app.db.repositories.incidents import get_incidents_by_run
from app.db.repositories.predictions import run_predictions
from app.db.repositories.runs import get_run, get_run_with_jobs

router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/runs", status_code=202)
def create(
    request: Request,
    db: Database,
    files: Annotated[list[UploadFile], File(alias="files[]")],
    lanes: Annotated[list[str], Form(alias="lanes[]")],
):
    if len(files) != len(lanes):
        raise HTTPException(422, "Provide one lane for each file")
    uploads = [
        (lane, upload.filename or "upload.csv", upload.file)
        for lane, upload in zip(lanes, files, strict=True)
    ]
    try:
        return submit_run(db, uploads, request.app.state.lanes, request.app.state.pools)
    except UploadTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except LaneUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/runs/{run_id}")
def status(run_id: UUID, db: Database):
    result = get_run_with_jobs(db, run_id)
    if result is None:
        raise HTTPException(404, "Run not found")
    return result


@router.get("/runs/{run_id}/results")
def results(
    run_id: UUID,
    db: Database,
    lane: str | None = None,
    min_conf: Annotated[float, Query(ge=0, le=1)] = 0.0,
    limit: Annotated[int, Query(ge=1, le=10000)] = 100,
):
    if get_run(db, run_id) is None:
        raise HTTPException(404, "Run not found")
    rows = run_predictions(db, run_id, lane, min_conf, limit)
    fields = (
        "entity",
        "window",
        "predicted_class",
        "confidence",
        "class_probabilities",
        "unified_class",
        "tactic",
    )
    return {"rows": [{key: row[key] for key in fields} for row in rows]}


@router.get("/runs/{run_id}/incidents")
def incidents(run_id: UUID, db: Database):
    if get_run(db, run_id) is None:
        raise HTTPException(404, "Run not found")
    return {"incidents": get_incidents_by_run(db, run_id)}
