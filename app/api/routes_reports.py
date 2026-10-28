import os

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.reporting.report_service import create_report as create_report_func

router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/runs/{run_id}/report")
def create_report(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a report for a run.

    Triggers LLM service in background, renders PDF, updates status.
    """
    result = create_report_func(run_id, background_tasks, db)
    return result


@router.get("/reports/{report_id}")
def get_report(
    report_id: str,
    db: Session = Depends(get_db),
):
    """Get report JSON."""
    from app.db.reports import get_report as db_get_report
    report = db_get_report(db, report_id)
    if not report:
        return {"error": "Report not found"}
    return report


@router.get("/reports/{report_id}/download")
def download_report(
    report_id: str,
    db: Session = Depends(get_db),
):
    """Download PDF report."""
    from app.db.reports import get_report as db_get_report
    report = db_get_report(db, report_id)
    if not report or not report.file_path:
        return {"error": "PDF not available"}

    if not os.path.exists(report.file_path):
        return {"error": "PDF file not found"}

    return FileResponse(
        report.file_path,
        media_type="application/pdf",
        filename=f"report_{report_id}.pdf",
    )
