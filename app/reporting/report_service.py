from __future__ import annotations

from datetime import datetime

from app.db.reports import create_report as db_create_report
from app.db.reports import update_report
from fastapi import BackgroundTasks

from app.db.session import get_db
from app.llm.client import LLMClient
from app.llm.service import LLMService
from app.reporting.pdf_renderer import PDFRenderer


def generate_report_background(
    run_id: str,
    db,
    llm_service: LLMService,
    pdf_path: str,
) -> None:
    """Background task to generate a report for a run.

    Creates the report JSON, renders PDF, and updates the report status.
    """
    try:
        # Get all incidents for this run
        from app.db.repositories.jobs import get_jobs_by_run
        jobs_list = get_jobs_by_run(run_id, db)

        # Get predictions and events to build incident data
        incidents_data = []
        for job in jobs_list:
            if job.status == "done":
                # Get events for this job
                from app.db.repositories.events import get_events_by_job
                events_list = get_events_by_job(db, job.job_id)
                for event in events_list:
                    # Get predictions for this event
                    from app.db.repositories.predictions import get_predictions_by_event
                    preds = get_predictions_by_event(db, event.event_id)
                    for pred in preds:
                        incidents_data.append({
                            "entity": event.entity_value,
                            "unified_class": pred.get("unified_class"),
                            "tactic": pred.get("tactic"),
                            "severity": pred.get("severity", "low"),
                            "score": pred.get("confidence", 0.0),
                            "entity_type": event.entity_type,
                            "lanes": [job.lane_id],
                            "flags": [],
                        })

        # Generate report JSON using LLM or fallback
        report, used_fallback = llm_service.generate_report(run_id, incidents_data, db)

        # Store report in database
        report_row = db_create_report(
            db,
            run_id=run_id,
            status="pending",
            report_json=report.model_dump(),
            llm_provider=report.llm_provider,
            llm_model=report.llm_model,
            prompt_version=report.prompt_version,
            used_fallback=used_fallback,
        )

        # Render PDF from stored report JSON
        renderer = PDFRenderer(report.model_dump(), pdf_path)
        renderer.render()

        # Update report status
        update_report(
            db,
            report_id=report_row.report_id,
            status="completed",
            report_json=report.model_dump(),
            generated=datetime.now(),
            file_path=pdf_path,
        )

    except Exception as e:
        print(f"Report generation failed: {e}")
        # Try fallback report generation
        try:
            db_session = get_db()
            from app.db.reports import get_report as db_get_report
            report_row = db_get_report(db_session, run_id)
            if report_row and report_row.used_fallback:
                # Render minimal PDF even with fallback
                pdf_path = f"outputs/reports/report_{run_id}.pdf"
                renderer = PDFRenderer(report_row.report_json, pdf_path)
                renderer.render()
                update_report(
                    db_session,
                    report_id=report_row.report_id,
                    status="completed",
                    file_path=pdf_path,
                    generated=datetime.now(),
                )
        except Exception as e2:
            print(f"Fallback report also failed: {e2}")


def create_report(
    run_id: str,
    background_tasks: BackgroundTasks,
    db=None,
) -> dict[str, str]:
    """Create a report for a run.

    Triggers LLM service in background, renders PDF, updates status.
    """
    if db is None:
        db = get_db()

    # Create report row initially
    from app.db.reports import create_report as db_create_report
    report_row = db_create_report(db, run_id=run_id, status="pending")

    # Get LLM service from app state or create default
    # In production, would get from app state
    from app.core.config import settings

    from app.llm.service import LLMService

    client = LLMClient(
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    llm_service = LLMService(client, prompt_version=settings.LLM_PROMPT_VERSION)

    # Generate report in background
    pdf_path = f"outputs/reports/report_{run_id}.pdf"
    background_tasks.add_task(
        generate_report_background,
        run_id,
        db,
        llm_service,
        pdf_path,
    )

    return {"report_id": report_row.report_id, "status": "pending"}
