from sqlalchemy.orm import Session

from ..models import Reports


def create_report(session: Session, run_id: str) -> Reports:
    report = Reports(run_id=run_id)
    session.add(report)
    session.flush()
    return report


def get_report(session: Session, report_id: str) -> Reports | None:
    return session.query(Reports).filter(Reports.report_id == report_id).first()


def update_report(
    session: Session,
    report_id: str,
    status: str,
    report_json: dict | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    prompt_version: str | None = None,
    used_fallback: bool = False,
    generated: dict | None = None,
    file_path: str | None = None,
) -> None:
    report = session.query(Reports).filter(Reports.report_id == report_id).first()
    if report:
        report.status = status
        if report_json is not None:
            report.report_json = report_json
        if llm_provider is not None:
            report.llm_provider = llm_provider
        if llm_model is not None:
            report.llm_model = llm_model
        if prompt_version is not None:
            report.prompt_version = prompt_version
        if used_fallback is not None:
            report.used_fallback = used_fallback
        if generated is not None:
            report.generated = generated
        if file_path is not None:
            report.file_path = file_path
