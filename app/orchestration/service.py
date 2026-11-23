from app.config import settings
from app.db.repositories.incidents import save_incident
from app.db.repositories.predictions import run_predictions
from app.orchestration.correlator import correlate_rules


def build_incidents(run_id, db_session):
    alerts = run_predictions(
        db_session, run_id, min_conf=settings.min_confidence_default, alerts_only=True
    )
    for result in correlate_rules(alerts, run_id, db_session):
        save_incident(db_session, run_id, result)
