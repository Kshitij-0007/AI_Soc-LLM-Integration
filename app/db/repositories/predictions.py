from sqlalchemy import select

from app.db.models import Events, IngestionJobs, Predictions


def run_predictions(
    session, run_id, lane=None, min_conf=0.0, limit=None, alerts_only=False
):
    stmt = (
        select(
            Events.entity_value.label("entity"),
            Events.window_start.label("window"),
            Events.event_id,
            Predictions.predicted_class,
            Predictions.confidence,
            Predictions.class_probabilities,
            Predictions.unified_class,
            Predictions.tactic,
            IngestionJobs.lane_id.label("lanes"),
        )
        .select_from(Predictions)
        .join(Events, Events.event_id == Predictions.event_id)
        .join(IngestionJobs, IngestionJobs.job_id == Events.job_id)
        .where(IngestionJobs.run_id == run_id, Predictions.confidence >= min_conf)
        .order_by(Events.window_start, Events.event_id)
    )
    if lane:
        stmt = stmt.where(IngestionJobs.lane_id == lane)
    if alerts_only:
        stmt = stmt.where(Predictions.predicted_class != "BENIGN")
    if limit is not None:
        stmt = stmt.limit(limit)
    return [
        dict(row, confidence=float(row["confidence"]))
        for row in session.execute(stmt).mappings()
    ]
