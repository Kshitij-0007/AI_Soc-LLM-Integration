from sqlalchemy import text
from sqlalchemy.orm import Session


def bulk_insert_events(session: Session, records: list[dict]) -> None:
    if not records:
        return
    data = [
        (
            r["event_id"],
            r["job_id"],
            r["entity_type"],
            r["entity_value"],
            r.get("ip"),
            r["window_start"],
            r["window_end"],
            r.get("event_count", 1),
        )
        for r in records
    ]
    INSERT_EVENTS = (
        "INSERT INTO events (event_id, job_id, entity_type, entity_value, ip, "
        "window_start, window_end, event_count) VALUES %s"
    )
    session.execute(text(INSERT_EVENTS), data)


def bulk_insert_predictions(session: Session, records: list[dict]) -> None:
    if not records:
        return
    data = [
        (
            r["prediction_id"],
            r["event_id"],
            r["model_id"],
            r["predicted_class"],
            float(r["confidence"]),
            r["class_probabilities"],
            r.get("unified_class"),
            r.get("tactic"),
            r.get("shap_top_features", []),
        )
        for r in records
    ]
    INSERT_PREDICTIONS = (
        "INSERT INTO predictions (prediction_id, event_id, model_id, predicted_class, "
        "confidence, class_probabilities, unified_class, tactic, shap_top_features) VALUES %s"
    )
    session.execute(text(INSERT_PREDICTIONS), data)


def bulk_insert_raw_records(session: Session, records: list[dict]) -> None:
    if not records:
        return
    data = [
        (r["record_id"], r["job_id"], r["raw_payload"], r["source_file"])
        for r in records
    ]
    INSERT_RAW = "INSERT INTO raw_records (record_id, job_id, raw_payload, source_file) VALUES %s"
    session.execute(text(INSERT_RAW), data)


def bulk_insert_rejected_records(session: Session, records: list[dict]) -> None:
    if not records:
        return
    data = [
        (r["record_id"], r["job_id"], r["raw_payload"], r["reason"], r["source_file"])
        for r in records
    ]
    INSERT_REJECTED = (
        "INSERT INTO rejected_records (record_id, job_id, raw_payload, reason, source_file) VALUES %s"
    )
    session.execute(text(INSERT_REJECTED), data)


def bulk_insert_model_versions(session: Session, records: list[dict]) -> None:
    if not records:
        return
    data = [
        (
            r["model_id"],
            r["lane_id"],
            r["version"],
            r["model_path"],
            r["class_labels"],
            r["feature_list"],
            r.get("train_stats"),
            r.get("metrics"),
            r.get("data_source"),
            bool(r.get("is_active", False)),
            r.get("activated_at"),
        )
        for r in records
    ]
    INSERT_MODEL = (
        "INSERT INTO model_versions (model_id, lane_id, version, model_path, "
        "class_labels, feature_list, train_stats, metrics, data_source, is_active, activated_at) VALUES %s"
    )
    session.execute(text(INSERT_MODEL), data)
