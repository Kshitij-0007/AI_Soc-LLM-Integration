"""Batched inserts for high-volume tables.

``session.execute(text(...), list_of_dicts)`` is an executemany; with
SQLAlchemy 2 + psycopg 3 it is sent as multi-row INSERT batches
("insertmanyvalues"), never row by row. Rows are sent in chunks to bound
memory on large uploads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from sqlalchemy import text
from sqlalchemy.orm import Session

BATCH_ROWS = 5_000


def _chunks(rows: list[dict], size: int = BATCH_ROWS) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _json(value: object) -> str:
    return json.dumps(value if value is not None else {}, default=str)


def _execute(session: Session, sql: str, rows: list[dict]) -> None:
    statement = text(sql)
    for chunk in _chunks(rows):
        session.execute(statement, chunk)


INSERT_EVENTS = (
    "INSERT INTO events (event_id, job_id, entity_type, entity_value, ip, "
    "window_start, window_end, event_count) VALUES (CAST(:event_id AS uuid), "
    "CAST(:job_id AS uuid), :entity_type, :entity_value, CAST(:ip AS inet), "
    ":window_start, :window_end, :event_count)"
)


def bulk_insert_events(session: Session, records: list[dict]) -> None:
    if not records:
        return
    rows = [
        {
            "event_id": r["event_id"],
            "job_id": r["job_id"],
            "entity_type": r["entity_type"],
            "entity_value": r["entity_value"],
            "ip": r.get("ip"),
            "window_start": r["window_start"],
            "window_end": r["window_end"],
            "event_count": int(r.get("event_count", 1)),
        }
        for r in records
    ]
    _execute(session, INSERT_EVENTS, rows)


INSERT_PREDICTIONS = (
    "INSERT INTO predictions (prediction_id, event_id, model_id, predicted_class, "
    "confidence, class_probabilities, unified_class, tactic, shap_top_features) "
    "VALUES (CAST(:prediction_id AS uuid), CAST(:event_id AS uuid), "
    "CAST(:model_id AS uuid), :predicted_class, :confidence, "
    "CAST(:class_probabilities AS jsonb), :unified_class, :tactic, "
    "CAST(:shap_top_features AS jsonb))"
)


def bulk_insert_predictions(session: Session, records: list[dict]) -> None:
    if not records:
        return
    rows = [
        {
            "prediction_id": r["prediction_id"],
            "event_id": r["event_id"],
            "model_id": r["model_id"],
            "predicted_class": r["predicted_class"],
            "confidence": float(r["confidence"]),
            "class_probabilities": _json(r["class_probabilities"]),
            "unified_class": r.get("unified_class"),
            "tactic": r.get("tactic"),
            "shap_top_features": json.dumps(r.get("shap_top_features") or []),
        }
        for r in records
    ]
    _execute(session, INSERT_PREDICTIONS, rows)


INSERT_RAW = (
    "INSERT INTO raw_records (record_id, job_id, raw_payload, source_file) "
    "VALUES (CAST(:record_id AS uuid), CAST(:job_id AS uuid), "
    "CAST(:raw_payload AS jsonb), :source_file)"
)


def bulk_insert_raw_records(session: Session, records: list[dict]) -> None:
    if not records:
        return
    rows = [
        {
            "record_id": r["record_id"],
            "job_id": r["job_id"],
            "raw_payload": _json(r["raw_payload"]),
            "source_file": r["source_file"],
        }
        for r in records
    ]
    _execute(session, INSERT_RAW, rows)


INSERT_REJECTED = (
    "INSERT INTO rejected_records (record_id, job_id, raw_payload, reason, "
    "source_file) VALUES (CAST(:record_id AS uuid), CAST(:job_id AS uuid), "
    "CAST(:raw_payload AS jsonb), :reason, :source_file)"
)


def bulk_insert_rejected_records(session: Session, records: list[dict]) -> None:
    if not records:
        return
    rows = [
        {
            "record_id": r["record_id"],
            "job_id": r["job_id"],
            "raw_payload": _json(r["raw_payload"]),
            "reason": r["reason"],
            "source_file": r["source_file"],
        }
        for r in records
    ]
    _execute(session, INSERT_REJECTED, rows)
