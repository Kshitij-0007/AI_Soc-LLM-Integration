"""Run one ingestion job: parse -> validate -> features -> predict -> store.

A failed job records its error and never raises, so other jobs keep running.
DB access goes through ``app.db`` repositories only.
"""

from __future__ import annotations

import ipaddress
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.core import predictor
from app.core.contracts import FeatureFrame, ValidationResult
from app.core.discovery import LANES_DIR, load_build_features, load_parser
from app.core.lane_spec import LaneSpec
from app.core.resources import xgb_threads
from app.core.taxonomy import load_taxonomy_yaml, map_prediction_to_taxonomy
from app.core.validator import validate

logger = logging.getLogger(__name__)

_FORMATS_BY_SUFFIX = {".csv": "csv", ".jsonl": "jsonl", ".json": "json", ".log": "csv"}


def detect_file_format(path: Path, spec: LaneSpec) -> str:
    fmt = _FORMATS_BY_SUFFIX.get(path.suffix.lower())
    if fmt is None:
        return spec.formats[0] if spec.formats else "csv"
    return fmt


def missing_columns_message(validated: ValidationResult, spec: LaneSpec) -> str:
    """Readable rejection text listing missing columns and the aliases tried."""
    parts = []
    for name in validated.missing_required:
        column = spec.columns.get_by_name(name)
        aliases = column.aliases if column else []
        parts.append(f"{name} (also tried: {', '.join(aliases)})" if aliases else name)
    message = "missing required columns: " + "; ".join(parts)
    if spec.source_formats:
        message += (
            f". Lane '{spec.lane_id}' accepts these exports: "
            + ", ".join(spec.source_formats)
            + ". Check that the right log type was selected."
        )
    return message


def score_frame(
    ff: FeatureFrame,
    lane_id: str,
    model_path: str | Path,
    shap_top_k: int = 5,
    shap_max_rows: int = 2000,
    lane_feature_set: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Predictions with taxonomy columns plus the compatibility report.

    Pure function (no DB), used by run_job and scripts/score_file.py.
    """
    model = predictor.load_model(model_path, threads=xgb_threads())
    predictor.check_feature_set(model, lane_feature_set)
    predictions = predictor.predict(
        model, ff.X, shap_top_k=shap_top_k, shap_max_rows=shap_max_rows
    )
    taxonomy = load_taxonomy_yaml(lane_id)
    mapping = {
        label: map_prediction_to_taxonomy(label, lane_id, taxonomy) or {}
        for label in model.class_labels
    }
    predictions["unified_class"] = predictions["predicted_class"].map(
        lambda c: mapping.get(c, {}).get("unified_class")
    )
    predictions["tactic"] = predictions["predicted_class"].map(
        lambda c: mapping.get(c, {}).get("tactic")
    )
    report = predictor.compatibility_report(model, ff.X, predictions)
    return predictions, report


def _lane_feature_set(lane_id: str) -> str | None:
    import importlib

    try:
        module = importlib.import_module(f"app.lanes.{lane_id}.features")
    except ImportError:
        return None
    return getattr(module, "FEATURE_SET_VERSION", None)


def _valid_ip(value: object) -> str | None:
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError:
        return None


def _timestamp(value: object, fallback: datetime) -> datetime:
    stamp = pd.Timestamp(value) if value is not None else pd.NaT
    if pd.isna(stamp):
        return fallback
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.to_pydatetime()


def build_records(
    job_id: str,
    model_id: str,
    ff: FeatureFrame,
    predictions: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Event and prediction rows linked by generated UUIDs."""
    now = datetime.now(UTC)
    events, rows = [], []
    meta = ff.meta.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    for i in range(len(meta)):
        event_id = str(uuid.uuid4())
        m = meta.iloc[i]
        start = _timestamp(m.get("window_start"), now)
        events.append(
            {
                "event_id": event_id,
                "job_id": job_id,
                "entity_type": str(m.get("entity_type", "unknown")),
                "entity_value": str(m.get("entity_value", "unknown")),
                "ip": _valid_ip(m.get("ip")),
                "window_start": start,
                "window_end": _timestamp(m.get("window_end"), start),
                "event_count": int(m.get("event_count", 1) or 1),
            }
        )
        p = predictions.iloc[i]
        rows.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "event_id": event_id,
                "model_id": model_id,
                "predicted_class": p["predicted_class"],
                "confidence": float(p["confidence"]),
                "class_probabilities": p["class_probabilities"],
                "unified_class": p.get("unified_class"),
                "tactic": p.get("tactic"),
                "shap_top_features": p.get("shap_top_features") or [],
            }
        )
    return events, rows


def run_job(
    job_id: str,
    file_path: str | Path,
    lane_id: str,
    spec: LaneSpec,
    db_session,
    run_id: str,
    lane_dir: Path | None = None,
    shap_top_k: int = 5,
    shap_max_rows: int = 2000,
) -> None:
    """Run a single ingestion job. Never raises."""
    from app.db.bulk import (
        bulk_insert_events,
        bulk_insert_predictions,
        bulk_insert_rejected_records,
    )
    from app.db.repositories.jobs import update_job_status
    from app.db.repositories.model_versions import get_active_model_per_lane

    lane_dir = lane_dir or LANES_DIR / lane_id
    path = Path(file_path)
    try:
        update_job_status(db_session, job_id, "running")
        db_session.commit()

        parse = load_parser(lane_id, lane_dir)
        df = parse(path, detect_file_format(path, spec))
        validated = validate(df, spec)
        report = dict(validated.validation_report)
        report["source_format"] = df.attrs.get("source_format")

        if validated.status == "rejected":
            message = missing_columns_message(validated, spec)
            bulk_insert_rejected_records(
                db_session,
                [
                    {
                        "record_id": str(uuid.uuid4()),
                        "job_id": job_id,
                        "raw_payload": {"columns": [str(c) for c in df.columns]},
                        "reason": message,
                        "source_file": path.name,
                    }
                ],
            )
            update_job_status(
                db_session,
                job_id,
                "failed",
                rows_total=validated.rows_total,
                rows_valid=0,
                rows_rejected=validated.rows_total,
                validation_report=report,
                error=message,
            )
            db_session.commit()
            return

        build_features = load_build_features(lane_id, lane_dir)
        ff: FeatureFrame = build_features(validated.clean_df, spec)

        model_row = get_active_model_per_lane(db_session).get(lane_id)
        if model_row is None:
            raise ValueError(
                f"No active model for lane '{lane_id}'. "
                f"Run: python scripts/register_models.py {lane_id} <version> --activate"
            )

        predictions, compatibility = score_frame(
            ff,
            lane_id,
            model_row.model_path,
            shap_top_k=shap_top_k,
            shap_max_rows=shap_max_rows,
            lane_feature_set=_lane_feature_set(lane_id),
        )
        report["model_compatibility"] = compatibility
        if compatibility["status"] == "warning" and report.get("status") == "ok":
            report["status"] = "warning"

        events, prediction_rows = build_records(
            job_id, str(model_row.model_id), ff, predictions
        )
        bulk_insert_events(db_session, events)
        bulk_insert_predictions(db_session, prediction_rows)
        update_job_status(
            db_session,
            job_id,
            "done",
            rows_total=validated.rows_total,
            rows_valid=validated.rows_valid,
            rows_rejected=validated.rows_rejected,
            validation_report=report,
        )
        db_session.commit()
    except Exception as error:  # noqa: BLE001 - a job must never break others
        logger.exception("Job %s failed", job_id)
        db_session.rollback()
        message = (
            str(error) if isinstance(error, ValueError)
            else "Scoring failed. See server logs for the job ID."
        )
        update_job_status(db_session, job_id, "failed", error=message)
        db_session.commit()
    finally:
        try:
            from app.core.finalizer import check_finalizer

            check_finalizer(run_id, db_session)
            db_session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Finalizer failed for run %s", run_id)
            db_session.rollback()
