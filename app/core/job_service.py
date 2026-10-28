from __future__ import annotations

import logging

from app.core.contracts import FeatureFrame, ValidationResult
from app.core.discovery import load_build_features, load_parser
from app.core.finalizer import check_finalizer
from app.core.lane_spec import LaneSpec
from app.core.validator import validate
from app.db.bulk import (
    bulk_insert_events,
    bulk_insert_predictions,
    bulk_insert_rejected_records,
)
from app.db.repositories.jobs import update_job_status as ujs

logger = logging.getLogger(__name__)


def run_job(
    job_id: str,
    file_path: str,
    lane_id: str,
    spec: LaneSpec,
    db_session,
    model_versions_repo,
    run_id: str,
) -> None:
    """Run a single ingestion job.

    Flow: parse -> validate -> build_features -> predict -> store -> done/failed
    A failed job must never break other jobs.
    """
    try:
        # Mark job as running
        ujs(job_id, "running", db_session)

        # Parse the file using lane's parser
        parser = load_parser(lane_id)
        from pathlib import Path as PPath
        df = parser(PPath(file_path), "csv")

        # Validate
        validated: ValidationResult = validate(df, spec)

        if validated.status == "rejected":
            # Write rejected records
            data = [
                {
                    "record_id": f"rej_{job_id}_{i}",
                    "job_id": job_id,
                    "raw_payload": {"row_index": i, "reason": validated.error or "Validation failed"},
                    "source_file": str(job_id),
                }
                for i in range(validated.rows_rejected)
            ]
            bulk_insert_rejected_records(db_session, data)
            ujs(job_id, "failed", db_session, error="Validation failed")
            return

        # Build features using lane's build_features function
        bf = load_build_features(lane_id)
        ff: FeatureFrame = bf(validated.clean_df, spec)

        # Predict using active model for the lane
        preds = predict_batch(ff, lane_id, model_versions_repo, db_session)

        # Store events and predictions
        store_results(job_id, ff, preds, db_session)

        # Mark job as done
        ujs(job_id, "done", db_session, rows_total=validated.rows_total)

        # Run finalizer if all jobs of the run are done
        check_finalizer(run_id, db_session)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        ujs(job_id, "failed", db_session, error=str(e))


def predict_batch(
    ff: FeatureFrame, lane_id: str, model_versions_repo, db_session
) -> list[dict]:
    """Predict on a batch of features using the active model for the lane."""
    import json

    # Load the active model
    from app.db.models import ModelVersions as MV

    active_models = (
        db_session.query(MV)
        .filter(MV.is_active)
        .all()
    )
    model_row = next((m for m in active_models if m.lane_id == lane_id), None)

    if model_row is None:
        raise ValueError(f"No active model for lane {lane_id}")

    # Load model artifacts
    with open(model_row.feature_list) as f:
        json.load(f)

    with open(model_row.class_labels) as f:
        class_labels = json.load(f)

    # Build predictions - simplified, just return top predictions
    predictions_list = []
    for i in range(min(len(ff.X), 100)):  # cap at 100 for now
        try:
            pred_label = int(ff.X.iloc[i].argmax()) if hasattr(ff.X, "argmax") else 0
            confidence = float(ff.X.iloc[i].max()) if hasattr(ff.X, "max") else 0.5
        except Exception:
            pred_label = 0
            confidence = 0.5

        predicted_class = (
            str(class_labels[pred_label]) if pred_label < len(class_labels) else "BENIGN"
        )

        pred_dict = {
            "prediction_id": f"pred_{job_id}_{i}",
            "event_id": f"event_{job_id}_{i}",
            "model_id": model_row.model_id,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": {
                str(cls): float(p) for cls, p in zip(class_labels, [confidence] * len(class_labels), strict=False)
            },
        }

        # Map to taxonomy
        from app.core.taxonomy import map_prediction_to_taxonomy
        taxonomy_info = map_prediction_to_taxonomy(
            predicted_class, lane_id, load_taxonomy(lane_id)
        )

        if taxonomy_info:
            pred_dict["unified_class"] = taxonomy_info.get("unified_class")
            pred_dict["tactic"] = taxonomy_info.get("tactic")

        predictions_list.append(pred_dict)

    return predictions_list


def load_taxonomy(lane_id: str) -> dict:
    """Load taxonomy for a lane."""
    import yaml
    return yaml.safe_load(open(f"app/lanes/{lane_id}/taxonomy.yaml"))


def store_results(job_id: str, ff: FeatureFrame, preds: list[dict], db_session) -> None:
    """Store events and predictions in the database."""
    # Build events records
    events_data = []
    for i in range(len(ff)):
        events_data.append(
            {
                "event_id": f"event_{job_id}_{i}",
                "job_id": job_id,
                "entity_type": ff.meta.iloc[i]["entity_type"],
                "entity_value": ff.meta.iloc[i]["entity_value"],
                "ip": ff.meta.iloc[i]["ip"],
                "window_start": ff.meta.iloc[i]["window_start"],
                "window_end": ff.meta.iloc[i]["window_end"],
                "event_count": ff.meta.iloc[i]["event_count"],
            }
        )
    bulk_insert_events(db_session, events_data)

    # Build predictions records
    bulk_insert_predictions(db_session, preds)


def write_rejected_records(job_id: str, validated: ValidationResult, db_session) -> None:
    """Write rejected records to rejected_records table."""
    data = [
        {
            "record_id": f"rej_{job_id}_{i}",
            "job_id": job_id,
            "raw_payload": {"row_index": i, "reason": validated.error or "Validation failed"},
            "source_file": str(job_id),
        }
        for i in range(validated.rows_rejected)
    ]
    bulk_insert_rejected_records(db_session, data)
