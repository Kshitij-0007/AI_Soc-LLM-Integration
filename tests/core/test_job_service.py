from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from app.core import predictor
from app.core.job_service import build_records, score_frame
from app.core.training import train_and_save
from app.core.validator import validate
from app.lanes.netflow.features import FEATURE_SET_VERSION, build_features
from app.lanes.netflow.parser import parse
from app.lanes.netflow.train import canonical_labels

PROJECT_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC = PROJECT_DIR / "data" / "raw" / "netflow" / "netflow.csv"


def test_score_frame_and_records(tmp_path, netflow_spec):
    df = parse(SYNTHETIC)
    df["label"] = canonical_labels(df).fillna("unknown")
    clean = validate(df, netflow_spec).clean_df
    ff = build_features(clean, netflow_spec)
    assert ff.y is not None
    result = train_and_save(
        ff, lane_id="netflow", version="test", models_dir=tmp_path,
        feature_set_version=FEATURE_SET_VERSION, n_jobs=1, cv_folds=0,
    )
    predictor.clear_cache()

    upload = build_features(clean.drop(columns=["label"]), netflow_spec)
    predictions, report = score_frame(
        upload, "netflow", result["artifact_dir"],
        shap_max_rows=10, lane_feature_set=FEATURE_SET_VERSION,
    )
    assert len(predictions) == len(upload.X)
    benign = predictions["predicted_class"] == "BENIGN"
    assert (predictions.loc[benign, "unified_class"] == "Benign").all()
    assert predictions.loc[~benign, "tactic"].notna().all()
    assert report["status"] in {"ok", "warning"}

    job_id = str(uuid.uuid4())
    events, rows = build_records(job_id, str(uuid.uuid4()), upload, predictions)
    assert len(events) == len(rows) == len(upload.X)
    assert [e["event_id"] for e in events] == [r["event_id"] for r in rows]
    assert all(e["job_id"] == job_id for e in events)
    assert events[0]["ip"] is not None
    assert np.isclose(
        sum(rows[0]["class_probabilities"].values()), 1.0, atol=1e-4
    )
