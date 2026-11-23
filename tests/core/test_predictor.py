from __future__ import annotations

import json
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from app.core import predictor
from app.core.contracts import FeatureFrame
from app.core.model_manifest import load_manifest, model_version_dir
from app.core.training import train_and_save


def test_netflow_folder_preserves_old_registration_and_lane_id(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor, "PROJECT_DIR", tmp_path)
    folder = model_version_dir(tmp_path / "models", "netflow", "v1")
    folder.mkdir(parents=True)
    (folder / "model.json").touch()
    assert folder == tmp_path / "models/netflow_model/v1"
    assert predictor.resolve_model_dir("models/netflow/v1") == folder
    assert predictor.resolve_model_dir("models/netflow/v1/model.json") == folder
    assert load_manifest(folder).lane_id == "netflow"


def _toy_frame(n: int = 300, seed: int = 0) -> FeatureFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    labels = np.where(a > 0.8, "Attack", np.where(b > 0.8, "Scan", "BENIGN"))
    X = pd.DataFrame({"a": a, "b": b, "c": rng.normal(0, 1, n)})
    X.loc[::10, "c"] = np.nan
    return FeatureFrame(X=X, meta=pd.DataFrame(index=X.index), y=pd.Series(labels))


@pytest.fixture
def trained(tmp_path):
    ff = _toy_frame()
    result = train_and_save(
        ff,
        lane_id="toy",
        version="1.0.0",
        models_dir=tmp_path,
        eval_frame=_toy_frame(seed=1),
        feature_set_version="7",
        n_jobs=1,
        cv_folds=3,
    )
    predictor.clear_cache()
    return result


def test_training_writes_manifest_and_stats(trained, tmp_path):
    model_dir = tmp_path / "toy" / "1.0.0"
    manifest = load_manifest(model_dir)
    assert manifest.missing_values == "native"
    assert manifest.feature_set_version == "7"
    stats = json.loads((model_dir / "train_stats.json").read_text())
    assert set(stats["feature_stats"]) == {"a", "b", "c"}
    assert stats["feature_stats"]["c"]["missing_rate"] > 0
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["holdout"] is not None
    assert metrics["macro_f1"] == metrics["holdout"]["macro_f1"]
    assert metrics["macro_f1"] > 0.8


def test_replacing_a_lane_model_releases_cached_booster(trained):
    folder = Path(trained["artifact_dir"])
    old = weakref.ref(predictor.load_model(folder))
    moved = folder.rename(folder.with_name("replacement"))
    current = predictor.load_model(moved)
    assert predictor.load_model(moved) is current
    assert old() is None


def test_predict_outputs_contract(trained):
    model = predictor.load_model(trained["artifact_dir"])
    X = _toy_frame(seed=2).X
    out = predictor.predict(model, X, shap_top_k=2, shap_max_rows=5)
    assert list(out.columns) == [
        "predicted_class", "confidence", "class_probabilities", "shap_top_features",
    ]
    assert len(out) == len(X)
    totals = out["class_probabilities"].map(lambda d: sum(d.values()))
    assert np.allclose(totals, 1.0, atol=1e-4)
    explained = out["shap_top_features"].map(len)
    assert explained.max() == 2
    assert (explained > 0).sum() <= 5
    assert (out.loc[explained > 0, "predicted_class"] != "BENIGN").all()


def test_feature_mismatch_is_rejected(trained):
    model = predictor.load_model(trained["artifact_dir"])
    with pytest.raises(ValueError, match="feature set mismatch"):
        predictor.predict(model, pd.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="feature set"):
        predictor.check_feature_set(model, "8")


def test_compatibility_report_flags_shifted_data(trained):
    model = predictor.load_model(trained["artifact_dir"])
    same = predictor.compatibility_report(model, _toy_frame(seed=3).X)
    assert same["status"] == "ok"
    shifted = _toy_frame(seed=3).X.assign(a=lambda d: d["a"] + 50, c=np.nan)
    report = predictor.compatibility_report(model, shifted)
    assert report["status"] == "warning"
    assert "a" in report["features_out_of_range"]
    assert "c" in report["features_missing"]


def test_legacy_folder_without_manifest_uses_median(tmp_path):
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0] * 10, "f2": [1.0, 0.0] * 20})
    y = np.array([0, 1, 0, 1] * 10)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 2},
        xgb.DMatrix(X, label=y),
        num_boost_round=3,
    )
    model_dir = tmp_path / "netflow" / "0.9.0"
    model_dir.mkdir(parents=True)
    booster.save_model(str(model_dir / "model.json"))
    (model_dir / "class_labels.json").write_text(json.dumps(["BENIGN", "DoS"]))
    (model_dir / "feature_list.json").write_text(json.dumps(["f1", "f2"]))
    (model_dir / "train_stats.json").write_text(
        json.dumps({"medians": {"f1": 1.0, "f2": 0.0}})
    )
    model = predictor.load_model(model_dir)
    assert model.manifest.missing_values == "median"
    upload = pd.DataFrame({"f1": [np.nan], "f2": [1.0]})
    matrix = predictor.prepare_matrix(model, upload)
    assert matrix.iloc[0]["f1"] == 1.0


def test_binary_external_model(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    y = (X["x"] > 0).astype(int)
    booster = xgb.train(
        {"objective": "binary:logistic"}, xgb.DMatrix(X, label=y), num_boost_round=5
    )
    model_dir = tmp_path / "lane" / "1.0.0"
    model_dir.mkdir(parents=True)
    booster.save_model(str(model_dir / "model.json"))
    (model_dir / "class_labels.json").write_text(json.dumps(["BENIGN", "Malicious"]))
    (model_dir / "manifest.json").write_text(
        json.dumps({"lane_id": "lane", "version": "1.0.0", "train_stats_file": None,
                    "metrics_file": None})
    )
    model = predictor.load_model(model_dir)
    assert model.feature_list == ["x"]  # read from the booster
    out = predictor.predict(model, pd.DataFrame({"x": [-0.9, 0.9]}), shap_top_k=1)
    assert list(out["predicted_class"]) == ["BENIGN", "Malicious"]
    assert out["shap_top_features"].iloc[1][0]["contribution"] > 0
