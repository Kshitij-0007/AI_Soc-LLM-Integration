#!/usr/bin/env python3
"""Measure how the NetFlow model behaves on data it was not trained on.

Modes
-----
score
    Score labelled files with an existing model folder. Use files from a
    different dataset / network than the training data: this is the number
    that predicts behaviour on new CSVs.

        python scripts/evaluate_netflow.py score --version 2.0.0 --data <files/dirs>

leave-one-file-out
    For every file: train on all other files, test on that file. Attack
    classes absent from training are reported as "detected as some attack"
    (any non-BENIGN prediction), which shows whether the model flags attack
    types it never saw. Benign false-positive rate per unseen capture is the
    key number for a SOC.

        python scripts/evaluate_netflow.py leave-one-file-out --data <dir>

Results go to outputs/evaluation/<mode>_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.core import predictor  # noqa: E402
from app.core.contracts import FeatureFrame  # noqa: E402
from app.core.training import BENIGN, classification_metrics, train_and_save  # noqa: E402
from app.lanes.netflow.features import FEATURE_SET_VERSION  # noqa: E402
from app.lanes.netflow.train import (  # noqa: E402
    LabeledSet,
    expand_paths,
    load_labeled_features,
    load_spec,
)

OUT_DIR = PROJECT_DIR / "outputs" / "evaluation"


def evaluate_predictions(
    actual: pd.Series, predicted: np.ndarray, class_labels: list[str]
) -> dict:
    """Metrics on known classes plus detection of classes the model lacks."""
    actual = actual.reset_index(drop=True)
    known = actual.isin(class_labels).to_numpy()
    index = {c: i for i, c in enumerate(class_labels)}
    result: dict = {"rows": int(len(actual))}
    if known.any():
        result["known_classes"] = classification_metrics(
            actual[known].map(index).to_numpy(),
            pd.Series(predicted[known]).map(index).to_numpy(),
            class_labels,
        )
    unseen = ~known & (actual != BENIGN).to_numpy()
    if unseen.any():
        result["unseen_attack_classes"] = {
            label: {
                "rows": int(mask.sum()),
                "detected_as_attack": float((predicted[mask] != BENIGN).mean()),
            }
            for label in sorted(actual[unseen].unique())
            for mask in [(actual == label).to_numpy()]
        }
    benign = (actual == BENIGN).to_numpy()
    if benign.any():
        result["benign_false_positive_rate"] = float(
            (predicted[benign] != BENIGN).mean()
        )
    attacks = (actual != BENIGN).to_numpy()
    if attacks.any():
        result["attack_detection_rate"] = float((predicted[attacks] != BENIGN).mean())
    return result


def _print_summary(name: str, result: dict) -> None:
    parts = [f"{name}: {result['rows']:,} rows"]
    if "attack_detection_rate" in result:
        parts.append(f"attack detection {result['attack_detection_rate']:.3f}")
    if "benign_false_positive_rate" in result:
        parts.append(f"benign FPR {result['benign_false_positive_rate']:.3f}")
    known = result.get("known_classes")
    if known:
        parts.append(f"macro-F1(known) {known['macro_f1']:.3f}")
    print("  " + " | ".join(parts))
    for label, info in result.get("unseen_attack_classes", {}).items():
        print(
            f"      unseen class {label}: {info['detected_as_attack']:.3f} "
            f"flagged as attack (n={info['rows']})"
        )
    if known:
        for label, scores in known["per_class"].items():
            if scores["support"]:
                print(
                    f"      {label:12s} recall {scores['recall']:.3f} "
                    f"precision {scores['precision']:.3f} n={scores['support']}"
                )


def mode_score(args: argparse.Namespace) -> dict:
    model = predictor.load_model(PROJECT_DIR / "models" / "netflow" / args.version)
    predictor.check_feature_set(model, FEATURE_SET_VERSION)
    spec = load_spec()
    report: dict = {"model": f"netflow/{args.version}", "files": {}}
    for path in expand_paths(args.data):
        data = load_labeled_features([path], spec, args.max_per_class, verbose=False)
        predictions = predictor.predict(model, data.X, shap_top_k=0)
        result = evaluate_predictions(
            data.y, predictions["predicted_class"].to_numpy(), model.class_labels
        )
        result["compatibility"] = predictor.compatibility_report(
            model, data.X, predictions
        )
        report["files"][path.name] = result
        _print_summary(path.name, result)
        print(f"      compatibility: {result['compatibility']['status']} "
              f"{result['compatibility']['reasons']}")
    return report


def mode_leave_one_file_out(args: argparse.Namespace) -> dict:
    spec = load_spec()
    files = expand_paths(args.data)
    print(f"Featurising {len(files)} files...")
    per_file: dict[str, LabeledSet] = {
        path.name: load_labeled_features([path], spec, args.max_per_class)
        for path in files
    }
    report: dict = {"files": {}}
    for held_out, test in per_file.items():
        train_parts = [d for name, d in per_file.items() if name != held_out]
        X = pd.concat([d.X for d in train_parts], ignore_index=True)
        y = pd.concat([d.y for d in train_parts], ignore_index=True)
        print(f"\nHeld out {held_out}; training on {len(X):,} rows...")
        with tempfile.TemporaryDirectory() as tmp:
            result = train_and_save(
                FeatureFrame(X=X, meta=pd.DataFrame(index=X.index), y=y),
                lane_id="netflow",
                version="lofo",
                models_dir=Path(tmp),
                data_source="leave-one-file-out",
                feature_set_version=FEATURE_SET_VERSION,
                n_jobs=os.cpu_count(),
                cv_folds=0,
            )
            model = predictor.load_model(result["artifact_dir"])
            predictions = predictor.predict(model, test.X, shap_top_k=0)
            predictor.clear_cache()
        evaluation = evaluate_predictions(
            test.y, predictions["predicted_class"].to_numpy(), model.class_labels
        )
        evaluation["trained_classes"] = model.class_labels
        report["files"][held_out] = evaluation
        _print_summary(held_out, evaluation)

    fprs = [
        r["benign_false_positive_rate"]
        for r in report["files"].values()
        if "benign_false_positive_rate" in r
    ]
    report["mean_benign_false_positive_rate"] = float(np.mean(fprs)) if fprs else None
    mean_fpr = report["mean_benign_false_positive_rate"]
    print(f"\nMean benign FPR on unseen captures: {mean_fpr}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    score = sub.add_parser("score", help="Score labelled files with a saved model.")
    score.add_argument("--version", required=True)
    score.add_argument("--data", type=Path, nargs="+", required=True)
    score.add_argument("--max-per-class", type=int, default=0)

    lofo = sub.add_parser("leave-one-file-out", help="Train on N-1 files, test on 1.")
    lofo.add_argument("--data", type=Path, nargs="+", required=True)
    lofo.add_argument("--max-per-class", type=int, default=20_000)

    args = parser.parse_args()
    args.max_per_class = args.max_per_class or None
    report = mode_score(args) if args.mode == "score" else mode_leave_one_file_out(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{args.mode}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
