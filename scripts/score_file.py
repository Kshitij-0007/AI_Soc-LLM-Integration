#!/usr/bin/env python3
"""Score one log file with a lane model, without the API or database.

    python scripts/score_file.py netflow path/to/flows.csv --version 2.0.0

Runs exactly the upload pipeline (parser -> validator -> features ->
predictor -> taxonomy) and prints the class summary and the compatibility
report. If the file has labels, it also prints accuracy against them.
Predictions are written to outputs/scores/<file>_<lane>_<version>.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.core.discovery import (  # noqa: E402
    LANES_DIR,
    discover_lanes,
    load_build_features,
    load_parser,
)
from app.core.job_service import (  # noqa: E402
    _lane_feature_set,
    detect_file_format,
    missing_columns_message,
    score_frame,
)
from app.core.model_manifest import model_version_dir  # noqa: E402
from app.core.validator import validate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a log file with a lane model.")
    parser.add_argument("lane_id")
    parser.add_argument("data", type=Path)
    parser.add_argument("--version", required=True, help="models/<lane>/<version>")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    lanes = discover_lanes()
    if args.lane_id not in lanes:
        raise SystemExit(f"Unknown lane '{args.lane_id}'. Lanes: {sorted(lanes)}")
    spec = lanes[args.lane_id]
    lane_dir = LANES_DIR / args.lane_id

    parse = load_parser(args.lane_id, lane_dir)
    df = parse(args.data, detect_file_format(args.data, spec))
    print(f"Detected source format: {df.attrs.get('source_format', 'n/a')}")
    labels = None
    if args.lane_id == "netflow":
        from app.lanes.netflow.train import canonical_labels

        labels = canonical_labels(df)
        df = df.drop(columns=["label"], errors="ignore")

    validated = validate(df, spec)
    if validated.status == "rejected":
        print("REJECTED:", missing_columns_message(validated, spec))
        return 1

    ff = load_build_features(args.lane_id, lane_dir)(validated.clean_df, spec)
    model_dir = model_version_dir(PROJECT_DIR / "models", args.lane_id, args.version)
    predictions, report = score_frame(
        ff,
        args.lane_id,
        model_dir,
        lane_feature_set=_lane_feature_set(args.lane_id),
    )

    print(f"\nRows scored: {len(predictions):,}")
    print(predictions["predicted_class"].value_counts().to_string())
    print("\nCompatibility report:")
    print(json.dumps(report, indent=2))

    if labels is not None and labels.notna().any():
        from app.core.training import classification_metrics

        known = labels.reset_index(drop=True)
        mask = known.notna() & known.isin(
            list(predictions["class_probabilities"].iloc[0].keys())
        )
        classes = list(predictions["class_probabilities"].iloc[0].keys())
        index = {c: i for i, c in enumerate(classes)}
        y_true = known[mask].map(index).to_numpy()
        y_pred = predictions["predicted_class"][mask.to_numpy()].map(index).to_numpy()
        metrics = classification_metrics(y_true, y_pred, classes)
        print(
            f"\nAgainst file labels ({int(mask.sum()):,} rows): accuracy "
            f"{metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}"
        )
        if metrics.get("attack_vs_benign"):
            print(f"Attack vs benign: {metrics['attack_vs_benign']}")
        for label, scores in metrics["per_class"].items():
            if scores["support"]:
                print(
                    f"  {label:12s} recall {scores['recall']:.3f}  "
                    f"precision {scores['precision']:.3f}  n={scores['support']}"
                )

    output = args.output or (
        PROJECT_DIR
        / "outputs"
        / "scores"
        / f"{args.data.stem}_{args.lane_id}_{args.version}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out = ff.meta[["entity_value", "target_value", "window_start"]].join(
        predictions[["predicted_class", "confidence", "unified_class", "tactic"]]
    )
    if labels is not None:
        out["actual_class"] = labels.reset_index(drop=True)
    out.to_csv(output, index=False)
    print(f"\nPredictions written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
