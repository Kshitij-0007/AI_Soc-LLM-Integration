"""Score an ordered feed of completed canonical NetFlow records from JSON lines."""

import argparse
import ipaddress
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from app.core import predictor
from app.core.validator import validate
from app.lanes.netflow.adapters import to_canonical
from app.lanes.netflow.features import (
    CONTEXT_WINDOW,
    FEATURE_SET_VERSION,
    build_features,
)
from app.lanes.netflow.train import load_spec


def closed_windows(records: Iterable[dict], max_rows: int) -> Iterator[list[dict]]:
    """A later minute closes the previous minute. EOF closes the final minute."""
    window = None
    pending = []
    for number, record in enumerate(records, 1):
        try:
            stamp = pd.to_datetime(record["timestamp"], utc=True, errors="raise")
            if not isinstance(stamp, pd.Timestamp) or pd.isna(stamp):
                raise ValueError("missing timestamp")
            minute = stamp.floor(CONTEXT_WINDOW)
            for name in ("src_ip", "dst_ip"):
                ipaddress.ip_address(str(record[name]))
            port = float(record["dst_port"])
            duration = float(record["flow_duration_ms"])
            if not port.is_integer() or not 0 <= port <= 65535:
                raise ValueError("invalid destination port")
            if not 0 <= duration < float("inf"):
                raise ValueError("invalid duration")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid canonical record at position {number}") from exc
        if window is not None and minute < window:
            raise ValueError(f"Record {number} belongs to an already closed minute")
        if pending and minute != window:
            yield pending
            pending = []
        if len(pending) >= max_rows:
            raise ValueError(f"Minute exceeds the configured {max_rows} flow limit")
        pending.append({**record, "timestamp": stamp})
        window = minute
    if pending:
        yield pending


def score_feed(records: Iterable[dict], model_path: Path, max_rows: int):
    spec = load_spec()
    model = predictor.load_model(model_path)
    predictor.check_feature_set(model, FEATURE_SET_VERSION)
    for window in closed_windows(records, max_rows):
        frame, _ = to_canonical(pd.DataFrame(window))
        checked = validate(frame, spec)
        if checked.status == "rejected":
            raise ValueError(f"Missing required columns: {checked.missing_required}")
        features = build_features(checked.clean_df, spec)
        scored = predictor.predict(model, features.X, shap_top_k=0)
        for flow, result in zip(window, scored.to_dict("records"), strict=True):
            yield {
                "timestamp": flow["timestamp"].isoformat(),
                "src_ip": flow["src_ip"],
                "dst_ip": flow["dst_ip"],
                "model_version": model.manifest.version,
                "predicted_class": result["predicted_class"],
                "class_probabilities": result["class_probabilities"],
                "confidence": result["confidence"],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--max-window-rows", type=int, default=100000)
    args = parser.parse_args()
    records = (json.loads(line) for line in sys.stdin if line.strip())
    for result in score_feed(records, args.model_dir, args.max_window_rows):
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
