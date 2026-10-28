from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.core.lane_spec import LaneSpec


@dataclass
class FeatureFrame:
    X: pd.DataFrame
    meta: pd.DataFrame
    y: Optional[pd.Series] = None


NUMERIC_FEATURES = [
    "src_port",
    "dst_port",
    "flow_duration_ms",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_packets",
    "total_bytes",
    "packets_per_second",
    "bytes_per_second",
    "failed_connections",
    "unique_destinations",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_rst_count",
]


def build_features(
    df: pd.DataFrame,
    spec: LaneSpec,
) -> FeatureFrame:
    """Create one numeric feature row and one label per network flow."""

    if df.empty:
        raise ValueError(
            "The validated dataset is empty. Check spec.yaml and validator.py."
        )

    df = df.reset_index(drop=True).copy()

    selected = [
        name for name in NUMERIC_FEATURES
        if name in df.columns
    ]

    if not selected:
        raise ValueError(
            f"No supported numeric features found. Columns: {list(df.columns)}"
        )

    X = df[selected].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns with no valid numeric values.
    X = X.dropna(axis=1, how="all")

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric features remain.")

    # Impute using training-dataset medians.
    for column in X.columns:
        median = X[column].median()
        X[column] = X[column].fillna(
            0.0 if pd.isna(median) else median
        )

    # Labels remain in their original form. The training module encodes them.
    label_column = spec.label_column
    y = None

    if label_column and label_column in df.columns:
        y = df[label_column].reset_index(drop=True)

        if y.isna().any():
            raise ValueError(
                f"Label column '{label_column}' contains missing values."
            )

        y = y.astype(str)

        if y.nunique() < 2:
            raise ValueError(
                "Training requires at least two distinct label classes."
            )

    source_column = next(
        (name for name in ("src_ip", "source_ip") if name in df.columns),
        None,
    )
    destination_column = next(
        (name for name in ("dst_ip", "dest_ip", "destination_ip")
         if name in df.columns),
        None,
    )

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )
    else:
        timestamps = pd.Series(pd.NaT, index=df.index)

    entity_field = (
        spec.entity.get("field", "unknown")
        if spec.entity
        else "unknown"
    )

    entity_values = (
        df[entity_field].astype(str)
        if entity_field in df.columns
        else pd.Series(["unknown"] * len(df))
    )

    source_values = (
        df[source_column].astype(str)
        if source_column
        else pd.Series(["unknown"] * len(df))
    )

    destination_values = (
        df[destination_column].astype(str)
        if destination_column
        else pd.Series(["unknown"] * len(df))
    )

    meta = pd.DataFrame({
        "entity_type": [entity_field] * len(df),
        "entity_value": entity_values.to_numpy(),
        "ip": source_values.to_numpy(),
        "target_value": destination_values.to_numpy(),
        "window_start": timestamps.to_numpy(),
        "window_end": timestamps.to_numpy(),
        "event_count": [1] * len(df),
    })

    return FeatureFrame(
        X=X.reset_index(drop=True),
        meta=meta.reset_index(drop=True),
        y=y.reset_index(drop=True) if y is not None else None,
    )