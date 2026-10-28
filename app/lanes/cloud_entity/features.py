from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.core.contracts import FeatureFrame


@dataclass
class FeatureFrame:
    X: pd.DataFrame  # numeric only, one row per unit of analysis
    meta: pd.DataFrame  # same index: entity_type, entity_value, ip, target_value,
        # window_start, window_end, event_count
    y: Optional[pd.Series] = None  # native labels, only when label column exists


def build_features(df: pd.DataFrame, spec: LaneSpec) -> FeatureFrame:
    """Build feature matrix and metadata from raw DataFrame.

    Same function for training and scoring. No label-dependent logic.
    """
    # Select numeric features, drop identifiers and timestamp
    drop_cols = ["timestamp", "principal_name", "user_name"]
    features = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric columns
    numeric_df = df[features].apply(pd.to_numeric, errors="coerce")

    # Impute missing values with median
    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    # Remove any remaining NaN rows
    numeric_df = numeric_df.dropna(how="all")

    # Build metadata
    entity_field = spec.entity.get("field", "unknown")
    entity_value = (
        df[entity_field].mode()[0] if entity_field in df.columns else "unknown"
    )

    ip_column = spec.ip_field or "unknown"
    ip_value = df[ip_column].mode()[0] if ip_column in df.columns else "unknown"

    # Window info - for principal_window_15m
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        window_start = ts.min()
        window_end = ts.max()
    else:
        window_start = None
        window_end = None

    meta_data = pd.DataFrame(
        {
            "entity_type": [entity_field],
            "entity_value": [entity_value],
            "ip": [ip_value],
            "target_value": ["unknown"],
            "window_start": [window_start],
            "window_end": [window_end],
            "event_count": [len(df)],
        }
    )

    # Labels if available
    y: Optional[pd.Series] = None
    if spec.label_column and spec.label_column in df.columns:
        y = df[spec.label_column]
        try:
            y = pd.factorize(y)[0]
        except Exception:
            y = None

    X = numeric_df.reset_index(drop=True)
    return FeatureFrame(X=X, meta=meta_data, y=y)
