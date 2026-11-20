"""Per-row feature contributions (TreeSHAP) for XGBoost predictions.

Uses XGBoost's built-in ``pred_contribs`` (exact TreeSHAP for tree models),
so no extra dependency is needed and values match the shap library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def top_contributions(
    booster: xgb.Booster,
    X: pd.DataFrame,
    class_index: np.ndarray,
    top_k: int = 5,
) -> list[list[dict]]:
    """Top ``top_k`` features pushing each row towards its predicted class.

    Returns one list per row: ``[{"feature", "value", "contribution"}, ...]``
    sorted by absolute contribution.
    """
    if X.empty:
        return []
    contributions = booster.predict(
        xgb.DMatrix(X, feature_names=list(X.columns)),
        pred_contribs=True,
    )
    # Multi-class: (rows, classes, features + bias). Binary: (rows, features + bias).
    if contributions.ndim == 3:
        rows = np.arange(len(X))
        contributions = contributions[rows, class_index, :]
    else:
        # Binary contributions push towards class 1; flip them for class 0.
        sign = np.where(np.asarray(class_index) == 0, -1.0, 1.0)
        contributions = contributions * sign[:, None]
    contributions = contributions[:, :-1]  # drop the bias column

    names = list(X.columns)
    values = X.to_numpy(dtype=float)
    result = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(contributions[i]))[:top_k]
        result.append(
            [
                {
                    "feature": names[j],
                    "value": None if np.isnan(values[i, j]) else float(values[i, j]),
                    "contribution": float(contributions[i, j]),
                }
                for j in order
            ]
        )
    return result
