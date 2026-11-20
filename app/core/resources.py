"""CPU budgeting shared by training and scoring."""

from __future__ import annotations

import os


def xgb_threads(
    num_lanes: int | None = None, workers_per_lane: int | None = None
) -> int:
    """Threads per XGBoost call so concurrent lane jobs do not oversubscribe.

    ``n_jobs = max(1, total_cores // (num_lanes * workers_per_lane))``.
    Lane count comes from discovery and workers from settings.yaml, so adding
    a lane folder needs no change here.
    """
    if num_lanes is None:
        from app.core.discovery import discover_lanes

        num_lanes = max(1, len(discover_lanes()))
    if workers_per_lane is None:
        from app.config import settings

        workers_per_lane = settings.default_workers_per_lane
    return max(1, (os.cpu_count() or 1) // (num_lanes * max(1, workers_per_lane)))
