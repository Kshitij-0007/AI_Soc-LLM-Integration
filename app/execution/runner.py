from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor

from app.config import settings


class LaneExecutor:
    """Per-lane executor using ThreadPoolExecutor (Windows safe).

    Creates a thread pool per lane so files for different lanes
    run in parallel. ProcessPoolExecutor is available if settings
    enable it, but job functions must be top-level with __main__ guard.
    """

    def __init__(self, lane_id: str, workers: int | None = None):
        self.lane_id = lane_id
        workers = workers or settings.workers_for(lane_id)
        self._executor: Executor = ThreadPoolExecutor(max_workers=workers)
        self._submitted: dict[str, int] = {}  # job_id -> future

    def submit(self, job_id: int, fn, *args, **kwargs) -> None:
        """Submit a job to the executor."""
        future = self._executor.submit(fn, *args, **kwargs)
        self._submitted[job_id] = future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)

    def done(self) -> bool:
        """Check if all submitted jobs are done."""
        return all(f.done() for f in self._submitted.values())


def n_jobs(workers_per_lane: int | None = None, num_lanes: int | None = None) -> int:
    """XGBoost threads per job: cores // (lanes * workers per lane).

    Lane count comes from discovery, so new lane folders need no change here.
    """
    from app.core.resources import xgb_threads

    return xgb_threads(num_lanes, workers_per_lane)
