from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock

from app.config import settings
from app.core.resources import xgb_threads


class LaneExecutor:
    def __init__(self, lane_id: str, workers: int | None = None):
        self.lane_id = lane_id
        pools = {"thread": ThreadPoolExecutor, "process": ProcessPoolExecutor}
        self._executor = pools[settings.executor_type](
            max_workers=workers or settings.workers_for(lane_id)
        )
        self._pending: set[Future] = set()
        self._lock = Lock()

    def submit(self, job_id: str, fn, *args, **kwargs) -> None:
        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._finished)

    def _finished(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def done(self) -> bool:
        with self._lock:
            return not self._pending


def n_jobs(workers_per_lane: int | None = None, num_lanes: int | None = None) -> int:
    return xgb_threads(num_lanes, workers_per_lane)
