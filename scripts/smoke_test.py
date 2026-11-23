"""Submit bundled NetFlow traffic to a running local API and verify stored results."""

import argparse
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def verify(base_url: str):
    path = ROOT / "data/raw/netflow/netflow.csv"
    with httpx.Client(base_url=base_url, timeout=15, trust_env=False) as client:
        health = client.get("/api/health").raise_for_status().json()
        models = client.get("/api/models").raise_for_status().json()
        assert health["db"] == "ok" and "netflow" in models
        with path.open("rb") as stream:
            run = client.post(
                "/api/runs",
                data={"lanes[]": "netflow"},
                files={"files[]": (path.name, stream, "text/csv")},
            )
        run_id = run.raise_for_status().json()["run_id"]
        route = f"/api/runs/{run_id}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            status = client.get(route).raise_for_status().json()
            if status["status"] in {"scored", "failed"}:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Timed out waiting for run {run_id}")
        job = status["jobs"][0]
        if status["status"] != "scored" or job["status"] != "done":
            raise RuntimeError(f"Run {run_id} failed: {job['error']}")
        rows = (
            client.get(route + "/results", params={"limit": 10000})
            .raise_for_status()
            .json()["rows"]
        )
        assert len(rows) == job["rows_valid"] > 0
        assert all(row["predicted_class"] in row["class_probabilities"] for row in rows)
        incidents = (
            client.get(route + "/incidents").raise_for_status().json()["incidents"]
        )
        print(
            f"Run {run_id}: {len(rows)} predictions, {len(incidents)} incidents, "
            f"model {models['netflow']['version']}"
        )
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    verify(parser.parse_args().url)
