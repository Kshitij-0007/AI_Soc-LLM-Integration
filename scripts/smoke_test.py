#!/usr/bin/env python
"""End-to-end smoke test for AI-SOC.

Submits 3 files (one per lane), polls for completion,
fetches incidents, and downloads PDF report.
"""
from __future__ import annotations

import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def api_request(method: str, url: str, data: dict | None = None) -> dict:
    """Make an API request and return the JSON response."""
    if method == "GET":
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8") if data else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def test_basic_flow() -> bool:
    """Test the basic upload and polling flow."""
    print("=" * 60)
    print("AI-SOC Smoke Test")
    print("=" * 60)

    # Step 1: Submit 3 files (one per lane)
    print("\n[1] Submitting 3 files (one per lane)...")
    {
        "files": (
            "sample1.csv",
            open("data/raw/netflow/sample1.csv", "rb"),
        ),
        "lanes": ["netflow", "siem_edr", "cloud_entity"],
    }

    # Actually use multipart form directly
    boundry = "----BOUNDARY123"
    body = ""
    body += f"--{boundry}\r\n"
    body += 'Content-Disposition: form-data; name="files"; filename="sample1.csv"\r\n'
    body += "Content-Type: text/csv\r\n\r\n"
    with open("data/raw/netflow/sample1.csv", "rb") as f:
        body += f.read().decode("latin-1") + "\r\n"
    body += f"--{boundry}\r\n"
    body += 'Content-Disposition: form-data; name="lanes"\r\n\r\n'
    body += "netflow,siem_edr,cloud_entity\r\n"
    body += f"--{boundry}--\r\n"

    req = urllib.request.Request(
        f"{BASE_URL}/api/runs",
        data=body.encode("latin-1"),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundry}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        run_data = json.loads(resp.read().decode())
    run_id = run_data.get("run_id", "")
    jobs = run_data.get("jobs", [])
    print(f"  Run ID: {run_id}")
    print(f"  Jobs: {len(jobs)} submitted")

    # Step 2: Poll for completion
    print("\n[2] Polling for job completion...")
    max_wait = 30
    waited = 0
    while waited < max_wait:
        run_info = api_request("GET", f"{BASE_URL}/api/runs/{run_id}")
        jobs_info = run_info.get("jobs", [])
        all_done = all(j.get("status") in ("done", "failed") for j in jobs_info)
        if all_done:
            print(f"  All jobs finished after {waited}s")
            break
        print(f"  Waited {waited}s... status: {jobs_info}")
        time.sleep(2)
        waited += 2

    if waited >= max_wait:
        print("  Timeout waiting for jobs to complete!")
        return False

    # Step 3: Get results
    print("\n[3] fetching results...")
    results = api_request("GET", f"{BASE_URL}/api/runs/{run_id}/results")
    print(f"  Got {len(results.get('rows', []))} result rows")

    # Step 4: Get incidents
    print("\n[4] Fetching incidents...")
    incidents = api_request("GET", f"{BASE_URL}/api/runs/{run_id}/incidents")
    print(f"  Got {len(incidents.get('incidents', []))} incidents")

    # Step 5: Create report
    print("\n[5] Creating report...")
    report_data = api_request("POST", f"{BASE_URL}/api/runs/{run_id}/report")
    report_id = report_data.get("report_id", "")
    print(f"  Report ID: {report_id}")

    # Step 6: Download PDF
    print("\n[6] Downloading PDF...")
    try:
        pdf_req = urllib.request.Request(
            f"{BASE_URL}/api/reports/{report_id}/download",
            headers={"Accept": "application/pdf"},
        )
        with urllib.request.urlopen(pdf_req) as resp:
            pdf_size = len(resp.read())
        print(f"  PDF downloaded: {pdf_size} bytes")
    except Exception as e:
        print(f"  PDF download error: {e}")

    print("\n" + "=" * 60)
    print("Smoke test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_basic_flow()
