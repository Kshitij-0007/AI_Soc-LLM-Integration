"""Fetch reproducible, label-independent CSV blocks from public flow collections."""

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "data/raw/netflow/real/cic-transfer-v1"
SOURCES = {
    "iot-train": (
        "NF-ToN-IoT-v2",
        "4e951f835487ee57ee5da5dae2eaeb704908ab38",
        ["NF-ToN-IoT-v2-train.csv"],
    ),
    "iot-test": (
        "NF-ToN-IoT-v2",
        "4e951f835487ee57ee5da5dae2eaeb704908ab38",
        ["NF-ToN-IoT-v2-test.csv"],
    ),
    "train": (
        "CIC-IDS2017",
        "4e9edfa727f800b6f595bc2ee7420527156562b2",
        [
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        ],
    ),
    "test": (
        "CIC-IDS2018",
        "236baa10495af451de38485123a90f60e7632bbd",
        [
            "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
        ],
    ),
    "train-2018": (
        "CIC-IDS2018",
        "236baa10495af451de38485123a90f60e7632bbd",
        [
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        ],
    ),
    "final": (
        "CIC-IDS2018",
        "236baa10495af451de38485123a90f60e7632bbd",
        ["Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"],
    ),
}


def read_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    for delay in (1, 2):
        try:
            return _request_range(url, start, end)
        except OSError:
            time.sleep(delay)
    return _request_range(url, start, end)


def _request_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    request = Request(
        f"{url}?offset={start}&end={end}", headers={"Range": f"bytes={start}-{end}"}
    )
    with urlopen(request, timeout=60) as response:
        span = re.fullmatch(
            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
        )
        if (
            response.status != 206
            or not span
            or int(span[1]) != start
            or int(span[2]) != end
        ):
            raise ValueError("Source returned an unexpected byte range")
        payload = response.read(end - start + 1)
        if len(payload) != end - start + 1:
            raise IOError("Incomplete CSV block")
        return payload, int(span[3])


def fetch(task: tuple[Path, str, str, str, str, int]) -> dict:
    root, role, dataset, revision, name, block_bytes = task
    owner = "Nora9029" if dataset == "NF-ToN-IoT-v2" else "c01dsnap"
    url = f"https://huggingface.co/datasets/{owner}/{dataset}/resolve/{revision}/{name}"
    header, size = read_range(url, 0, 4095)
    lines = [header.splitlines()[0]]
    spans = []
    for percent in range(5, 100, 10):
        start = size * percent // 100
        end = min(size - 1, start + min(block_bytes, size // 10) - 1)
        chunk, _ = read_range(url, start, end)
        lines.extend(chunk.splitlines()[1:-1])
        spans.append([start, end])
    payload = b"\n".join(lines) + b"\n"
    path = root / role / name
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "file": name,
        "role": role,
        "url": url,
        "ranges": spans,
        "rows": len(lines) - 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument(
        "--training-block-bytes", type=int, default=262144,
        help="Bytes per sampled training block. Evaluation blocks remain 256 KiB.",
    )
    parser.add_argument(
        "--roles", nargs="+", choices=SOURCES,
        default=["train", "test", "train-2018", "final"],
    )
    args = parser.parse_args()
    if args.training_block_bytes < 4096:
        parser.error("--training-block-bytes must be at least 4096")
    root = args.output
    tasks = [
        (
            root, role, dataset, revision, name,
            args.training_block_bytes if "train" in role else 262144,
        )
        for role, (dataset, revision, files) in SOURCES.items()
        if role in args.roles
        for name in files
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, tasks))
    manifest = root / "sources.json"
    previous = json.loads(manifest.read_text("utf-8")) if manifest.exists() else []
    entries = [row for row in previous if row["role"] not in args.roles] + results
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(
        f"Fetched {sum(r['rows'] for r in results)} flow records in {len(tasks)} files"
    )


if __name__ == "__main__":
    main()
