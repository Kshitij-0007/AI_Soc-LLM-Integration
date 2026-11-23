# AI-SOC local NetFlow setup

The local API accepts NetFlow CSV and JSONL uploads and writes jobs, events and predictions to PostgreSQL. Run these commands from the project root with Python 3.11 or later.

1. Install the dependencies with `python -m pip install -r requirements.txt`.
2. On a new checkout, copy `.env.example` to `.env` and enter the connection URL for your PostgreSQL database. Set `ENABLED_LANES=["netflow"]` for NetFlow only.
3. Run `python scripts/init_db.py` to create the tables and apply pending migrations in the configured database.
4. Use the single trained NetFlow model. On a fresh checkout, first follow the [NetFlow training instructions](app/lanes/netflow/README.md#reproduce-the-fit) to create the git-ignored artifacts. Register the model and start the API.

```powershell
python scripts/register_models.py netflow public-10k-v1 --activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

With the server running, execute `python scripts/smoke_test.py` in another terminal. It uploads 300 bundled flows, polls the run, checks the result count and prints the run ID. Each invocation creates one run and retains the upload and its predictions.

Open `http://127.0.0.1:8000/docs` for the upload form. The multipart fields are `files[]` and `lanes[]`, with one lane per file in matching order. `GET /api/health` checks database connectivity and `GET /api/models` lists the active model. LLM and PDF reporting are outside this local setup.

## Uploaded datasets

Each API run stores original files in `data/uploads/<run_id>/raw/` and creates
`train/` and `test/` directories for later preparation. Scoring accepts unlabeled files.

Prepare a labeled binary demo dataset with the existing separation script:

```powershell
python -m app.lanes.netflow.prepare_data --data data/raw/netflow/netflow.csv --output-dir data/datasets/netflow --sections
```

Each invocation creates `data/datasets/netflow/<dataset_id>/raw/`, `train/train.csv`
and `test/test.csv`. The raw copy preserves the input bytes. The split is the
existing stratified 80/20 split for the binary demo. Real capture evaluation
uses the temporal or separate-capture workflow in the NetFlow lane README.

## Live NetFlow input and acceptance criteria

An exporter or collector can pipe canonical JSON lines into the local scorer:

```powershell
Get-Content capture.jsonl | python -m app.lanes.netflow.live --model-dir models/netflow_model/public-10k-v1
```

Each line represents a completed flow. Supply an ISO timestamp (UTC recommended),
source and destination IP, destination port and duration in milliseconds, plus
the directional counters and flags available from the exporter. Feed one sensor
per process in event-minute order. Predictions are emitted on standard output
when the next minute arrives or the input closes. An idle open pipe retains its
last window. A late record from a closed minute, an invalid required value or an
oversized window stops the scorer. The caller is responsible for buffering,
replay and storing outputs. This entry point scores locally without PostgreSQL.

The proposed target is at least 90% binary accuracy on each independent labeled
capture, with 95% as the goal. Record attack recall, benign false positives,
per-family support, and input rejection rates alongside accuracy. Reserve
untouched captures for the final check. A deployment decision also requires
collector-specific tests for delayed flow exports, idle periods, restart and
replay, sustained peak flows per second, and end-to-end alert delay.
