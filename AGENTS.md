# AI-SOC: Project Reset and Build Specification

> **Audience:** an AI coding agent (OpenCode) working inside `C:\ai_soc`.
> **Status of this file:** the single source of truth. It supersedes every earlier design document, any old code, and the old `ai_soc_schema.sql`.
> **Read this whole file before writing any code. Follow the build order in section 12.**

---

## 0. What is happening

The previous contents of `C:\ai_soc` have been deleted. We are rebuilding from scratch with a new workflow.

Already done by the human (do not redo):
- A PostgreSQL database exists and is reachable through `DATABASE_URL` in `.env`.
- The folder `C:\ai_soc` is empty except for this file and, optionally, `schema/ai_soc_schema_v2.sql`.

Your job: scaffold the folder structure in section 3, then implement the platform in the order in section 12.

---

## 1. Product summary

AI-SOC is a multi-model security analytics platform.

- **Three lanes**, one XGBoost model per lane: `netflow`, `siem_edr`, `cloud_entity`.
- The user uploads one or more files in the web UI. **Each file has a dropdown to choose its log type.** The file goes straight to that lane's model. There is no auto-detection and no shared queue.
- Each lane has its **own worker pool**, so files for different lanes run at the same time.
- `POST /api/runs` returns a `run_id` immediately. The UI polls for status.
- Every lane emits the **same prediction contract** and maps its native labels to a shared taxonomy (MITRE tactics).
- When all jobs of a run finish, a **run finalizer** builds incidents (group by entity + tactic, cross-lane corroboration), then an **LLM** writes problems and remedies as validated JSON, then a **PDF** is rendered from the stored JSON.
- Everything is stored in PostgreSQL (12 tables, schema v2).

### Non-negotiable decisions

| Topic | Decision |
|---|---|
| Models | XGBoost, `multi:softprob`, `tree_method=hist`, one per lane, no multi-input model |
| Routing | Dropdown selection per file. Wrong selection is caught by validation (missing required columns). |
| Parallelism | One executor per lane, created at startup. No Redis, no Celery. |
| Executor type | **Default `ThreadPoolExecutor`** (Windows safe). `ProcessPoolExecutor` only if a setting enables it. |
| Validation | Generic, in `app/core/validator.py`, driven by each lane's `spec.yaml` |
| DB access | SQLAlchemy 2 + psycopg 3. Schema applied from the SQL file. **No Alembic.** |
| LLM | Any OpenAI-compatible endpoint via `httpx`. Endpoint and model from `.env`. Never decides severity or confidence. |
| Training | Reads files directly. Only scored uploads write to `events` and `predictions`. |
| Calibration, day-held-out eval, analyst feedback | Stretch. Do not block anything on them. |

---

## 2. Database (PostgreSQL) rules

The database already exists. The authoritative DDL is `schema/ai_soc_schema_v2.sql` (12 tables):

`lanes`, `taxonomy_map`, `runs`, `ingestion_jobs`, `raw_records`, `rejected_records`, `events`, `model_versions`, `predictions`, `incidents`, `incident_evidence`, `reports`.

### Critical warning about old tables

The DDL uses `CREATE TABLE IF NOT EXISTS`. If the database still contains tables from the old schema (`log_sources`, `ingestion_batches`, old `events`, `predictions`, `reports`, `taxonomy_map`, `incidents`), the new DDL will **silently skip** them and leave the wrong columns.

Therefore:

1. Create `scripts/reset_db.py`. It drops every table in the `public` schema (including old ones) and re-applies the v2 SQL. It must **refuse to run** unless called with `--yes` and must print the target database name first.
2. Create `scripts/init_db.py`. It applies `schema/ai_soc_schema_v2.sql` (idempotent) and then verifies that all 12 tables exist and exit non-zero if any is missing.
3. Tell the human to run `python scripts/reset_db.py --yes` once, because the DB may contain old tables.

### Rules for DB code

- All SQL access goes through `app/db/repositories/*`. No SQL in API routes, lanes, or the orchestrator.
- Bulk writes (`events`, `predictions`, `rejected_records`, `raw_records`) use `psycopg` `COPY` or `execute_values`-style batching. **Never insert row by row.**
- `app/db/models.py` holds SQLAlchemy ORM models that mirror the SQL file exactly. If the SQL and the ORM disagree, the SQL wins.
- One partial unique index guarantees one active model per lane (`uq_one_active_model_per_lane`). Activate a model by deactivating the old one in the same transaction.
- Optional additive changes go in `migrations/NNN_description.sql` as plain SQL, applied by `init_db.py` in filename order and recorded in a small `schema_migrations(filename text primary key, applied_at timestamptz)` table. Do not edit the v2 file for them. Suggested optional migration `001_incident_extras.sql`: add `entity_type TEXT`, `unified_classes TEXT[]`, `status TEXT DEFAULT 'open'` to `incidents`, and `calibrated_confidence REAL` to `predictions`. Only create it if the code needs these.

---

## 3. Target folder structure

Create exactly this tree under `C:\ai_soc`. Files marked `(stub)` get a minimal importable skeleton with docstring and TODO. Files marked `(impl)` must be fully implemented in the phase noted.

```
C:\ai_soc\
├── AGENTS.md                         # this file
├── README.md                         # human setup + demo guide (Phase 8)
├── .env.example                      # template, see section 4
├── .env                              # real values, git-ignored (human creates)
├── .gitignore
├── .gitattributes
├── pyproject.toml                    # ruff + pytest config, project metadata
├── requirements.txt                  # runtime deps
├── requirements-dev.txt              # pytest, ruff, httpx test client
├── docker-compose.yml                # optional: postgres 16 service (human already has a DB)
│
├── configs/
│   ├── settings.yaml                 # executors, workers, limits, paths
│   ├── orchestrator.yaml             # windows, boosts, thresholds, severity bands
│   └── llm.yaml                      # provider, timeouts, retry, prompt_version
│
├── schema/
│   └── ai_soc_schema_v2.sql          # authoritative DDL (12 tables)
├── migrations/                       # optional additive .sql files only
│   └── .gitkeep
│
├── data/                             # git-ignored except samples/.gitkeep
│   ├── raw/
│   │   ├── netflow/                  # CIC-IDS2017 CSVs
│   │   ├── siem_edr/
│   │   └── cloud_entity/
│   ├── processed/
│   ├── uploads/                      # runtime: uploaded files, <run_id>/<job_id>_<filename>
│   └── samples/                      # demo files, can be committed if small
│
├── models/                           # git-ignored artifacts
│   ├── netflow/1.0.0/                # model.json, feature_list.json, class_labels.json,
│   ├── siem_edr/1.0.0/               #   train_stats.json, metrics.json
│   └── cloud_entity/1.0.0/
│
├── knowledge/
│   └── attack_snippets.yaml          # per-class remedies used by LLM prompt + fallback
│
├── outputs/
│   ├── reports/                      # generated PDFs: <report_id>.pdf
│   └── charts/                       # temporary matplotlib PNGs
│
├── app/
│   ├── __init__.py
│   ├── main.py                       # (impl) FastAPI app, lifespan: load config, discover lanes,
│   │                                 #        upsert lanes + taxonomy, build executors, load models
│   ├── config.py                     # (impl) Settings via pydantic-settings (.env + settings.yaml)
│   │
│   ├── api/                          # HTTP layer only, no business logic
│   │   ├── __init__.py
│   │   ├── deps.py                   # (impl) db session, settings, runner dependencies
│   │   ├── schemas.py                # (impl) Pydantic response/request models
│   │   ├── routes_lanes.py           # GET /api/lanes
│   │   ├── routes_runs.py            # POST /api/runs, GET /api/runs/{id}, /results, /incidents
│   │   ├── routes_reports.py         # POST /api/runs/{id}/report, GET /api/reports/{id}, /download
│   │   └── routes_models.py          # GET /api/models (active model per lane + metrics)
│   │
│   ├── core/                         # shared machinery, lane owners do NOT rewrite this
│   │   ├── __init__.py
│   │   ├── contracts.py              # (impl) FeatureFrame, UnifiedPrediction, JobResult, ValidationResult
│   │   ├── lane_spec.py              # (impl) LaneSpec Pydantic model for spec.yaml
│   │   ├── discovery.py              # (impl) scan app/lanes/*/spec.yaml, load parser/features modules
│   │   ├── validator.py              # (impl) required/optional/unknown column handling + report
│   │   ├── taxonomy.py               # (impl) load lane taxonomy.yaml, upsert taxonomy_map, map labels
│   │   ├── training.py               # (impl) train_and_save(ff, lane_id, version)
│   │   ├── predictor.py              # (impl) load model, predict_proba, SHAP top 5, taxonomy, DB writes
│   │   ├── explain.py                # (impl) SHAP helpers
│   │   ├── calibration.py            # (stub) stretch
│   │   ├── job_service.py            # (impl) run_job(job_id): validate -> features -> predict -> store
│   │   └── finalizer.py              # (impl) when all jobs done: incidents, summary, run status
│   │
│   ├── execution/                    # one worker pool per lane
│   │   ├── __init__.py
│   │   └── runner.py                 # (impl) per-lane executors, submit(job), shutdown, n_jobs math
│   │
│   ├── lanes/                        # one folder per lane, one owner per folder
│   │   ├── _template/                # copy-me template (skipped by discovery)
│   │   │   ├── spec.yaml
│   │   │   ├── parser.py
│   │   │   ├── features.py
│   │   │   ├── train.py
│   │   │   ├── taxonomy.yaml
│   │   │   ├── sample/.gitkeep
│   │   │   └── README.md
│   │   ├── netflow/                  # King: CIC-IDS2017 CSVs
│   │   │   ├── __init__.py
│   │   │   ├── spec.yaml
│   │   │   ├── parser.py
│   │   │   ├── features.py
│   │   │   ├── train.py
│   │   │   ├── taxonomy.yaml
│   │   │   ├── sample/
│   │   │   │   ├── valid_sample.csv
│ │   │   │   ├── missing_optional.csv
│ │   │   │   └── missing_required.csv
│ │   │   └── README.md
│   │   ├── siem_edr/                 # Atharva: host events, unit = host per 5 min
│   │   │   ├── __init__.py
│ │   │   ├── spec.yaml
│ │   │   ├── parser.py
│ │   │   ├── features.py
│ │   │   ├── train.py
│ │   │   ├── taxonomy.yaml
│ │   │   ├── sample/
│ │   │   │   ├── valid_sample.csv
│ │   │   │   └── missing_optional.csv
│ │   │   └── README.md
│   │   └── cloud_entity/             # Mansi: principal events, unit = principal per 15 min
│   │       ├── __init__.py
│ │       │   ├── spec.yaml
│ │       │   ├── parser.py
│ │       │   ├── features.py
│ │       │   ├── train.py
│ │       │   ├── taxonomy.yaml
│ │       │   ├── sample/
│ │       │   │   └── valid_sample.csv
│ │       │   └── README.md
│   │
│   ├── orchestration/                # post-scoring logic, no model code
│   │   ├── __init__.py
│ │   ├── correlator.py             # (impl) rules R1 to R5
│ │   ├── scoring.py                # (impl) incident score + severity bands
│ │   ├── incident_builder.py       # (impl) writes incidents + incident_evidence
│ │   └── service.py                # (impl) build_incidents(run_id)
│   │
│   ├── llm/                          # LLM client, schema, service, fallback, prompts
│   │   ├── __init__.py
│   │   ├── client.py                 # (impl) httpx OpenAI-compatible client
│ │   │   ├── schema.py               # (impl) Pydantic report schema
│ │   │   ├── service.py              # (impl) prompt -> call -> validate -> retry once -> fallback
│ │   │   ├── fallback.py             # (impl) template remedies from knowledge/attack_snippets.yaml
│ │   │   └── prompts/
│ │   │   └── incident_report_v1.md
│ │
│   ├── reporting/                    # PDF renderer, charts, report service
│   │   ├── __init__.py
│ │   ├── pdf_renderer.py           # (impl) reportlab, built only from stored report_json
│ │   ├── charts.py                 # (impl) matplotlib severity chart
│ │   └── report_service.py         # (impl) create report row, call llm, render pdf, update status
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                # structured logging setup
│       ├── files.py                  # safe filenames, upload paths, size limits
│       └── timing.py
│
├── scripts/
│   ├── init_db.py                    # apply schema + migrations, verify 12 tables
│   ├── reset_db.py                   # DROP everything in public + re-apply (needs --yes)
│   ├── train_lane.py                 # thin wrapper: python scripts/train_lane.py netflow --data <path>
│   ├── register_models.py            # insert model_versions row from models/<lane>/<ver>/ and activate
│   ├── make_demo_data.py             # writes 3 sample files sharing one IP for corroboration demo
│   └── smoke_test.py                 # end-to-end: submit 3 files, poll, fetch incidents, download PDF
│
├── tests/
│   ├── conftest.py                   # test DB fixture, sample paths
│   ├── contract/
│   │   └── test_lane_contract.py     # parametrized over every discovered lane
│   ├── core/
│   │   ├── test_validator.py
│   │   └── test_taxonomy.py
│   ├── api/
│   │   └── test_runs_api.py
│   ├── orchestration/
│   │   └── test_rules_r1_r5.py       # scenario tests, one positive + one negative per rule
│   └── execution/
│       └── test_parallel_isolation.py
│
├── frontend/                         # Mansi: Vite + React + TypeScript + Recharts
│   ├── package.json
│   ├── vite.config.ts                # dev proxy /api -> http://localhost:8000
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts             # typed fetch wrappers for section 8
│       ├── api/mock.ts               # mocked JSON for early UI work
│       ├── types.ts
│       ├── components/
│       │   ├── UploadRows.tsx        # dropdown + file picker rows, Analyze button
│       │   ├── JobStatusList.tsx     # per-job status + validation warnings
│       │   ├── ResultsTable.tsx      # confidence bars, alternatives
│       │   ├── SeverityChart.tsx
│       │   ├── IncidentList.tsx
│       │   └── ReportButton.tsx
│       └── hooks/usePolling.ts       # 2 second polling
│
├── notebooks/                        # exploration only, never imported by app code
└── docs/
    ├── architecture.md
    └── er_diagram.png
```

### Structure rules

1. **One owner per folder.** `app/lanes/netflow` (King), `app/lanes/siem_edr` (Atharva), `app/lanes/cloud_entity` (Mansi). Everything under `app/core`, `app/api`, `app/db`, `app/execution`, `app/orchestration` is King's.
2. Lanes are **auto-discovered** from `app/lanes/*/spec.yaml`. Folders starting with `_` are skipped. There is no shared registry file to edit.
3. A lane may import from `app.core.*` but **never from another lane** and never from `app.db`.
4. `app.core` and `app.orchestration` never import a specific lane. They only use what discovery returns.
5. API routes call services. Services call repositories. Repositories talk to Postgres. No skipping layers.
6. Model artifacts and datasets are git-ignored.

---

## 4. Configuration files

### `.env.example`

```
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_soc
APP_ENV=dev
STORE_RAW=false
MAX_UPLOAD_MB=200

LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=local-model
LLM_API_KEY=not-needed
LLM_TIMEOUT_SECONDS=120
```

The human fills in `.env` with the real DB credentials. Never commit `.env`. Never hard-code credentials or absolute paths. Use `pathlib` relative to the project root (`Path(__file__).resolve().parents[1]`).

### `configs/settings.yaml`

```yaml
executor:
  type: thread            # thread | process
  default_workers_per_lane: 2
  lanes:
    netflow: {workers: 2}
    siem_edr: {workers: 2}
    cloud_entity: {workers: 2}
paths:
  uploads: data/uploads
  models: models
  reports: outputs/reports
ingestion:
  chunk_rows: 200000
  max_rejected_ratio: 0.5     # job fails if more than this share of rows is rejected
prediction:
  min_confidence_store: 0.0   # store everything
  shap_top_k: 5
  shap_max_rows: 2000         # sample for SHAP on big files, never skip predictions
```

### `configs/orchestrator.yaml`

```yaml
filter:
  min_confidence_default: 0.30
  per_lane: {}
windows:
  dedupe_minutes: 5           # R1
  multistage_minutes: 30      # R2
boosts:
  extra_lane: 0.10
  extra_tactic: 0.10
weak_signal_max_confidence: 0.50   # R5
severity_bands: {critical: 0.80, high: 0.60, medium: 0.40}
```

### `configs/llm.yaml`

```yaml
prompt_version: "1"
max_retries: 1
temperature: 0.2
max_incidents_in_prompt: 20
```

---

## 5. Lane specification (`spec.yaml`)

Each lane describes itself. Core code reads this through `LaneSpec` in `app/core/lane_spec.py`.

```yaml
lane_id: siem_edr
display_name: SIEM / EDR logs
version: "1"
formats: [csv, jsonl]
unit_of_analysis: host_window_5m      # flow | host_window_5m | principal_window_15m
entity: {type: host, field: host_name}
ip_field: src_ip                      # optional, enables cross-lane matching
time_field: timestamp
label_column: label                   # used only in training/evaluation
columns:
  required:
    - {name: timestamp, type: datetime}
    - {name: host_name, type: string}
    - {name: event_id, type: int}
  optional:
    - {name: user_name, type: string, default: "unknown"}
    - {name: src_ip, type: string, default: null, aliases: [ip, source_ip]}
behavior:
  on_missing_required: reject_job
  on_missing_optional: fill_default
  on_unknown_columns: ignore
```

### Validator behaviour (`app/core/validator.py`)

| Category | If missing | If bad values |
|---|---|---|
| required | Whole file rejected. Error lists missing names **and aliases that were tried**. | Row goes to `rejected_records` with a reason |
| optional | Column created with spec default. Warning in `validation_report`. | Replaced with default, counted |
| unknown | Ignored. Kept in raw payload only if `STORE_RAW=true`. | n/a |

Normalize column names first (strip whitespace, apply aliases), then check required, then coerce types.

`validation_report` JSON (stored on `ingestion_jobs.validation_report`):

```json
{
  "status": "ok | warning | rejected",
  "rows_total": 5120, "rows_valid": 5098, "rows_rejected": 22,
  "missing_required": [], "missing_optional": ["command_line"],
  "renamed": {"ip": "src_ip"},
  "coerced": {"timestamp": {"failed_rows": 22}},
  "unknown_columns": ["agent_version"]
}
```

Wrong lane selected (for example a cloud file sent to netflow) must produce a readable `missing required columns: ...` error and set the job to `failed`.

---

## 6. Lane contract (what every lane provides)

Lane files, nothing more:

```python
# parser.py
def parse(path: Path, fmt: str) -> pd.DataFrame:
    """Raw file to DataFrame. No cleaning. Validation happens in core."""

# features.py
@dataclass
class FeatureFrame:
    X: pd.DataFrame            # numeric only, one row per unit of analysis
    meta: pd.DataFrame         # same index: entity_type, entity_value, ip, target_value,
                               # window_start, window_end, event_count
    y: pd.Series | None        # native labels, only when label column exists

def build_features(df: pd.DataFrame, spec: LaneSpec) -> FeatureFrame:
    """Same function for training and scoring. No label-dependent logic."""

# train.py   (python -m app.lanes.<lane_id>.train --data <path>)
def main(data_path: Path) -> None:
    df = parse(data_path, "csv")
    df = core.validator.validate(df, spec).clean_df
    ff = build_features(df, spec)
    core.training.train_and_save(ff, lane_id="<lane_id>", version="1.0.0")
```

`FeatureFrame` and `LaneSpec` live in `app/core/contracts.py` and `app/core/lane_spec.py`. **Implement and push these first** (Phase 0) because all lane owners depend on them.

### Model and artifact rules (identical for all lanes)

- XGBoost `multi:softprob`, `tree_method=hist`.
- One class named exactly `BENIGN`. 4 to 7 classes. Merge rare classes.
- Numeric features in a fixed order stored in `feature_list.json`. Missing values imputed with medians from `train_stats.json`.
- Do not use identifiers as features (raw IP, host name, user name, file name, absolute timestamp). Use behaviour: counts, ratios, rates, distinct counts.
- Split by capture, host or time rather than random rows whenever possible. State it in the lane README.
- `metrics.json` contains `macro_f1`, per-class precision/recall/f1, and class counts used for training.
- Artifacts saved to `models/<lane_id>/<version>/`: `model.json`, `feature_list.json`, `class_labels.json`, `train_stats.json`, `metrics.json`.
- Sample weights: inverse square-root class frequency.
- XGBoost `n_jobs = max(1, total_cores // (num_lanes * workers_per_lane))`.

### Netflow specifics (CIC-IDS2017)

- Strip header whitespace, drop the duplicated `Fwd Header Length` column, replace `inf` with NaN then impute, drop exact duplicates in training.
- Drop Flow ID, IPs, ports as identifiers, and timestamp from features.
- Entity: `ip` = Source IP, `target_value` = Destination IP.
- Classes: `BENIGN, DoS, DDoS, PortScan, Brute Force (FTP+SSH Patator), Web Attack (3 merged), Bot`. Fold Heartbleed and Infiltration into `Other` or drop.
- Required columns: Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, plus the packet-length and flag-count fields the model needs.

### `taxonomy.yaml` per lane

```yaml
lane_id: netflow
classes:
  BENIGN:      {unified_class: Benign, tactic: None, mitre: [], weight: 0.0}
  PortScan:    {unified_class: "Reconnaissance: Network Scan", tactic: Reconnaissance, mitre: [T1046], weight: 0.5}
  Brute Force: {unified_class: "Credential Access: Brute Force", tactic: Credential Access, mitre: [T1110], weight: 0.85}
  DoS:         {unified_class: "Impact: Denial of Service", tactic: Impact, mitre: [T1499], weight: 0.9}
  DDoS:        {unified_class: "Impact: Network DoS", tactic: Impact, mitre: [T1498], weight: 0.95}
  Web Attack:  {unified_class: "Initial Access: Web Exploit", tactic: Initial Access, mitre: [T1190], weight: 0.9}
  Bot:         {unified_class: "Command and Control", tactic: Command and Control, mitre: [T1071], weight: 0.9}
```

On startup, core upserts these into `taxonomy_map` (`UNIQUE (lane_id, native_label)`). Every class in `class_labels.json` must have an entry.

---

## 7. Execution flow

1. **Startup (`app/main.py` lifespan):** load settings, check DB, discover lanes, upsert lanes and taxonomy_map, create one executor per lane via `execution/runner.py`, load active model per lane (lazy load is fine, but fail clearly if the lane has no active model when a job is submitted).
2. **`POST /api/runs`** (multipart, `files[]` and `lanes[]` in the same order):
   - validate that every `lane_id` exists and `is_enabled`
   - create `runs` row, save files to `data/uploads/<run_id>/`, create one `ingestion_jobs` row per file (`queued`)
   - submit each job to **its lane's executor**
   - return `{run_id, jobs:[...]}` immediately
3. **`core/job_service.run_job(job_id)`:** mark `running` -> `parse` -> `validate` (write `validation_report`, rejected rows) -> `build_features` -> `insert events` (bulk) -> `predict` -> `insert predictions` (bulk, with unified_class and tactic) -> mark `done`. Any exception sets the job to `failed` with `error` text. **A failed job must never break other jobs.**
4. **Finalizer (`core/finalizer.py`):** called after every job completion. When no job of the run is `queued`/`running`, call `orchestration.service.build_incidents(run_id)`, then set `runs.status='scored'` and `finished_at`. If every job failed, set `runs.status='failed'`.
5. **Report (`POST /api/runs/{id}/report`):** create `reports` row (`pending`), run LLM service in the background, render PDF, set `generated` or `failed`, run status becomes `reported`.
6. **Polling:** the UI calls `GET /api/runs/{id}` every 2 seconds.

### Predictor details (`core/predictor.py`)

- Load artifacts once per lane and cache them.
- Score in chunks. Reorder columns to `feature_list.json`, impute with `train_stats.json`.
- For each row store: `predicted_class` (native), `confidence` (max prob), full `class_probabilities`, `shap_top_features` (top 5, computed only for rows where predicted_class != BENIGN, capped by `shap_max_rows`), `unified_class`, `tactic`.
- Return the top alternative classes through `class_probabilities` (API sorts them).
- Predictions are linked to `events.event_id` and `model_versions.model_id`.

---

## 8. API contract (fixed, do not change names)

| Endpoint | Request | Response |
|---|---|---|
| `GET /api/lanes` | none | `[{lane_id, display_name, formats, is_enabled}]` |
| `POST /api/runs` | multipart `files[]`, `lanes[]` | `{run_id, jobs:[{job_id, lane_id, filename, status}]}` |
| `GET /api/runs/{run_id}` | none | `{status, jobs:[{job_id, lane_id, filename, status, rows_total, rows_valid, rows_rejected, validation_report, error}]}` |
| `GET /api/runs/{run_id}/results` | `?lane=&min_conf=&limit=` | `{rows:[{entity, window, predicted_class, confidence, class_probabilities, unified_class, tactic}]}` |
| `GET /api/runs/{run_id}/incidents` | none | `{incidents:[{incident_id, severity, score, entity, tactics, lanes, flags}]}` |
| `POST /api/runs/{run_id}/report` | none | `{report_id, status}` |
| `GET /api/reports/{report_id}` | none | report JSON |
| `GET /api/reports/{report_id}/download` | none | PDF file |
| `GET /api/models` | none | active model per lane with version and metrics |
| `GET /api/health` | none | `{db: ok, lanes: n}` |

Use CORS for `http://localhost:5173` in dev. All responses are JSON except the PDF.

---

## 13. Coding rules for the agent

- Python 3.11, type hints, `pathlib`, no hard-coded absolute paths. The project runs on **Windows** (`C:\ai_soc`) and must also work on Linux.
- Start the API with `uvicorn app.main:app --port 8000` (no `--reload` when process pools are enabled).
- Use `ThreadPoolExecutor` unless `executor.type: process` is set. If process mode is used, job functions must be top-level and the entry point must have a `__main__` guard.
- Never log full file contents or credentials.
- Sanitize upload filenames, enforce `MAX_UPLOAD_MB`, store uploads under `data/uploads/<run_id>/`.
- Large files: read in chunks (`chunk_rows`), bulk insert, keep `STORE_RAW=false` by default.
- Do not add Redis, Celery, Alembic, Docker-only requirements, or new top-level folders without being asked.
- Do not edit files outside the owner's folder when told to work on a lane.
- Prefer small commits per phase. Update `README.md` when a command changes.
- When something in this file is ambiguous, choose the simplest option that keeps the API contract in section 8 and the schema in section 2 unchanged, and write the assumption in `docs/architecture.md`.

---

## 14. Definition of done

- `python scripts/init_db.py` creates all 12 tables on a clean database.
- `GET /api/lanes` returns three lanes.
- Three files submitted together create three jobs that overlap in time and finish as `done`.
- Wrong lane selection fails with a readable missing-columns message. Missing optional column completes with a warning.
- `events` joined to `predictions` returns scored rows per run.
- Incidents are created per run with correct severity bands.
- A PDF can be downloaded for a run, with and without the LLM.
- `pytest` and `ruff check .` pass.
- Per-lane README states dataset, split method, classes, macro-F1, per-class recall, and limits (and says "synthetic" where applicable).