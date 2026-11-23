# AI-SOC architecture notes: model layer

This file records how log files become predictions, and the assumptions made
where AGENTS.md was silent.

## Pipeline per uploaded file

```
upload (user picks log type = lane)
  -> lanes/<lane>/parser.py      detect exporter, convert to canonical columns
  -> core/validator.py           required / optional / unknown columns
  -> lanes/<lane>/features.py    one numeric row per unit of analysis
  -> core/predictor.py           model manifest, NaN policy, probabilities,
                                 TreeSHAP top-k, compatibility report
  -> core/taxonomy.py            native class -> unified class + MITRE tactic
  -> db/bulk.py                  batched events + predictions
```

`core/job_service.score_frame()` runs predictor + taxonomy without the DB.
`scripts/score_file.py` uses it to score any file from the command line.

## Adding a log type (lane)

1. Copy `app/lanes/_template/` to `app/lanes/<lane_id>/`.
2. `spec.yaml`: canonical columns, aliases, `source_formats`.
3. `parser.py`: convert every exporter you support into the canonical
   columns (see `app/lanes/netflow/adapters.py` for the pattern). Unit
   conversion belongs here, not in aliases.
4. `features.py`: `build_features()`, `FEATURE_COLUMNS` and
   `FEATURE_SET_VERSION`. Bump the version whenever features change.
5. `taxonomy.yaml`: one entry per class, including `BENIGN`.
6. `train.py`: call `app.core.training.train_and_save(..., eval_frame=...,
   feature_set_version=FEATURE_SET_VERSION)`.
7. `python scripts/register_models.py <lane_id> <version> --activate`.

Discovery picks up the folder automatically. Worker count defaults to
`executor.default_workers_per_lane` in `configs/settings.yaml`, and XGBoost
threads are `cores // (lanes * workers)` (`app/core/resources.py`).

## Model folders and manifests

NetFlow stores each version in `models/netflow_model/<version>/`. Upload routing,
model manifests and PostgreSQL use the lane ID `netflow`. Other lanes resolve to
`models/<lane>/<version>/`. Each version contains the following artifacts.

| file | required | content |
|---|---|---|
| `model.json` / `model.ubj` | yes | XGBoost booster |
| `class_labels.json` | yes | class names in model output order |
| `feature_list.json` | yes unless the booster stores feature names | feature order |
| `manifest.json` | recommended | format, NaN policy, feature set version, data source |
| `train_stats.json` | optional | medians and per-feature ranges (compatibility report) |
| `metrics.json` | optional | holdout and cross-validation metrics |

External models: `scripts/convert_xgb_model.py` turns a `.json`/`.ubj`
booster or a pickled scikit-learn `XGBClassifier` into such a folder.
Binary (`binary:logistic`) and multi-class (`multi:softprob`) models are
both supported. The app never unpickles files.

The predictor refuses to score when the model's `feature_set_version`
differs from the lane's `FEATURE_SET_VERSION`, or when the lane doesn't
produce a feature the model needs. A silent feature mismatch was one cause
of the earlier accuracy collapse.

## Why accuracy dropped on new CSVs, and the countermeasures

| Cause | Countermeasure |
|---|---|
| Model 1.0.0 trained on 1,840 synthetic rows | 2.0.0 trained on real CIC-IDS2017 flows |
| Same column name, different meaning per exporter (SYN count vs 0/1 flag, payload vs IP bytes, µs vs ms) | Per-exporter converters (`adapters.py`) to one canonical schema; flags used only as seen/not seen |
| Features that no real exporter provides (`failed_connections`, `unique_destinations`) were median-imputed | Removed; replaced by source-context features computed from the file itself |
| Exporter-computed rates | Rates derived from packets / bytes / duration |
| Median imputation invents "typical" values | NaN passed to XGBoost (native missing handling) |
| Random row split leaked neighbouring flows (inflated 95-97%) | Temporal holdout per file and class, grouped CV by 5-minute capture block, leave-one-file-out and cross-dataset evaluation scripts |
| Unfamiliar uploads scored silently | Compatibility report on every job (`validation_report.model_compatibility`), job status becomes `warning` |
| Class weights too small for `min_child_weight` | Weights normalised to mean 1 |

## Assumptions

* **Canonical bytes are IP-layer bytes.** CICFlowMeter reports payload
  bytes; the converter adds the exporter's header length (or 40/28 bytes
  per TCP/UDP packet when that value is invalid). For UNSW-NB15 `sbytes`
  and `dbytes`, IP-layer bytes are assumed.
* **CIC-IDS2017 timestamps** use a 12-hour clock without AM/PM. Hours 1-7
  are treated as afternoon, since the captures ran during working hours.
* **Source-context window = 60 s**, computed per uploaded file. A file
  that holds only a filtered slice of the traffic gives smaller counts.
  Shares and ratios are less affected than counts.
* **Files without timestamps** (e.g. NF-v2 datasets) get the job's
  processing time as `window_start` in `events`, because the column is
  NOT NULL.
* **Labels not modelled** (Infiltration, Heartbleed, Fuzzers, ...) are
  excluded from training, not merged into an "Other" class.
* **The model reports native classes only.** Confidence and compatibility
  are separate signals; severity and risk belong in `app/orchestration/`.

## Local NetFlow setup

The API registers enabled lanes at startup and shares one executor per lane.
An upload commits its job IDs and staged files before dispatch. Workers own
their sessions and write linked events and predictions in one transaction.
The last worker locks the run before building incidents and finishing it.
Local setup and the HTTP smoke test are documented in the root README.
The bundled demo verifies the connections using synthetic traffic.
