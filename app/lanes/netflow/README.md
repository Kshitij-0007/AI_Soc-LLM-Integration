# NetFlow lane

One prediction per network flow. Owner: King.

## Accepted inputs

The parser detects the exporter from the header and converts it to the
canonical columns in `spec.yaml` (`adapters.py`):

| Exporter | Detected by | Notes |
|---|---|---|
| canonical CSV | `flow_duration_ms`, `dst_port` | this lane's own column names |
| CICFlowMeter (CIC-IDS2017, CSE-CIC-IDS2018) | `Flow Duration` + `Total Fwd Packets` / `Tot Fwd Pkts` | µs -> ms, payload -> IP bytes |
| Zeek `conn.log` (CSV, JSON lines, TSV) | `id.orig_h`, `id.resp_h`, `id.resp_p` | flags from `history` |
| nProbe NetFlow v9 (UQ NF-* datasets) | `IPV4_SRC_ADDR`, `L4_DST_PORT`, `IN_PKTS` | flags from `TCP_FLAGS` bitmask |
| UNSW-NB15 | `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes` | no flags |

Other layouts must use the canonical names, otherwise the job fails with
`missing required columns: ...`.

## Features (feature set 2)

30 behavioural features, `features.py`: log sizes and duration, derived
rates, bytes per packet, direction shares, "no response", TCP flags seen,
protocol, port service group, and source context per 60 s (flows, distinct
destination IPs/ports, no-response share, same-port share, flows per pair).
No IPs, raw ports or timestamps. Missing values stay NaN (XGBoost handles
them natively).

## Model 2.0.0

* **Dataset:** CIC-IDS2017, all 8 capture files (`TrafficLabelling`), real
  traffic. Up to 30,000 flows per class per file, then per class overall.
* **Classes:** BENIGN, Bot, Brute Force (FTP/SSH-Patator), DDoS, DoS
  (Hulk, GoldenEye, slowloris, Slowhttptest), PortScan, Web Attack (brute
  force, XSS, SQLi). Infiltration (36 flows) and Heartbleed (11) are not
  modelled.
* **Split:** for every file and class, the latest 20% of flows by time are
  held out (27,601 rows). CV inside training groups flows by 5-minute capture
  block. No random row split.
* **Training:** XGBoost `multi:softprob`, `hist`, depth 6, eta 0.1, rounds
  from grouped 3-fold CV, inverse-sqrt class weights (normalised to mean
  1), plus low-weight copies with flags / context / ports blanked so the
  model copes with exporters that lack them.

Temporal holdout (same network as training):

| | value |
|---|---|
| Macro-F1 | **0.953** |
| Accuracy | 0.941 |
| Attack detection rate | 0.979 |
| Benign false positive rate | 0.007 |

| Class | Recall | Precision | n |
|---|---|---|---|
| BENIGN | 0.993 | 0.930 | 6,004 |
| Bot | 0.997 | 0.908 | 394 |
| Brute Force | 0.963 | 1.000 | 2,767 |
| DDoS | 0.947 | 0.835 | 6,000 |
| DoS | 0.808 | 1.000 | 6,000 |
| PortScan | 0.999 | 1.000 | 6,000 |
| Web Attack | 1.000 | 1.000 | 436 |

Most DoS misses (1,125 of 1,149) are predicted as DDoS, so they still raise an alert.

### Leave-one-file-out (unseen capture day)

`scripts/evaluate_netflow.py leave-one-file-out`: train on 7 files, test
on the 8th. Every attack class lives in a single CIC-IDS2017 file, so the
held-out attack class is one the model has never seen.

| Held-out file | Benign FPR | Unseen attack flagged as some attack |
|---|---|---|
| Monday (benign only) | 0.002 | - |
| Tuesday | 0.001 | Brute Force 0.000 |
| Wednesday | 0.002 | DoS 0.803 |
| Thursday morning | 0.003 | Web Attack 0.834 |
| Thursday afternoon | 0.003 | - |
| Friday morning | 0.000 | Bot 0.012 |
| Friday PortScan | 0.006 | PortScan 0.838 |
| Friday DDoS | 0.003 | DDoS 0.990 |

Mean benign FPR on unseen captures: **0.25%**.

## Limits (read before trusting a score)

* **One network.** All numbers come from CIC-IDS2017. A different network
  or exporter is likely to score lower. Measure it: get a second labelled
  dataset (CSE-CIC-IDS2018, UNSW-NB15, NF-UNSW-NB15-v2) and run
  `scripts/evaluate_netflow.py score --version 2.0.0 --data <files>`, or
  train with `--test-data <files>` so that dataset becomes the headline.
* **Supervised.** Attack types missing from training are flagged
  inconsistently (Brute Force and Bot almost never). Add labelled examples
  of each attack type you care about.
* **Synthetic data does not transfer.** `data/raw/netflow/netflow.csv` is
  not physically realistic (e.g. ICMP flows to port 5432). On it, 2.0.0
  detects none of its attacks and the compatibility report says `warning`. Don't
  use it for training or evaluation.
* **Check the compatibility report** (`validation_report.model_compatibility`
  on each job, also printed by `scripts/score_file.py`). `warning` means
  the file doesn't look like the training data and the metrics above
  don't apply.

## Commands

```bash
# train (temporal holdout)
python -m app.lanes.netflow.train --data "data/raw/netflow/real/TrafficLabelling " --version 2.0.0
# train on one dataset, evaluate on another (preferred once available)
python -m app.lanes.netflow.train --data <dataset A> --test-data <dataset B> --version 2.1.0
# evaluate
python scripts/evaluate_netflow.py leave-one-file-out --data "data/raw/netflow/real/TrafficLabelling "
python scripts/evaluate_netflow.py score --version 2.0.0 --data <labelled files>
# score one file like an upload
python scripts/score_file.py netflow <file.csv> --version 2.0.0
# check, register and activate
python scripts/register_models.py netflow 2.0.0 --check-only
python scripts/register_models.py netflow 2.0.0 --activate
```
