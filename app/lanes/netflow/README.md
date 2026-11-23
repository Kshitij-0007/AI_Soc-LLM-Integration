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

## Public 100k model

`public-100k-v1` uses 30 numeric features and seven labels. The source
checksums were verified for this fit. The trainer derived features per file and selected
100,000 distinct labeled feature rows without replacement (seed 42) from
11 training files. It drew from CIC-IDS2017, CSE-CIC-IDS2018 and
NF-ToN-IoT-v2. Labels with identical feature vectors are deduplicated
within class. Examples with unmapped labels are excluded. The sampler
splits its row budget across the available classes, and the trainer applies
inverse-square-root class weights. This changes the class proportions
relative to production traffic.

Three grouped validation folds within training selected 151 boosting
rounds via early stopping. The groups use five-minute capture blocks, or
row blocks for files missing timestamps. The saved booster is fitted on
all 100,000 selected examples. Five separate capture files supply the
45,326 mapped evaluation rows. Another 1,851 IoT records in those files
have unmapped attack families.

| Class | Training examples | Held-out examples | Held-out recall |
|---|---:|---:|---:|
| BENIGN | 28,940 | 30,475 | 83.8% |
| Bot | 323 | 1 | 0.0% |
| Brute Force | 5,553 | 0 | N/A |
| DDoS | 28,939 | 7,337 | 25.6% |
| DoS | 28,939 | 1,307 | 64.7% |
| PortScan | 1,986 | 3,797 | 99.7% |
| Web Attack | 5,320 | 2,409 | 99.2% |

On the same five held-out capture samples, accuracy moved from 75.40% to
76.00% and macro-F1 from 0.568 to 0.587 compared with the 10k model.
On 175,341 independent UNSW-NB15 flows, attack recall moved from 34.22%
to 34.69% and the benign false-positive rate fell from 8.88% to 7.60%.
The model missed 77,942 attacks there. The single held-out Bot example
and absence of Brute Force examples limit conclusions about those families.

The 100k fit uses labeled public lab captures. The bundled synthetic CSV
only exercises the upload pipeline. Public benchmark results can overstate
accuracy on a new network, particularly where exports omit endpoint
addresses or timestamps. Collect recent labeled captures before relying
on alert accuracy in a production network.

### Files and reproduction

The single booster lives in `models/netflow_model/public-100k-v1/model.json`.
The same folder contains evaluation metrics in `metrics.json` and training
class counts, feature ranges, fold scores and tree settings in
`train_stats.json`. The scorer uses the manifest, feature list and label
list to interpret the booster. PostgreSQL also serves registered metrics
through `GET /api/models`.

On a fresh checkout, these commands download pinned byte ranges, then
fit the 100k model. Each public sample's URL, SHA-256 and source byte
ranges are recorded in `data/raw/netflow/real/public-100k/sources.json`.

```powershell
$captures = "data/raw/netflow/real/public-100k"
python scripts/fetch_netflow_benchmark.py --output $captures --roles train train-2018 iot-train test final iot-test --training-block-bytes 1572864
python -m app.lanes.netflow.train --data "$captures/train" "$captures/train-2018" "$captures/iot-train" --test-data "$captures/test" "$captures/final" "$captures/iot-test" --version public-100k-v1 --max-per-class 0 --training-rows 100000 --no-augment --cv-folds 3 --data-source "11 pinned CIC-IDS2017, CSE-CIC-IDS2018 and NF-ToN-IoT-v2 training capture samples; input hashes in data/raw/netflow/real/public-100k/sources.json"
python scripts/register_models.py netflow public-100k-v1 --check-only
```

To reproduce the separate UNSW binary evaluation, use the binary evaluator
with the mirror's `test.csv` and `--label-column label`. The pinned URL,
SHA-256 and confusion counts appear under `external_binary` in the metrics.
