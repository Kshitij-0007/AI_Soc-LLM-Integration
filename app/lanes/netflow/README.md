# Netflow Lane

## Dataset
- **Source**: CIC-IDS2017 dataset
- **Unit of analysis**: One flow record
- **Features**: Network flow features including duration, packet counts, flag counts, and packet lengths
- **Classes**: BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Bot
- **Class merging**: Web Attack classes merged into "Web Attack"; Heartbleed and Infiltration folded into "Other" or dropped
- **Macro-F1**: Reported per model version
- **Per-class recall**: Reported per model version

## Data splits
- Training captured using full capture files
- No random row split; all records from capture used

## Model artifacts
- `models/netflow/1.0.0/model.json` - XGBoost model (multi:softprob, tree_method=hist)
- `models/netflow/1.0.0/feature_list.json` - Numeric feature names in fixed order
- `models/netflow/1.0.0/class_labels.json` - Class labels: BENIGN, PortScan, Brute Force, DoS, DDoS, Web Attack, Bot
- `models/netflow/1.0.0/train_stats.json` - Class counts and macro-F1
- `models/netflow/1.0.0/metrics.json` - Per-class precision/recall/f1 and class counts

## Training
- XGBoost with `multi:softprob` objective, `tree_method=hist`
- Sample weights: inverse square-root class frequency
- Rare classes merged into "Other"
- Numeric features imputed with medians from train_stats.json

## Requirements
- Required columns: Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, plus packet-length and flag-count fields
- Identifier columns (Flow ID, IPs, ports, timestamp) dropped from features