# SIEM / EDR Lane

## Dataset
- **Source**: Simulated SIEM and EDR log data
- **Unit of analysis**: Host per 5-minute window
- **Features**: Security event logs including authentication, process execution, file changes, and network connections
- **Classes**: BENIGN, Lateral Movement, Persistence, Credential Access, Defense Evasion
- **Macro-F1**: Reported per model version
- **Per-class recall**: Reported per model version

## Data splits
- Training captured using labeled security event logs
- No random row split; all records from captured logs used

## Model artifacts
- `models/siem_edr/1.0.0/model.json` - XGBoost model (multi:softprob, tree_method=hist)
- `models/siem_edr/1.0.0/feature_list.json` - Numeric feature names in fixed order
- `models/siem_edr/1.0.0/class_labels.json` - Class labels: BENIGN, Lateral Movement, Persistence, Credential Access, Defense Evasion
- `models/siem_edr/1.0.0/train_stats.json` - Class counts and macro-F1
- `models/siem_edr/1.0.0/metrics.json` - Per-class precision/recall/f1 and class counts

## Training
- XGBoost with `multi:softprob` objective, `tree_method=hist`
- Sample weights: inverse square-root class frequency
- Rare classes handled via thresholding
- Numeric features imputed with medians from train_stats.json

## Requirements
- Required columns: timestamp, host_name, event_id
- Identifier columns (host names, user names, timestamps) dropped from features