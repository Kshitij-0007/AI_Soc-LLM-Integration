# Cloud Entity Lane

## Dataset
- **Source**: Simulated cloud entity event logs (AWS, Azure, GCP)
- **Unit of analysis**: Principal per 15-minute window
- **Features**: Cloud security events including IAM actions, S3 access, EC2 instance states, and VPC flow logs
- **Classes**: BENIGN, Initial Access, Persistence, Command and Control, Exfiltration
- **Macro-F1**: Reported per model version
- **Per-class recall**: Reported per model version

## Data splits
- Training captured using labeled cloud security event data
- No random row split; all records from captured event logs used

## Model artifacts
- `models/cloud_entity/1.0.0/model.json` - XGBoost model (multi:softprob, tree_method=hist)
- `models/cloud_entity/1.0.0/feature_list.json` - Numeric feature names in fixed order
- `models/cloud_entity/1.0.0/class_labels.json` - Class labels: BENIGN, Initial Access, Persistence, Command and Control, Exfiltration
- `models/cloud_entity/1.0.0/train_stats.json` - Class counts and macro-F1
- `models/cloud_entity/1.0.0/metrics.json` - Per-class precision/recall/f1 and class counts

## Training
- XGBoost with `multi:softprob` objective, `tree_method=hist`
- Sample weights: inverse square-root class frequency
- Rare classes handled via thresholding
- Numeric features imputed with medians from train_stats.json

## Requirements
- Required columns: timestamp, principal_name, event_id
- Identifier columns (principal names, user names, timestamps) dropped from features