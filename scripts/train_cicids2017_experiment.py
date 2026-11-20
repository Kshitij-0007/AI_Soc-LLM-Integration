from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.lanes.netflow.features import compute_features


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/netflow/real"
OLD_MODEL_DIR = ROOT / "models/netflow/1.0.0"
OUT_DIR = ROOT / "models/netflow/cicids2017_experiment"
REPORT_DIR = ROOT / "outputs/cicids2017_experiment"

SEED = 42
MAX_PER_CLASS = 30000
CLASSES = ["BENIGN", "Bot", "Brute Force", "DoS", "PortScan"]

LABEL_MAP = {
    "BENIGN": "BENIGN",
    "BOT": "Bot",
    "FTP PATATOR": "Brute Force",
    "SSH PATATOR": "Brute Force",
    "DOS HULK": "DoS",
    "DOS GOLDENEYE": "DoS",
    "DOS SLOWLORIS": "DoS",
    "DOS SLOWHTTPTEST": "DoS",
    "PORTSCAN": "PortScan",
}

# CIC-IDS2017 exports often contain whitespace in column names.
def norm_label(value):
    return " ".join(
        str(value).strip().upper().replace("_", " ")
        .replace("-", " ").split()
    )


def find_column(columns, wanted):
    lookup = {str(c).strip().lower(): c for c in columns}
    key = wanted.strip().lower()
    if key not in lookup:
        raise ValueError(f"Missing required column: {wanted}")
    return lookup[key]


def canonicalize(raw):
    col = lambda name: find_column(raw.columns, name)
    result = pd.DataFrame(index=raw.index)

    result["timestamp"] = raw[col("Timestamp")]
    result["src_ip"] = raw[col("Source IP")].astype(str)
    result["dst_ip"] = raw[col("Destination IP")].astype(str)
    result["src_port"] = pd.to_numeric(raw[col("Source Port")], errors="coerce")
    result["dst_port"] = pd.to_numeric(raw[col("Destination Port")], errors="coerce")

    # CIC Flow Duration is microseconds; this pipeline expects milliseconds.
    result["flow_duration_ms"] = (
        pd.to_numeric(raw[col("Flow Duration")], errors="coerce") / 1000.0
    )

    pairs = {
        "total_fwd_packets": "Total Fwd Packets",
        "total_bwd_packets": "Total Backward Packets",
        "total_fwd_bytes": "Total Length of Fwd Packets",
        "total_bwd_bytes": "Total Length of Bwd Packets",
        "packets_per_second": "Flow Packets/s",
        "bytes_per_second": "Flow Bytes/s",
        "tcp_syn_count": "SYN Flag Count",
        "tcp_ack_count": "ACK Flag Count",
        "tcp_rst_count": "RST Flag Count",
    }
    for target, source in pairs.items():
        result[target] = pd.to_numeric(raw[col(source)], errors="coerce")

    result["total_packets"] = (
        result["total_fwd_packets"] + result["total_bwd_packets"]
    )
    result["total_bytes"] = (
        result["total_fwd_bytes"] + result["total_bwd_bytes"]
    )

    # CIC-IDS2017 does not provide these two counters. Leave them missing;
    # imputation will be fitted on training rows only.
    result["failed_connections"] = np.nan
    result["unique_destinations"] = np.nan
    return result


def load_data():
    files = sorted(DATA_ROOT.rglob("*ISCX.csv"))
    if not files:
        raise FileNotFoundError(f"No *ISCX.csv files found under {DATA_ROOT}")

    # Reservoir-like capped sampling per target class from each file keeps
    # memory usage manageable and prevents BENIGN from dominating the dataset.
    candidates = {label: [] for label in CLASSES}
    rng = np.random.default_rng(SEED)
    source_rows = {}

    print(f"Found {len(files)} CIC-IDS2017 files.")
    for path in files:
        print(f"Reading: {path.name}")
        file_seen = 0
        for chunk in pd.read_csv(
            path, encoding="latin1", low_memory=False, chunksize=50000
        ):
            label_col = find_column(chunk.columns, "Label")
            mapped = chunk[label_col].map(norm_label).map(LABEL_MAP)
            supported = mapped.notna()
            if not supported.any():
                continue

            usable = chunk.loc[supported].copy()
            usable["_target"] = mapped.loc[supported].to_numpy()
            file_seen += len(usable)

            for label, group in usable.groupby("_target"):
                # Keep a bounded candidate pool per class, per source file.
                current = candidates[label]
                current.append(group)
        source_rows[path.name] = file_seen

    parts = []
    for label, groups in candidates.items():
        if not groups:
            continue
        pool = pd.concat(groups, ignore_index=True)
        # Sampling after concatenation gives every class a bounded size.
        take = min(MAX_PER_CLASS, len(pool))
        pool = pool.sample(n=take, random_state=SEED)
        parts.append(pool)
        print(f"  {label:12s}: sampled {len(pool):,} rows from {len(groups)} chunks")

    data = pd.concat(parts, ignore_index=True)
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\nSelected target class counts:")
    print(data["_target"].value_counts().reindex(CLASSES, fill_value=0).to_string())
    return data, source_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not OLD_MODEL_DIR.exists():
        raise FileNotFoundError(f"Original model directory missing: {OLD_MODEL_DIR}")

    data, source_rows = load_data()
    y = data["_target"].astype(str)
    raw_features = canonicalize(data)
    X = compute_features(raw_features)

    # Split first. Only training data is used to fit imputation.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Fit medians on training rows only. Keep columns that are entirely
    # unavailable in CIC-IDS2017; fill those columns with 0.0 consistently.
    train_medians = X_train.median().fillna(0.0)
    X_train_filled = X_train.fillna(train_medians).fillna(0.0)
    X_valid_filled = X_valid.fillna(train_medians).fillna(0.0)
    X_test_filled = X_test.fillna(train_medians).fillna(0.0)

    X_train_np = X_train_filled.to_numpy(dtype=float)
    X_valid_np = X_valid_filled.to_numpy(dtype=float)
    X_test_np = X_test_filled.to_numpy(dtype=float)

    # Persist all 31 medians, including zero fallbacks for unavailable fields.
    medians = {name: float(value) for name, value in train_medians.items()}

    y_train_id = pd.Categorical(y_train, categories=CLASSES).codes
    y_valid_id = pd.Categorical(y_valid, categories=CLASSES).codes
    y_test_id = pd.Categorical(y_test, categories=CLASSES).codes

    dtrain = xgb.DMatrix(
        X_train_np, label=y_train_id, feature_names=list(X.columns)
    )
    dvalid = xgb.DMatrix(
        X_valid_np, label=y_valid_id, feature_names=list(X.columns)
    )
    dtest = xgb.DMatrix(
        X_test_np, label=y_test_id, feature_names=list(X.columns)
    )

    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASSES),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "eta": 0.1,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": SEED,
        "nthread": 4,
    }

    print("\nTraining experimental XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=30,
        verbose_eval=25,
    )

    probabilities = model.predict(dtest)
    predictions_id = probabilities.argmax(axis=1)
    predictions = np.array(CLASSES)[predictions_id]
    y_true = y_test.to_numpy()

    metrics = {
        "experiment": "CIC-IDS2017 five-class XGBoost",
        "seed": SEED,
        "rows_total": int(len(data)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "classes": CLASSES,
        "class_counts_total": {
            str(k): int(v)
            for k, v in y.value_counts().reindex(CLASSES, fill_value=0).items()
        },
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, labels=CLASSES,
                            average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, labels=CLASSES,
                         average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=CLASSES,
                     average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, labels=CLASSES,
                     average="weighted", zero_division=0)
        ),
        "best_iteration": int(model.best_iteration),
        "excluded_labels": [
            "DDoS", "Web Attack Brute Force", "Web Attack XSS",
            "Web Attack Sql Injection", "Infiltration", "Heartbleed"
        ],
        "notes": [
            "CIC-IDS2017 Flow Duration converted from microseconds to milliseconds.",
            "failed_connections and unique_destinations are unavailable and median-imputed.",
            "The test split was not used for training or early stopping.",
            "Random stratified split; related flows may share capture/session context.",
            "This experiment is separate from the original model artifacts."
        ],
        "source_rows_supported_by_file": source_rows,
    }

    model.save_model(str(OUT_DIR / "model.json"))
    (OUT_DIR / "feature_list.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    (OUT_DIR / "class_labels.json").write_text(
        json.dumps(CLASSES, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "train_stats.json").write_text(
        json.dumps({
            "medians": medians,
            "rows": int(len(X_train)),
            "features": int(X.shape[1]),
            "params": params,
            "best_iteration": int(model.best_iteration),
            "training_class_counts": {
                str(k): int(v)
                for k, v in y_train.value_counts().reindex(
                    CLASSES, fill_value=0
                ).items()
            },
        }, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_true, predictions, labels=CLASSES, zero_division=0
    )
    (REPORT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_true, predictions, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="CIC-IDS2017 Experimental Model — Test Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    cutoff = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:,}",
                ha="center", va="center",
                color="white" if matrix[i, j] > cutoff else "black",
            )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    predictions_out = pd.DataFrame({
        "actual_class": y_true,
        "predicted_class": predictions,
        "correct": y_true == predictions,
    })
    for i, label in enumerate(CLASSES):
        predictions_out[f"prob_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}"] = probabilities[:, i]
    predictions_out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    summary = [
        "CIC-IDS2017 FIVE-CLASS EXPERIMENT",
        f"Rows: {len(data)}",
        f"Train / validation / test: {len(X_train)} / {len(X_valid)} / {len(X_test)}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        f"Best boosting iteration: {model.best_iteration}",
        "",
        report,
    ]
    (REPORT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n===== EXPERIMENT RESULTS =====")
    print(f"Rows: {len(data):,}")
    print(f"Train / validation / test: {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\n" + report)
    print(f"\nModel saved to: {OUT_DIR}")
    print(f"Evaluation evidence saved to: {REPORT_DIR}")
    print("Original model directory was not modified:", OLD_MODEL_DIR)


if __name__ == "__main__":
    main()
