from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.lanes.netflow.features import compute_features

MODEL_DIR = ROOT / "models/netflow/1.0.0"
INPUT = (
    ROOT / "data/raw/netflow/real/TrafficLabelling "
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
OUT = ROOT / "outputs/cicids2017_ddos_evaluation"

CLASSES = ["BENIGN", "Bot", "Brute Force", "DoS", "PortScan"]
BINARY = ["BENIGN", "DDoS"]


def find_col(df, name):
    matches = [c for c in df.columns if c.strip().lower() == name.lower()]
    if not matches:
        raise ValueError(f"Missing CIC-IDS2017 column: {name}")
    return matches[0]


def canonicalize(df):
    # Normalize header whitespace for consistent column lookup.
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    def num(name):
        return pd.to_numeric(df[name], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["Timestamp"]
    out["src_ip"] = df["Source IP"]
    out["dst_ip"] = df["Destination IP"]
    out["src_port"] = num("Source Port")
    out["dst_port"] = num("Destination Port")
    out["protocol"] = num("Protocol")

    # CIC-IDS2017 Flow Duration is in microseconds; model expects milliseconds.
    out["flow_duration_ms"] = num("Flow Duration") / 1000.0

    out["total_fwd_packets"] = num("Total Fwd Packets")
    out["total_bwd_packets"] = num("Total Backward Packets")
    out["total_fwd_bytes"] = num("Total Length of Fwd Packets")
    out["total_bwd_bytes"] = num("Total Length of Bwd Packets")

    out["total_packets"] = (
        out["total_fwd_packets"] + out["total_bwd_packets"]
    )
    out["total_bytes"] = out["total_fwd_bytes"] + out["total_bwd_bytes"]
    out["packets_per_second"] = num("Flow Packets/s")
    out["bytes_per_second"] = num("Flow Bytes/s")
    out["tcp_syn_count"] = num("SYN Flag Count")
    out["tcp_ack_count"] = num("ACK Flag Count")
    out["tcp_rst_count"] = num("RST Flag Count")

    # These two fields are not supplied by this CIC-IDS2017 CSV.
    # The saved training medians will be used during imputation.
    out["failed_connections"] = np.nan
    out["unique_destinations"] = np.nan

    out["actual_label"] = df["Label"].astype(str).str.strip()
    return out


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    OUT.mkdir(parents=True, exist_ok=True)

    feature_names = json.loads(
        (MODEL_DIR / "feature_list.json").read_text()
    )
    class_labels = json.loads(
        (MODEL_DIR / "class_labels.json").read_text()
    )
    stats = json.loads((MODEL_DIR / "train_stats.json").read_text())
    medians = pd.Series(stats.get("medians", {}), dtype=float).reindex(
        feature_names
    )

    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / "model.json"))

    prediction_parts = []
    total_rows = 0

    print("Evaluating original five-class model on CIC-IDS2017 DDoS CSV")
    print(f"Input: {INPUT}")
    print(f"Model: {MODEL_DIR}")
    print("Reading and scoring in chunks of 25,000 rows...")

    for chunk in pd.read_csv(
        INPUT, encoding="latin1", low_memory=False, chunksize=25000
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        canonical = canonicalize(chunk)
        actual = canonical["actual_label"].to_numpy()

        X = compute_features(canonical)
        X = X.reindex(columns=feature_names)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(medians).fillna(0.0)

        probabilities = model.predict(
            xgb.DMatrix(X, feature_names=feature_names)
        )
        pred_indices = probabilities.argmax(axis=1)
        predicted_class = np.array(class_labels)[pred_indices]

        # Binary alert rule: BENIGN prediction = benign; any other
        # model prediction = attack alert. DDoS remains the actual label.
        predicted_binary = np.where(
            predicted_class == "BENIGN", "BENIGN", "DDoS"
        )

        result = pd.DataFrame({
            "actual_label": actual,
            "actual_binary": np.where(actual == "BENIGN", "BENIGN", "DDoS"),
            "predicted_class": predicted_class,
            "predicted_binary": predicted_binary,
            "confidence": probabilities.max(axis=1),
        })

        for i, label in enumerate(class_labels):
            safe = "".join(
                ch.lower() if ch.isalnum() else "_"
                for ch in label
            ).strip("_")
            result[f"probability_{safe}"] = probabilities[:, i]

        prediction_parts.append(result)
        total_rows += len(result)
        print(f"  Scored {total_rows:,} flows")

    results = pd.concat(prediction_parts, ignore_index=True)
    y_true = results["actual_binary"]
    y_pred = results["predicted_binary"]

    matrix = confusion_matrix(y_true, y_pred, labels=BINARY)
    report = classification_report(
        y_true, y_pred, labels=BINARY, zero_division=0
    )

    metrics = {
        "experiment": "CIC-IDS2017 DDoS binary evaluation",
        "model_version": "1.0.0",
        "rows": int(len(results)),
        "actual_class_counts": {
            str(k): int(v)
            for k, v in results["actual_label"].value_counts().items()
        },
        "binary_mapping": (
            "Actual BENIGN remains BENIGN; every other actual label is DDoS. "
            "Predicted BENIGN means no alert; any other predicted model "
            "class means attack alert."
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(
            precision_score(y_true, y_pred, pos_label="DDoS", zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_true, y_pred, pos_label="DDoS", zero_division=0)
        ),
        "f1_attack": float(
            f1_score(y_true, y_pred, pos_label="DDoS", zero_division=0)
        ),
        "confusion_matrix_labels": BINARY,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "limitations": [
            "The model has no dedicated DDoS output class.",
            "Non-BENIGN model predictions are treated as attack alerts.",
            "failed_connections and unique_destinations are unavailable in this CSV.",
            "This is a cross-dataset evaluation, not a DDoS-specific classifier test."
        ],
    }

    results.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "classification_report.txt").write_text(report, encoding="utf-8")
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=BINARY,
        yticklabels=BINARY,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="CIC-IDS2017 DDoS — Binary Alert Evaluation",
    )
    cutoff = matrix.max() / 2 if matrix.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{matrix[i, j]:,}",
                ha="center", va="center",
                color="white" if matrix[i, j] > cutoff else "black",
            )
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    summary = [
        "CIC-IDS2017 DDoS BINARY EVALUATION",
        f"Rows evaluated: {len(results):,}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"DDoS precision: {metrics['precision_attack']:.4f}",
        f"DDoS recall: {metrics['recall_attack']:.4f}",
        f"DDoS F1-score: {metrics['f1_attack']:.4f}",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        str(matrix),
        "",
        "Classification report:",
        report,
        "",
        "IMPORTANT: This maps all non-BENIGN predictions to attack alerts.",
        "The model has no dedicated DDoS output class.",
    ]
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n===== DDoS BINARY EVALUATION RESULTS =====")
    print(f"Rows evaluated: {len(results):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"DDoS precision: {metrics['precision_attack']:.4f}")
    print(f"DDoS recall: {metrics['recall_attack']:.4f}")
    print(f"DDoS F1-score: {metrics['f1_attack']:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix)
    print("\n" + report)
    print(f"\nEvidence saved to: {OUT}")


if __name__ == "__main__":
    main()
