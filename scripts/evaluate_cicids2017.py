from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.lanes.netflow.features import compute_features


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "netflow" / "real"
MODEL_DIR = ROOT / "models" / "netflow" / "1.0.0"
OUT_DIR = ROOT / "outputs" / "cicids2017_evaluation"

MAX_PER_CLASS = 5000
CHUNK_SIZE = 50000
SEED = 42

CLASSES = ["BENIGN", "Bot", "Brute Force", "DoS", "PortScan"]

FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

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


def norm(value: object) -> str:
    return " ".join(
        str(value).strip().upper().replace("_", " ").replace("-", " ").split()
    )


def find_column(columns, name: str) -> str:
    lookup = {str(c).strip().lower(): c for c in columns}
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(f"Required CIC-IDS2017 column not found: {name}")
    return lookup[key]


def make_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert CIC-IDS2017 fields into the units and names expected by the lane."""
    col = lambda name: find_column(raw.columns, name)
    result = pd.DataFrame(index=raw.index)

    result["timestamp"] = raw[col("Timestamp")]
    result["src_ip"] = raw[col("Source IP")].astype(str)
    result["dst_ip"] = raw[col("Destination IP")].astype(str)
    result["src_port"] = pd.to_numeric(
        raw[col("Source Port")], errors="coerce"
    )
    result["dst_port"] = pd.to_numeric(
        raw[col("Destination Port")], errors="coerce"
    )

    # CIC-IDS2017 Flow Duration is in microseconds.
    # The model's feature pipeline expects milliseconds.
    duration_us = pd.to_numeric(raw[col("Flow Duration")], errors="coerce")
    result["flow_duration_ms"] = duration_us / 1000.0

    result["total_fwd_packets"] = pd.to_numeric(
        raw[col("Total Fwd Packets")], errors="coerce"
    )
    result["total_bwd_packets"] = pd.to_numeric(
        raw[col("Total Backward Packets")], errors="coerce"
    )
    result["total_fwd_bytes"] = pd.to_numeric(
        raw[col("Total Length of Fwd Packets")], errors="coerce"
    )
    result["total_bwd_bytes"] = pd.to_numeric(
        raw[col("Total Length of Bwd Packets")], errors="coerce"
    )

    result["total_packets"] = (
        result["total_fwd_packets"] + result["total_bwd_packets"]
    )
    result["total_bytes"] = (
        result["total_fwd_bytes"] + result["total_bwd_bytes"]
    )

    result["packets_per_second"] = pd.to_numeric(
        raw[col("Flow Packets/s")], errors="coerce"
    )
    result["bytes_per_second"] = pd.to_numeric(
        raw[col("Flow Bytes/s")], errors="coerce"
    )

    result["tcp_syn_count"] = pd.to_numeric(
        raw[col("SYN Flag Count")], errors="coerce"
    )
    result["tcp_ack_count"] = pd.to_numeric(
        raw[col("ACK Flag Count")], errors="coerce"
    )
    result["tcp_rst_count"] = pd.to_numeric(
        raw[col("RST Flag Count")], errors="coerce"
    )

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for artifact in (
        "model.json",
        "feature_list.json",
        "class_labels.json",
        "train_stats.json",
    ):
        if not (MODEL_DIR / artifact).is_file():
            raise FileNotFoundError(f"Missing model artifact: {MODEL_DIR / artifact}")

    feature_names = json.loads(
        (MODEL_DIR / "feature_list.json").read_text(encoding="utf-8")
    )
    model_classes = json.loads(
        (MODEL_DIR / "class_labels.json").read_text(encoding="utf-8")
    )
    train_stats = json.loads(
        (MODEL_DIR / "train_stats.json").read_text(encoding="utf-8")
    )

    if model_classes != CLASSES:
        raise ValueError(
            f"Expected classes {CLASSES}, but model contains {model_classes}. "
            "Stop: update the evaluation class order before continuing."
        )

    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / "model.json"))

    rng = np.random.default_rng(SEED)
    candidates = []
    source_counts = Counter()
    mapped_counts = Counter()
    unsupported_counts = Counter()
    missing_features = []

    print("=" * 76)
    print("CIC-IDS2017 EXTERNAL EVALUATION — EXISTING XGBOOST MODEL")
    print("=" * 76)
    print(f"Model: {MODEL_DIR}")
    print(f"Maximum sample per class: {MAX_PER_CLASS:,}")
    print("Model will not be retrained.")
    print("\nReading datasets in chunks...")

    for filename in FILES:
        matches = list(DATA_ROOT.rglob(filename))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {filename} under {DATA_ROOT}"
            )

        path = matches[0]
        print(f"\nDataset: {filename}")

        for chunk in pd.read_csv(
            path, chunksize=CHUNK_SIZE, encoding="latin1", low_memory=False
        ):
            label_col = find_column(chunk.columns, "Label")
            original_labels = chunk[label_col].astype(str).str.strip()
            canonical_labels = original_labels.map(
                lambda value: LABEL_MAP.get(norm(value))
            )

            for value, count in norm_counts(original_labels).items():
                source_counts[value] += count

            for value, count in canonical_labels.value_counts().items():
                mapped_counts[value] += int(count)

            unknown = canonical_labels.isna()
            if unknown.any():
                for value, count in norm_counts(original_labels[unknown]).items():
                    unsupported_counts[value] += count

            keep = canonical_labels.notna()
            if not keep.any():
                continue

            raw_supported = chunk.loc[keep].copy()
            actual = canonical_labels.loc[keep].reset_index(drop=True)
            canonical = make_canonical(raw_supported).reset_index(drop=True)

            features = compute_features(canonical)
            features = features.replace([np.inf, -np.inf], np.nan)

            # Preserve identifying/context fields for the prediction evidence.
            records = features.copy()
            records["actual_class"] = actual.to_numpy()
            records["source_label"] = original_labels.loc[keep].to_numpy()
            records["source_ip"] = canonical["src_ip"].to_numpy()
            records["destination_ip"] = canonical["dst_ip"].to_numpy()
            records["destination_port"] = canonical["dst_port"].to_numpy()
            records["flow_duration_ms"] = canonical["flow_duration_ms"].to_numpy()
            records["timestamp"] = canonical["timestamp"].to_numpy()

            # Sample each class from every chunk to avoid selecting only the
            # beginning of a file, which may contain mostly benign records.
            for class_name, group in records.groupby("actual_class"):
                take = min(500, len(group))
                if take:
                    candidates.append(
                        group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
                    )

        print("Finished reading file.")

    if not candidates:
        raise ValueError("No supported labeled records were found.")

    pool = pd.concat(candidates, ignore_index=True)
    samples = []

    for class_name in CLASSES:
        group = pool[pool["actual_class"] == class_name]
        if group.empty:
            print(f"WARNING: No samples available for {class_name}")
            continue
        take = min(MAX_PER_CLASS, len(group))
        samples.append(
            group.sample(n=take, random_state=SEED)
        )

    evaluation = pd.concat(samples, ignore_index=True)
    evaluation = evaluation.sample(frac=1, random_state=SEED).reset_index(drop=True)

    y_true = evaluation["actual_class"].astype(str).to_numpy()
    X = evaluation.reindex(columns=feature_names).copy()

    absent = [name for name in feature_names if name not in evaluation.columns]
    if absent:
        missing_features = absent

    medians = pd.Series(
        train_stats.get("medians", {}), dtype=float
    ).reindex(feature_names)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians).fillna(0.0)

    probabilities = model.predict(
        xgb.DMatrix(X, feature_names=feature_names)
    )
    predictions = np.array(model_classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)

    report = classification_report(
        y_true,
        predictions,
        labels=CLASSES,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        predictions,
        labels=CLASSES,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(y_true, predictions, labels=CLASSES)

    # Attack-vs-benign metrics: all supported attack classes count as attack.
    actual_attack = y_true != "BENIGN"
    predicted_attack = predictions != "BENIGN"
    tn = int((~actual_attack & ~predicted_attack).sum())
    fp = int((~actual_attack & predicted_attack).sum())
    fn = int((actual_attack & ~predicted_attack).sum())
    tp = int((actual_attack & predicted_attack).sum())

    metrics = {
        "dataset": "CIC-IDS2017",
        "model_version": "1.0.0",
        "model_retrained": False,
        "classes": CLASSES,
        "evaluation_rows": int(len(evaluation)),
        "class_distribution": {
            str(k): int(v)
            for k, v in evaluation["actual_class"].value_counts().items()
        },
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(
            y_true, predictions, labels=CLASSES, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, predictions, labels=CLASSES, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, predictions, labels=CLASSES, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, predictions, labels=CLASSES, average="weighted", zero_division=0
        )),
        "attack_vs_benign": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        },
        "confusion_matrix_labels_order": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "unsupported_source_label_counts": dict(unsupported_counts),
        "feature_mapping_note": (
            "CIC Flow Duration converted from microseconds to milliseconds. "
            "failed_connections and unique_destinations are unavailable in this "
            "CIC export and are imputed with the saved training medians. "
            "Treat results as an external benchmark, not production validation."
        ),
        "missing_model_features": missing_features,
    }

    output = evaluation[
        [
            "timestamp",
            "source_ip",
            "destination_ip",
            "destination_port",
            "flow_duration_ms",
            "source_label",
            "actual_class",
        ]
    ].copy()
    output["predicted_class"] = predictions
    output["confidence"] = confidence

    for index, label in enumerate(model_classes):
        safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        output[f"probability_{safe}"] = probabilities[:, index]

    predictions_path = OUT_DIR / "predictions.csv"
    output.to_csv(predictions_path, index=False)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="CIC-IDS2017 — XGBoost Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    cutoff = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, f"{matrix[row, col]:,}",
                ha="center", va="center",
                color="white" if matrix[row, col] > cutoff else "black",
            )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    summary = [
        "=" * 76,
        "CIC-IDS2017 EXTERNAL EVALUATION RESULTS",
        "=" * 76,
        f"Evaluation rows: {len(evaluation):,}",
        f"Accuracy:        {metrics['accuracy']:.4f}",
        f"Macro precision: {metrics['macro_precision']:.4f}",
        f"Macro recall:    {metrics['macro_recall']:.4f}",
        f"Macro F1:        {metrics['macro_f1']:.4f}",
        f"Weighted F1:     {metrics['weighted_f1']:.4f}",
        "",
        "Class distribution:",
        evaluation["actual_class"].value_counts().reindex(CLASSES, fill_value=0).to_string(),
        "",
        "Classification report:",
        report_text,
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        str(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)),
        "",
        "Binary attack-vs-benign:",
        json.dumps(metrics["attack_vs_benign"], indent=2),
        "",
        "Caution: CIC features failed_connections and unique_destinations are",
        "unavailable and median-imputed. Results are an external benchmark only.",
        "",
        f"Output directory: {OUT_DIR}",
    ]

    summary_text = "\n".join(summary)
    (OUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print("\nSaved evidence:")
    print(f"  {OUT_DIR / 'confusion_matrix.png'}")
    print(f"  {OUT_DIR / 'classification_report.txt'}")
    print(f"  {OUT_DIR / 'metrics.json'}")
    print(f"  {predictions_path}")
    print(f"  {OUT_DIR / 'summary.txt'}")


def norm_counts(values):
    return Counter(norm(v) for v in values.dropna())


if __name__ == "__main__":
    main()
