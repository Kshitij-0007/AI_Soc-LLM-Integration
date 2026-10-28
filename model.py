
from __future__ import annotations
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "netflow" / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "netflow" / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "netflow" / "demo"
)

TARGET_COLUMN = "label"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "src_port",
    "dst_port",
    "flow_duration_ms",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_packets",
    "total_bytes",
    "packets_per_second",
    "bytes_per_second",
    "failed_connections",
    "unique_destinations",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_rst_count",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset and validate its required columns."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing target labels."
        )

    return df


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert selected NetFlow fields to numeric features."""

    X = df[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    # Columns that contain no numeric training values cannot
    # provide useful information to this demonstration model.
    return X


def evaluate_edge_cases():
    import json

    challenge_path = (
        PROJECT_ROOT
    / "data"
    / "raw"
    / "netflow"
    / "netflow_edge_case_challenge.csv"
    )
    model_path = (
        PROJECT_ROOT
        / "models"
        / "netflow"
        / "demo"
        / "netflow_xgboost.joblib"
    )
    output_dir = PROJECT_ROOT / "models" / "netflow" / "demo"

    if not challenge_path.exists():
        raise FileNotFoundError(
            f"Challenge dataset not found: {challenge_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run the normal training process first."
        )

    # Load the existing model without retraining it.
    artifact = joblib.load(model_path)
    challenge = pd.read_csv(challenge_path)

    feature_columns = artifact["feature_columns"]
    label_encoder = artifact["label_encoder"]

    missing = [
        col for col in feature_columns
        if col not in challenge.columns
    ]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    if "label" not in challenge.columns:
        raise ValueError("Challenge CSV must contain a label column.")

    X = challenge[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    y_true = challenge["label"].astype(str).str.lower()
    known_labels = set(label_encoder.classes_)
    unknown = sorted(set(y_true) - known_labels)

    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    predictions_encoded = artifact["model"].predict(X)
    y_pred = label_encoder.inverse_transform(
        predictions_encoded.astype(int)
    ).astype(str)

    probabilities = artifact["model"].predict_proba(X)
    malicious_index = list(label_encoder.classes_).index("malicious")
    malicious_probability = probabilities[:, malicious_index]

    results = challenge.copy()
    results["predicted_label"] = y_pred
    results["malicious_probability"] = malicious_probability
    results["correct"] = (
        y_true.to_numpy() == results["predicted_label"].str.lower().to_numpy()
    )

    labels = ["benign", "malicious"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "records": int(len(results)),
        "accuracy": float(results["correct"].mean()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "class_order": labels,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "edge_case_predictions.csv",
        index=False,
    )
    (output_dir / "edge_case_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n=== EDGE-CASE EVALUATION ===")
    print("Synthetic stress test; not a real-world benchmark.")
    print(f"Records: {len(results)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"False-negative rate: {metrics['false_negative_rate']:.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    if "scenario" in results.columns:
        print("\nResults by scenario:")
        print(
            results.groupby("scenario").agg(
                records=("correct", "size"),
                accuracy=("correct", "mean"),
                avg_malicious_probability=(
                    "malicious_probability", "mean"
                ),
            ).round(4).to_string()
        )

    print("\nPredictions saved to:")
    print(output_dir / "edge_case_predictions.csv")
    print("Metrics saved to:")
    print(output_dir / "edge_case_metrics.json")


def main() -> None:
    print("=" * 65)
    print("AI SOC - NETFLOW INTRUSION CLASSIFICATION")
    print("=" * 65)

    # 1. Load the already prepared datasets.
    print("\n[1/5] Loading training and testing data...")

    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    print(f"Training records: {len(train_df)}")
    print(f"Testing records : {len(test_df)}")

    # 2. Prepare features and labels.
    print("\n[2/5] Preparing model inputs...")

    X_train = build_features(train_df)
    X_test = build_features(test_df)

    y_train_text = (
        train_df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y_test_text = (
        test_df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {"benign", "malicious"}

    if set(y_train_text.unique()) != allowed_labels:
        raise ValueError(
            "Training data must contain benign and malicious labels."
        )

    if not set(y_test_text.unique()).issubset(allowed_labels):
        raise ValueError(
            "Testing data contains an unexpected label."
        )

    # Encode text labels into integers for XGBoost.
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)

    unknown_test_labels = (
        set(y_test_text.unique()) - set(label_encoder.classes_)
    )

    if unknown_test_labels:
        raise ValueError(
            f"Unknown test labels: {unknown_test_labels}"
        )

    y_test = label_encoder.transform(y_test_text)

    # 3. Train the classifier.
    print("\n[3/5] Training XGBoost classifier...")
    print("      Fitting imputation on training data only.")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("      Training completed successfully.")

    # 4. Evaluate against the held-out test dataset.
    print("\n[4/5] Evaluating on unseen test data...")

    y_pred = model.predict(X_test).astype(int)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    class_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    print("\n" + "=" * 65)
    print("TEST SET RESULTS")
    print("=" * 65)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print("Rows = actual classes; columns = predicted classes")
    print("Class order:", class_names)
    print(matrix)

    # 5. Save the model and evaluation artifacts separately.
    print("\n[5/5] Saving model and evaluation results...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "netflow_xgboost.joblib"
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    metrics = {
        "model": "XGBoost",
        "task": "binary classification",
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "features": FEATURE_COLUMNS,
        "classes": class_names,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "actual_label": y_test_text.to_numpy(),
            "predicted_label": label_encoder.inverse_transform(
                y_pred
            ),
            "malicious_probability": y_probability,
            "correct": y_test == y_pred,
        }
    )

    predictions.to_csv(predictions_path, index=False)

    print(f"Saved model      : {model_path}")
    print(f"Saved metrics    : {metrics_path}")
    print(f"Saved predictions: {predictions_path}")

    print("\nSample predictions:")
    print(predictions.head(10).to_string(index=False))

    print("\nDEMO COMPLETED SUCCESSFULLY")
    print(
        "Note: results describe this held-out sample, "
        "not guaranteed real-world SOC performance."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--edge-test":
        evaluate_edge_cases()
    else:
        main()