
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from app.core.lane_spec import LaneSpec
from app.core.validator import validate
from app.lanes.netflow.parser import parse


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RAW_DATA = (
    PROJECT_ROOT / "data" / "raw" / "netflow" / "netflow.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "netflow"
)

SPEC_PATH = (
    PROJECT_ROOT / "app" / "lanes" / "netflow" / "spec.yaml"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_LABELS = {"benign", "malicious"}


def load_lane_spec() -> LaneSpec:
    """Load the existing NetFlow lane configuration."""

    if not SPEC_PATH.is_file():
        raise FileNotFoundError(
            f"NetFlow specification not found: {SPEC_PATH}"
        )

    with SPEC_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return LaneSpec(**config)


def prepare_datasets(
    raw_path: Path,
    output_dir: Path,
) -> None:
    """Validate raw NetFlow data and create stratified datasets."""

    print("=" * 60)
    print("NETFLOW DATA PREPARATION")
    print("=" * 60)

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path.resolve()}"
        )

    # 1. Ingest the raw CSV through the existing parser.
    print("\n[1/5] Ingesting raw NetFlow data...")

    raw_df = parse(raw_path, "csv")

    print(f"      Raw rows: {len(raw_df)}")
    print(f"      Raw columns: {len(raw_df.columns)}")

    if raw_df.empty:
        raise ValueError("The raw dataset is empty.")

    # 2. Validate against the existing lane specification.
    print("\n[2/5] Validating the dataset...")

    spec = load_lane_spec()
    result = validate(raw_df, spec)

    if result.status == "rejected":
        raise ValueError(
            "Dataset validation failed. "
            f"Missing required columns: {result.missing_required}"
        )

    df = result.clean_df.copy()

    print(f"      Validation status: {result.status}")
    print(f"      Valid rows: {result.rows_valid}")
    print(f"      Rejected rows: {result.rows_rejected}")

    if df.empty:
        raise ValueError("No rows remain after validation.")

    label_column = spec.label_column

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' is missing."
        )

    # 3. Normalize and validate binary classification labels.
    print("\n[3/5] Checking classification labels...")

    labels = (
        df[label_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if labels.isna().any() or labels.eq("").any():
        raise ValueError(
            "Missing or empty target labels were found."
        )

    actual_labels = set(labels.unique())
    unexpected = actual_labels - EXPECTED_LABELS

    if unexpected:
        raise ValueError(
            f"Unexpected labels: {sorted(unexpected)}. "
            f"Expected: {sorted(EXPECTED_LABELS)}"
        )

    if actual_labels != EXPECTED_LABELS:
        raise ValueError(
            "Both benign and malicious samples are required."
        )

    df[label_column] = labels

    print("      Class distribution:")
    print(df[label_column].value_counts().to_string())

    # Each row keeps its original identity during splitting.
    # Stratification preserves the class ratio in both partitions.
    train_indices, test_indices = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[label_column],
    )

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    if set(train_indices).intersection(set(test_indices)):
        raise RuntimeError(
            "Data leakage detected: a row appears in both partitions."
        )

    if len(train_df) + len(test_df) != len(df):
        raise RuntimeError(
            "Train/test row counts do not match validated row count."
        )

    # 4. Save the datasets without changing the raw source.
    print("\n[4/5] Saving train/test datasets...")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 5. Print the split summary.
    print("\n[5/5] Split verification")
    print("-" * 60)
    print(f"Total validated rows : {len(df)}")
    print(f"Training rows        : {len(train_df)}")
    print(f"Testing rows         : {len(test_df)}")
    print(f"Training features CSV: {train_path}")
    print(f"Testing features CSV : {test_path}")

    print("\nTraining class distribution:")
    print(train_df[label_column].value_counts().to_string())

    print("\nTesting class distribution:")
    print(test_df[label_column].value_counts().to_string())

    print("\nSUCCESS: Train/test datasets created.")
    print("The original raw CSV was not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare stratified NetFlow train/test datasets."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_RAW_DATA,
        help="Path to the raw NetFlow CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed train/test CSV files.",
    )

    args = parser.parse_args()

    prepare_datasets(
        raw_path=args.data.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
