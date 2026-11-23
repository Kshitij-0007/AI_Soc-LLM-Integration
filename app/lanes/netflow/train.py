"""Train the NetFlow lane model from labelled flow files.

    python -m app.lanes.netflow.train --data "data/raw/netflow/real/TrafficLabelling "
    python -m app.lanes.netflow.train --data <train files/dirs> \
        --test-data <files from another dataset>

Any exporter the parser understands can be mixed (CICFlowMeter, Zeek,
NetFlow v9, UNSW-NB15, canonical CSV). Files are converted, features are
computed per whole file (the source-context features need every flow of the
file), and only then are rows sampled per class.

Evaluation (headline numbers in metrics.json):

* ``--test-data`` given: the model is scored on those files only. Use a
  different dataset or network here; this is the number that predicts how
  the model behaves on new CSVs.
* otherwise: per file and class, the latest ``--holdout-share`` of flows by
  time are held out. This avoids the leak of a random row split, but it is
  still the same network, so expect lower scores on other networks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.core.contracts import FeatureFrame
from app.core.lane_spec import LaneSpec
from app.core.training import train_and_save
from app.core.validator import validate

LANE_DIR = Path(__file__).resolve().parent
LANE_ID = "netflow"
BENIGN = "BENIGN"
SEED = 42
DATA_SUFFIXES = {".csv", ".log", ".tsv", ".jsonl"}

# Exact names first (synthetic attack_type values), then substring rules.
EXACT_LABELS = {
    "benign": BENIGN,
    "normal": BENIGN,
    "background": BENIGN,
    "malicious_high_rate": "DoS",
    "malicious_low_and_slow": "Bot",
    "c2": "Bot",
}

AUGMENT_DROP_PROBABILITY = 0.3
AUGMENT_WEIGHT = 0.25
TIME_BLOCK = "5min"


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonical_label(value: object) -> str | None:
    """Map a native attack name from any supported dataset to a lane class.

    Returns None for classes the lane does not model (Infiltration,
    Heartbleed, Fuzzers, ...); those rows are excluded from training.
    """
    text = _normalise(value)
    if not text or text in {"nan", "none", "unknown"}:
        return None
    if text in EXACT_LABELS:
        return EXACT_LABELS[text]
    if text.startswith("benign"):
        return BENIGN
    if "ddos" in text:
        return "DDoS"
    if text.startswith("web_attack") or any(k in text for k in ("xss", "sql_inj")):
        return "Web Attack"
    if "brute" in text and "web" in text:
        return "Web Attack"
    if text.startswith("dos") or any(
        k in text for k in ("slowloris", "slowhttp", "hulk", "goldeneye")
    ):
        return "DoS"
    if "patator" in text or "brute" in text:
        return "Brute Force"
    if "scan" in text or text == "reconnaissance":
        return "PortScan"
    if text.startswith("bot") or "beacon" in text:
        return "Bot"
    return None


def canonical_labels(df: pd.DataFrame) -> pd.Series:
    """Lane class per row from attack_type, scenario or label (first match wins)."""
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in ("attack_type", "scenario", "label"):
        if column in df.columns:
            result = result.fillna(df[column].map(canonical_label))
    return result


def load_spec(lane_id: str = LANE_ID) -> LaneSpec:
    path = LANE_DIR.parent / lane_id / "spec.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return LaneSpec(**yaml.safe_load(handle))


def expand_paths(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in DATA_SUFFIXES)
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if not files:
        raise FileNotFoundError(f"No data files found in {paths}")
    return files


def check_capture_files(train_paths: list[Path], test_paths: list[Path]) -> None:
    """Reject identical input files even when copies have different names."""
    seen = {}
    for path in expand_paths(train_paths) + (
        expand_paths(test_paths) if test_paths else []
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            raise ValueError(
                f"Duplicate capture file: {path} duplicates {seen[digest]}"
            )
        seen[digest] = path


@dataclass
class LabeledSet:
    X: pd.DataFrame
    y: pd.Series
    timestamp: pd.Series
    source_file: pd.Series

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> LabeledSet:
        mask = np.asarray(mask)
        return LabeledSet(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            timestamp=self.timestamp.loc[mask].reset_index(drop=True),
            source_file=self.source_file.loc[mask].reset_index(drop=True),
        )

    def cap_per_class(self, limit: int | None, seed: int = SEED) -> LabeledSet:
        if limit is None or limit == 0:
            return self
        if limit < 0:
            raise ValueError("Row cap must be non-negative")
        positions = _cap_per_class(self.y, limit, seed)
        return self.subset(np.isin(np.arange(len(self)), positions))

    def frame(self) -> FeatureFrame:
        return FeatureFrame(X=self.X, meta=pd.DataFrame(index=self.X.index), y=self.y)

    def sample_training_rows(self, count: int, seed: int = SEED) -> LabeledSet:
        """Sample distinct labeled feature rows, sharing the budget across classes."""
        distinct = ~self.X.assign(_label=self.y).duplicated()
        labels = self.y.loc[distinct]
        if count < max(1, labels.nunique()) or count > len(labels):
            raise ValueError(
                f"Training row count must cover {labels.nunique()} classes "
                f"and fit within {len(labels)} distinct labeled examples."
            )
        shuffled = labels.sample(frac=1, random_state=seed)
        # Take one example per class each round until the budget is exhausted.
        rank = shuffled.groupby(shuffled).cumcount()
        chosen = rank.sort_values(kind="stable").index[:count]
        return self.subset(self.y.index.isin(chosen))


def _cap_per_class(labels: pd.Series, cap: int | None, seed: int) -> np.ndarray:
    """Row positions keeping at most ``cap`` random rows of each class."""
    if not cap:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for _, positions in labels.groupby(labels, sort=True).indices.items():
        if len(positions) > cap:
            positions = rng.choice(positions, size=cap, replace=False)
        keep.append(np.sort(positions))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def load_file_features(path: Path, spec: LaneSpec) -> tuple[LabeledSet, dict]:
    """Parse, validate and featurise one whole file; keep labelled rows."""
    from app.lanes.netflow.features import build_features
    from app.lanes.netflow.parser import parse

    df = parse(path, "jsonl" if path.suffix.lower() == ".jsonl" else "csv")
    source_format = df.attrs.get("source_format", "unknown")
    labels = canonical_labels(df)
    raw_labels = df["label"] if "label" in df.columns else pd.Series("unlabelled")
    dropped = raw_labels[labels.isna()].astype(str).value_counts().to_dict()
    df["label"] = labels.fillna("unknown").astype(str)

    validated = validate(df, spec)
    if validated.status == "rejected":
        raise ValueError(
            f"{path.name} ({source_format}): missing required columns "
            f"{validated.missing_required}"
        )
    clean = validated.clean_df.reset_index(drop=True)
    clean_labels = clean["label"].astype(str)
    # Features use every flow (context needs them); labels are kept apart.
    features = build_features(clean.assign(label="unknown"), spec)

    keep = (clean_labels != "unknown").to_numpy()
    info = {
        "file": path.name,
        "format": source_format,
        "rows": int(len(df)),
        "labelled_rows": int(keep.sum()),
        "dropped_labels": dropped,
    }
    labelled = LabeledSet(
        X=features.X.loc[keep].reset_index(drop=True),
        y=clean_labels.loc[keep].reset_index(drop=True),
        timestamp=pd.to_datetime(
            features.meta.loc[keep, "window_start"], errors="coerce"
        ).reset_index(drop=True),
        source_file=pd.Series(str(path.resolve()), index=range(int(keep.sum()))),
    )
    return labelled, info


def load_labeled_features(
    paths: list[Path],
    spec: LaneSpec,
    max_per_class: int | None = None,
    seed: int = SEED,
    verbose: bool = True,
) -> LabeledSet:
    """Featurise every file, cap rows per class per file, then cap globally."""
    parts = []
    for path in expand_paths(paths):
        labelled, info = load_file_features(path, spec)
        labelled = labelled.cap_per_class(max_per_class, seed)
        if verbose:
            counts = ", ".join(f"{k}={v}" for k, v in labelled.y.value_counts().items())
            dropped = info["dropped_labels"]
            print(
                f"      {info['file']} [{info['format']}]: {info['rows']:,} rows -> "
                f"{len(labelled):,} kept ({counts})"
                + (f"; not modelled: {dropped}" if dropped else ""),
                flush=True,
            )
        parts.append(labelled)
    combined = LabeledSet(
        X=pd.concat([p.X for p in parts], ignore_index=True),
        y=pd.concat([p.y for p in parts], ignore_index=True),
        timestamp=pd.concat([p.timestamp for p in parts], ignore_index=True),
        source_file=pd.concat([p.source_file for p in parts], ignore_index=True),
    )
    return combined.cap_per_class(max_per_class, seed)


def temporal_holdout(data: LabeledSet, share: float, seed: int = SEED) -> np.ndarray:
    """True for the latest ``share`` of each (file, class) by timestamp.

    Rows without a timestamp are held out at random with the same share.
    """
    rng = np.random.default_rng(seed)
    order = data.timestamp.rank(method="first")
    random_rank = pd.Series(rng.random(len(data)), index=data.y.index)
    order = order.fillna(random_rank * order.max() if order.notna().any() else 0)
    key = data.source_file + "|" + data.y
    percentile = order.groupby(key).rank(pct=True)
    return (percentile > 1.0 - share).to_numpy()


def time_block_groups(data: LabeledSet) -> np.ndarray:
    """Group id per (file, 5-minute block) so CV never splits a capture window."""
    block = data.timestamp.dt.floor(TIME_BLOCK).astype(str)
    no_time = data.timestamp.isna()
    chunk = (pd.Series(np.arange(len(data))) // 500).astype(str)
    block[no_time] = "row" + chunk[no_time]
    return pd.factorize(data.source_file + "|" + block)[0]


def augment_missing_groups(
    X: pd.DataFrame,
    groups: np.ndarray,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Append one low-weight copy of every row with random feature groups blanked.

    Teaches the model to cope with exporters that lack flags, timestamps/IPs
    (context) or ports. Returns augmented X, weights, CV groups and a mask of
    original rows.
    """
    from app.lanes.netflow.features import OPTIONAL_FEATURE_GROUPS

    rng = np.random.default_rng(seed)
    copy = X.copy()
    for columns in OPTIONAL_FEATURE_GROUPS.values():
        mask = rng.random(len(copy)) < AUGMENT_DROP_PROBABILITY
        copy.loc[mask, columns] = np.nan
    augmented = pd.concat([X, copy], ignore_index=True)
    weights = np.concatenate([np.ones(len(X)), np.full(len(copy), AUGMENT_WEIGHT)])
    cv_groups = np.concatenate([groups, groups])
    original = np.concatenate([np.ones(len(X), bool), np.zeros(len(copy), bool)])
    return augmented, weights, cv_groups, original


def main(
    data_paths: list[Path],
    test_paths: list[Path] | None = None,
    version: str = "2.0.0",
    max_per_class: int | None = 30_000,
    holdout_share: float = 0.2,
    augment: bool = True,
    cv_folds: int = 3,
    data_source: str | None = None,
    training_rows: int | None = None,
) -> dict:
    from app.lanes.netflow.features import FEATURE_SET_VERSION

    check_capture_files(data_paths, test_paths or [])
    spec = load_spec()
    print(f"[1/4] Loading training data from {len(data_paths)} path(s)...", flush=True)
    data = load_labeled_features(data_paths, spec, max_per_class)
    if not len(data):
        raise ValueError("No labelled rows to train on.")

    if test_paths:
        print(f"[2/4] Loading test data from {len(test_paths)} path(s)...", flush=True)
        test = load_labeled_features(test_paths, spec, None)
        train = data
        eval_description = "separate test dataset: " + ", ".join(
            sorted(test.source_file.unique())
        )
    else:
        print(
            f"[2/4] Holding out the latest {holdout_share:.0%} of each file/class "
            "by time...",
            flush=True,
        )
        held = temporal_holdout(data, holdout_share)
        train, test = data.subset(~held), data.subset(held)
        eval_description = (
            f"temporal holdout: latest {holdout_share:.0%} of flows per file and "
            "class (same network as training)"
        )

    if training_rows is not None:
        if augment:
            raise ValueError("Use --no-augment with an exact training row count.")
        train = train.sample_training_rows(training_rows)

    print(
        f"      train {len(train):,} rows ({train.y.value_counts().to_dict()})\n"
        f"      test  {len(test):,} rows ({test.y.value_counts().to_dict()})",
        flush=True,
    )

    X = train.X
    weights = np.ones(len(train))
    groups = time_block_groups(train)
    reference = np.ones(len(train), bool)
    y = train.y
    if augment:
        X, weights, groups, reference = augment_missing_groups(train.X, groups)
        y = pd.concat([train.y, train.y], ignore_index=True)

    print(f"[3/4] Training on {len(X):,} rows x {X.shape[1]} features...", flush=True)
    sources = data_source or ", ".join(sorted(train.source_file.unique()))
    notes = [f"NetFlow feature set {FEATURE_SET_VERSION}"]
    if max_per_class:
        notes.append(f"Rows capped at {max_per_class} per class")
    if training_rows is not None:
        notes.append(
            f"{training_rows} distinct labeled examples, class-balanced sampling, "
            f"seed {SEED}, no augmentation"
        )
    result = train_and_save(
        FeatureFrame(X=X, meta=pd.DataFrame(index=X.index), y=y),
        lane_id=LANE_ID,
        version=version,
        sample_weight=weights,
        groups=groups,
        eval_frame=test.frame(),
        eval_description=eval_description,
        data_source=sources,
        feature_set_version=FEATURE_SET_VERSION,
        reference_rows=reference,
        n_jobs=os.cpu_count(),
        cv_folds=cv_folds,
        notes=notes,
    )

    metrics = result["metrics"]
    print("[4/4] Done.", flush=True)
    print(f"      Saved to {result['artifact_dir']}")
    print(f"      Evaluation: {result['evaluation']}")
    print(
        f"      Macro-F1: {metrics['macro_f1']:.4f}  "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )
    if metrics.get("attack_vs_benign"):
        avb = metrics["attack_vs_benign"]
        print(
            f"      Attack detection rate: {avb['detection_rate']:.4f}  "
            f"False positive rate: {avb['false_positive_rate']:.4f}"
        )
    for label, scores in metrics["per_class"].items():
        print(
            f"        {label:12s} recall {scores['recall']:.3f}  "
            f"precision {scores['precision']:.3f}  n={scores['support']}"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NetFlow lane model.")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Labelled training files or folders (any supported exporter).",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        nargs="+",
        default=None,
        help="Labelled files from a different dataset/network for evaluation.",
    )
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=30_000,
        help="Row cap per class (per file and overall). 0 = no cap.",
    )
    parser.add_argument("--holdout-share", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3, help="0 skips CV.")
    parser.add_argument(
        "--training-rows", type=int, help="Exact fit size. Requires --no-augment."
    )
    parser.add_argument("--data-source", default=None)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable missing-feature-group augmentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_paths=args.data,
        test_paths=args.test_data,
        version=args.version,
        max_per_class=args.max_per_class or None,
        holdout_share=args.holdout_share,
        augment=not args.no_augment,
        cv_folds=args.cv_folds,
        data_source=args.data_source,
        training_rows=args.training_rows,
    )
