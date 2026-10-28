from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypeVar

import pandas as pd

T = TypeVar("T")


@dataclass
class FeatureFrame:
    X: pd.DataFrame  # numeric only, one row per unit of analysis
    meta: pd.DataFrame  # same index: entity_type, entity_value, ip, target_value,
        # window_start, window_end, event_count
    y: Optional[pd.Series] = None  # native labels, only when label column exists


@dataclass
class UnifiedPrediction:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    unified_class: str | None = None
    tactic: str | None = None
    shap_top_features: list[dict] | None = None


@dataclass
class JobResult:
    job_id: str
    lane_id: str
    status: str  # "done", "failed", "queued", "running"
    rows_total: int
    rows_valid: int
    rows_rejected: int
    validation_report: dict
    error: str | None = None


@dataclass
class ValidationResult:
    status: str  # "ok", "warning", "rejected"
    clean_df: pd.DataFrame
    rows_total: int
    rows_valid: int
    rows_rejected: int
    validation_report: dict
    missing_required: list[str]
    renamed: dict[str, str]
    coerced: dict
    unknown_columns: list[str]


@dataclass
class Incident:
    incident_id: str
    run_id: str
    severity: str
    score: float
    entity: str
    tactics: list[str] = field(default_factory=list)
    lanes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class IncidentEvidence:
    evidence_id: str
    incident_id: str
    event_id: str
    predicted_class: str
    confidence: float
    unified_class: str | None = None
    tactic: str | None = None
