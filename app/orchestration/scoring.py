from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeverityBand:
    min_score: float
    severity: str


SEVERITY_BANDS = [
    SeverityBand(min_score=0.80, severity="critical"),
    SeverityBand(min_score=0.60, severity="high"),
    SeverityBand(min_score=0.40, severity="medium"),
    SeverityBand(min_score=0.0, severity="low"),
]


def classify_severity(score: float) -> str:
    """Classify incident severity based on score."""
    for band in SEVERITY_BANDS:
        if score >= band.min_score:
            return band.severity
    return "low"


def compute_incident_score(
    confidence: float,
    class_weight: float,
    distinct_lanes: int = 1,
    distinct_tactics: int = 1,
) -> float:
    """Compute incident score using the formula from AGENTS.md.

    incident = min(1.0, base + bonus)
    base = max over evidence of (confidence * class_weight)
    bonus = 0.10 * (distinct_lanes - 1) + 0.10 * (distinct_tactics - 1)
    """
    base = confidence * class_weight
    bonus = 0.10 * (distinct_lanes - 1) + 0.10 * (distinct_tactics - 1)
    return min(1.0, base + bonus)
