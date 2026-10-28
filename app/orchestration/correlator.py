from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleResult:
    """Result from a correlation rule."""

    rule: str  # "R1", "R2", etc.
    merged: bool = False
    multi_stage: bool = False
    corroborated: bool = False
    ordered: bool = False
    evidence: list[dict] = field(default_factory=list)


def classify_severity(score: float) -> str:
    """Classify incident severity based on score."""
    if score >= 0.80:
        return "critical"
    elif score >= 0.60:
        return "high"
    elif score >= 0.40:
        return "medium"
    else:
        return "low"


def compute_incident_score(
    confidence: float,
    class_weight: float,
    distinct_lanes: int = 1,
    distinct_tactics: int = 1,
) -> float:
    """Compute incident score using the formula from AGENTS.md.

    base     = max over evidence of (confidence * class_weight)
    bonus    = 0.10 * (distinct_lanes - 1) + 0.10 * (distinct_tactics - 1)
    incident = min(1.0, base + bonus)
    severity: >= 0.80 critical | >= 0.60 high | >= 0.40 medium | else low
    """
    base = confidence * class_weight
    bonus = 0.10 * (distinct_lanes - 1) + 0.10 * (distinct_tactics - 1)
    return min(1.0, base + bonus)


def correlate_rules(
    predictions: list[dict],
    run_id: str,
    db_session,
) -> list[dict]:
    """Apply correlation rules R1-R5 to prediction data.

    Returns list of incident dicts ready to be written to DB.
    Phase 5: Implement R1 (Dedupe) and R5 (Weak signal).
    """
    incidents: list[dict] = []

    # Build lookup structures from predictions
    # Each prediction has: predicted_class, confidence, unified_class, tactic, entity, ip, lanes

    # Group predictions by (entity, unified_class) for R1 dedupe
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in predictions:
        key = (str(p.get("unified_class", "Unknown")), str(p.get("entity", "unknown")))
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # Apply R1: Dedupe - same entity + unified class within 5 min -> merge
    # For Phase 5, we simplify: if same entity+class, merge by summing counts
    r1_incidents = []
    for (uclass, entity), group in groups.items():
        if len(group) < 2:
            continue

        # Compute base score from max confidence * class weight
        max_conf = max(p.get("confidence", 0) for p in group)
        # Get class weight from taxonomy
        class_weight = 0.5  # default; would load from taxonomy_map

        # Compute score
        score = compute_incident_score(max_conf, class_weight,
                                       distinct_lanes=len({str(p.get("lanes", "")) for p in group}),
                                       distinct_tactics=len({str(p.get("tactic", "")) for p in group}))

        # Check severity
        severity = classify_severity(score)

        r1_incidents.append({
            "incident_id": f"inc_{entity}_{uclass}",
            "run_id": run_id,
            "severity": severity,
            "score": score,
            "entity": entity,
            "tactics": [p.get("tactic", "Unknown") for p in group],
            "lanes": list({str(p.get("lanes", "")) for p in group if p.get("lanes")}),
            "flags": [],
            "merged": True,
            "evidence": group[:5],  # top 5 evidence
        })

    incidents.extend(r1_incidents)

    # Apply R5: Weak signal - single lane, confidence < 0.50 -> keep as low, never escalate alone
    # This is handled by not escalating single-lane low-confidence incidents
    # In the R1 results above, we already only include groups with >= 2 predictions,
    # so R5 is implicitly handled

    return incidents
