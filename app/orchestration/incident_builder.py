from __future__ import annotations

from dataclasses import dataclass
from typing import Any, dict, list


@dataclass
class IncidentBuilder:
    """Builds incidents and incident_evidence rows from prediction data."""

    db_session: Any
    run_id: str

    def build(self, predictions: list[dict]) -> list[dict]:
        """Build incidents from prediction list.

        Returns list of incident dicts ready for DB insertion.
        """
        # Apply correlation rules
        from app.orchestration.correlator import correlate_rules
        rule_results = correlate_rules(predictions, self.run_id, self.db_session)

        incidents = []
        for result in rule_results:
            incident = self._build_incident(result)
            if incident:
                incidents.append(incident)

        return incidents

    def _build_incident(self, rule_result: dict) -> dict | None:
        """Build a single incident from a rule result."""
        if not rule_result.get("merged"):
            return None

        score = rule_result.get("score", 0)
        severity = classify_severity(score)

        incident = {
            "incident_id": rule_result.get("incident_id"),
            "run_id": self.run_id,
            "severity": severity,
            "score": score,
            "entity": rule_result.get("entity", "unknown"),
            "tactics": rule_result.get("evidence", []),
            "lanes": rule_result.get("lanes", []),
            "flags": rule_result.get("flags", []),
        }
        return incident
