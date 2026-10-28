from __future__ import annotations

from app.orchestration.correlator import (
    correlate_rules,
)


def build_incidents(run_id: str, db_session) -> None:
    """Build incidents for a run when all jobs are done.

    1. Get all predictions for the run
    2. Apply correlation rules R1-R5
    3. Write incidents and incident_evidence to DB
    """
    # Get predictions for this run - simplified query
    # In full implementation, would query DB for all predictions of this run
    from app.db.repositories.predictions import get_predictions_by_event as get_preds

    all_predictions = get_preds(db_session, run_id)  # simplified

    if not all_predictions:
        return

    # Apply correlation rules
    rule_results = correlate_rules(all_predictions, run_id, db_session)

    # Write incidents to DB
    from app.db.repositories.incidents import upsert_incident

    for rule_result in rule_results:
        # Create incident
        severity = rule_result.get("severity", "low")
        score = rule_result.get("score", 0)
        incident = {
            "incident_id": rule_result.get("incident_id"),
            "run_id": run_id,
            "severity": severity,
            "score": score,
            "entity": rule_result.get("entity", "unknown"),
            "tactics": json.dumps(rule_result.get("tactics", [])),
            "lanes": json.dumps(rule_result.get("lanes", [])),
            "flags": json.dumps(rule_result.get("flags", [])),
        }

        upsert_incident(db_session, incident)

        # Write incident evidence - capped at 200 per incident
        evidence = rule_result.get("evidence", [])
        from app.db.repositories.incident_evidence import upsert_evidence

        for ev in evidence[:200]:
            evidence_row = {
                "evidence_id": f"ev_{rule_result.get('incident_id')}_{ev.get('id', 0)}",
                "incident_id": incident["incident_id"],
                "event_id": ev.get("event_id"),
                "predicted_class": ev.get("predicted_class"),
                "confidence": ev.get("confidence"),
                "unified_class": ev.get("unified_class"),
                "tactic": ev.get("tactic"),
            }
            upsert_evidence(db_session, evidence_row)
