from app.db.models import IncidentEvidence, Incidents


def save_incident(session, run_id, result):
    incident = Incidents(
        run_id=run_id,
        **{
            key: result[key]
            for key in ("severity", "score", "entity", "tactics", "lanes", "flags")
        },
    )
    session.add(incident)
    session.flush()
    for item in result["evidence"]:
        session.add(
            IncidentEvidence(
                incident_id=incident.incident_id,
                **{
                    key: item[key]
                    for key in (
                        "event_id",
                        "predicted_class",
                        "confidence",
                        "unified_class",
                        "tactic",
                    )
                },
            )
        )


def get_incidents_by_run(session, run_id):
    return session.query(Incidents).filter_by(run_id=run_id).all()
