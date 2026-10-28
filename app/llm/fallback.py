from __future__ import annotations

from typing import Any

from app.llm.schema import ReportIncident, ReportJSON, ReportRemedy


def get_remedies() -> dict[str, Any]:
    """Get remedy templates from knowledge/attack_snippets.yaml.

    Returns a dict with problem/remedy templates for each unified class.
    """
    import yaml

    with open("knowledge/attack_snippets.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def build_fallback_report(incidents_data: list[dict[str, Any]]) -> ReportJSON:
    """Build a report JSON using fallback templates when LLM is unavailable.

    For each incident, use the template remedies if available,
    otherwise use generic remedies.
    """
    templates = get_remedies()

    problems = []
    remedies_list = []

    for inc in incidents_data:
        uclass = inc.get("unified_class", "Unknown")
        template = templates.get(uclass, {})

        if template:
            problem = template.get("description", f"Security incident: {uclass}")
            problem_entry = {
                "title": uclass,
                "explanation": problem,
            }
            remedies = template.get("remedies", [])
            remedy_entry = {
                "immediate": remedies.get("immediate", "Monitor the situation"),
                "short_term": remedies.get("short_term", "Implement additional controls"),
                "long_term": remedies.get("long_term", "Review and improve security posture"),
            }
        else:
            problem_entry = {
                "title": uclass,
                "explanation": f"Security incident: {uclass} - no template available",
            }
            remedy_entry = {
                "immediate": "Isolate affected systems",
                "short_term": "Review security logs and policies",
                "long_term": "Implement comprehensive security program",
            }

        problems.append(problem_entry)
        remedies_list.append(remedy_entry)

    report = ReportJSON(
        executive_summary="Threat analysis report generated using fallback templates",
        incidents=[
            ReportIncident(
                severity=inc.get("severity", "low"),
                score=inc.get("score", 0.0),
                entity=inc.get("entity", "unknown"),
                tactics=inc.get("tactics", []),
                lanes=inc.get("lanes", []),
                flags=inc.get("flags", []),
                problems=problems,
                remedies=ReportRemedy(**remedy_entry),
            )
            for inc, remedy_entry in zip(incidents_data, remedies_list, strict=False)
        ],
        used_fallback=True,
    )

    return report
