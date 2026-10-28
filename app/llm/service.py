from __future__ import annotations

import json
import time
from typing import Any

from app.knowledge.attack_snippets import get_remedies as get_template_remedies

from app.llm.client import LLMClient
from app.llm.schema import ReportJSON


class LLMService:
    """LLM service with retry and fallback."""

    def __init__(self, client: LLMClient, prompt_version: str = "1"):
        self.client = client
        self.prompt_version = prompt_version

    def _build_prompt(self, incidents_data: list[dict[str, Any]]) -> str:
        """Build the prompt for the LLM from incident data."""
        # Build incident summaries for the prompt
        incident_summaries = []
        for inc in incidents_data:
            summary = f"""
Incident:
- Severity: {inc.get('severity', 'unknown')}
- Entity: {inc.get('entity', 'unknown')}
- Score: {inc.get('score', 0):.2f}
- Tactics: {', '.join(inc.get('tactics', []))}
- Lanes: {', '.join(inc.get('lanes', []))}
- Flags: {', '.join(inc.get('flags', []))}
"""
            incident_summaries.append(summary.strip())

        prompt = f"""
You are a security analyst. Write a threat analysis report based on the following incidents.

For each incident, provide:
- A title (short, descriptive)
- An explanation of the problem
- Immediate remedy
- Short-term remedy
- Long-term remedy

Copy the severity, score, entity, tactics and lanes from the database - do not derive these from the LLM.

Incidents:
{''.join(incident_summaries)}

Write the output as validated JSON only, matching this schema:
- executive_summary: brief summary of all incidents
- incidents: array of objects with:
  - problems: array of {title, explanation}
  - remedies: {immediate, short_term, long_term}

Return ONLY valid JSON. No explanatory text outside the JSON.
"""
        return prompt

    def call(
        self,
        incidents_data: list[dict[str, Any]],
        timeout: int = 120,
    ) -> tuple[Optional[ReportJSON], bool]:
        """Call the LLM with retry and fallback.

        Returns (report_json, used_fallback) tuple.
        """
        prompt = self._build_prompt(incidents_data)

        # First attempt
        try:
            response = self.client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            # Extract the JSON from the response
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in LLM response")
            text = choices[0].get("message", {}).get("content", "{}")
            # Try to parse JSON from the response text
            json_text = text.strip()
            if json_text.startswith("```"):
                # Remove code block markers
                json_text = json_text.split("\n", 2)[2] if "\n" in json_text else json_text
                json_text = json_text.rstrip("`").strip()
            data = json.loads(json_text)

            # Validate with Pydantic
            report = ReportJSON(**data)
            return report, False

        except Exception as e:
            print(f"LLM call failed: {e}")
            # Retry once
            try:
                time.sleep(1)
                response = self.client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                choices = response.get("choices", [])
                if not choices:
                    raise ValueError("No choices in LLM response (2nd attempt)")
                text = choices[0].get("message", {}).get("content", "{}")
                json_text = text.strip()
                if json_text.startswith("```"):
                    json_text = json_text.split("\n", 2)[2] if "\n" in json_text else json_text
                    json_text = json_text.rstrip("`").strip()
                data = json.loads(json_text)
                report = ReportJSON(**data)
                return report, False
            except Exception as e2:
                print(f"LLM retry also failed: {e2}")
                # Use fallback from attack snippets
                get_template_remedies()
                # Build a basic report from the template
                fallback_report = ReportJSON(
                    executive_summary="Threat analysis report generated using fallback templates",
                    used_fallback=True,
                    llm_provider=self.client.base_url,
                    llm_model=self.client.model,
                    prompt_version=self.prompt_version,
                )
                return fallback_report, True

    def generate_report(
        self,
        run_id: str,
        incidents_data: list[dict[str, Any]],
        db_session,
    ) -> tuple[ReportJSON, bool]:
        """Generate a full report JSON from incident data.

        Returns (report_json, used_fallback) tuple.
        """
        report, used_fallback = self.call(incidents_data)
        return report, used_fallback
