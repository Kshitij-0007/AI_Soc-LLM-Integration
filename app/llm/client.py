from __future__ import annotations

from typing import Any

import httpx


class LLMClient:
    """OpenAI-compatible LLM client via httpx."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def call(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> dict[str, Any]:
        """Call the LLM and return the response JSON."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def fallback_generate(self, prompt_text: str, template: dict[str, Any]) -> dict[str, Any]:
        """Generate remedies using a template when LLM fails.

        Returns dict with: executive_summary, problems, remedies
        """
        problems = []
        for p in template.get("problems", []):
            problems.append({
                "title": p.get("title", "Issue"),
                "explanation": p.get("explanation", "See incident details"),
            })

        remedies = []
        for r in template.get("remedies", []):
            remedies.append({
                "immediate": r.get("immediate", ""),
                "short_term": r.get("short_term", ""),
                "long_term": r.get("long_term", ""),
            })

        return {
            "executive_summary": prompt_text[:200] + "...",
            "problems": problems,
            "remedies": remedies,
        }
