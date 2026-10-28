from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ReportProblem(BaseModel):
    title: str = Field(..., description="Problem title")
    explanation: str = Field(..., description="Problem explanation")


class ReportRemedy(BaseModel):
    immediate: str = Field(..., description="Immediate remedy")
    short_term: str = Field(..., description="Short-term remedy")
    long_term: str = Field(..., description="Long-term remedy")


class ReportIncident(BaseModel):
    severity: str = Field(..., description="Severity: critical, high, medium, low")
    score: float = Field(..., description="Incident score 0-1")
    entity: str = Field(..., description="Entity (IP, host, principal)")
    tactics: List[str] = Field(default_factory=list, description="MITRE tactics")
    lanes: List[str] = Field(default_factory=list, description="Lanes involved")
    flags: List[str] = Field(default_factory=list, description="Incident flags")
    problems: List[ReportProblem] = Field(default_factory=list, description="LLM-generated problems")
    remedies: ReportRemedy = Field(default_factory=ReportRemedy, description="LLM-generated remedies")


class ReportJSON(BaseModel):
    executive_summary: str = Field(..., description="Executive summary of findings")
    incidents: List[ReportIncident] = Field(default_factory=list, description="List of incidents")
    data_source_note: Optional[str] = Field(
        default=None, description="Note if data is synthetic"
    )
    used_fallback: bool = Field(default=False, description="Whether fallback was used")
    llm_provider: str = Field(default="", description="LLM provider name")
    llm_model: str = Field(default="", description="LLM model name")
    prompt_version: str = Field(default="1", description="Prompt version")
