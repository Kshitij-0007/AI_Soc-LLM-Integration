from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ColumnSpec(BaseModel):
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="SQL type")
    aliases: list[str] = Field(default_factory=list, description="Alternative column names")
    default: object = Field(default=None, description="Default value if missing")


class LaneBehavior(BaseModel):
    on_missing_required: str = Field(
        default="reject_job", description="Action on missing required columns"
    )
    on_missing_optional: str = Field(
        default="fill_default", description="Action on missing optional columns"
    )
    on_unknown_columns: str = Field(
        default="ignore", description="Action on unknown columns"
    )


class ColumnSpecs(BaseModel):
    required: list[ColumnSpec] = Field(
        default_factory=list, description="Required column definitions"
    )
    optional: list[ColumnSpec] = Field(
        default_factory=list, description="Optional column definitions"
    )

    def get_by_name(self, name: str) -> ColumnSpec | None:
        for c in self.required:
            if c.name == name:
                return c
        for c in self.optional:
            if c.name == name:
                return c
        return None


class LaneSpec(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANE_")

    lane_id: str = Field(..., description="Unique lane identifier")
    display_name: str = Field(..., description="Human-readable name")
    version: str = Field(default="1", description="Lane version string")
    formats: list[str] = Field(default=["csv"], description="Supported file formats")
    unit_of_analysis: str = Field(
        ..., description="Unit of analysis (flow | host_window_5m | principal_window_15m)"
    )
    entity: dict = Field(
        ..., description="Entity definition {type, field}"
    )
    ip_field: str = Field(
        default="", description="Optional IP field for cross-lane matching"
    )
    time_field: str = Field(..., description="Timestamp column name")
    label_column: str = Field(
        default="", description="Label column name (training only)"
    )
    columns: ColumnSpecs = Field(
        ..., description="Required and optional column definitions"
    )
    behavior: LaneBehavior = Field(
        ..., description="Behavior on missing/unknown columns"
    )

    def get_alias(self, name: str) -> str | None:
        """Try to find a column by name or its aliases."""
        # Check direct match
        cols = self.columns.required + self.columns.optional
        for col in cols:
            if col.name == name:
                return name
            for alias in col.aliases:
                if alias == name:
                    return col.name
        return None
