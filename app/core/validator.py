from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.contracts import ValidationResult
from app.core.lane_spec import LaneSpec


def validate(df: pd.DataFrame, spec: LaneSpec) -> ValidationResult:
    """Validate a DataFrame against a LaneSpec.

    Steps:
    1. Normalize column names (strip whitespace, apply aliases)
    2. Check required columns
    3. Coerce optional columns with defaults
    4. Handle unknown columns
    5. Return ValidationResult with clean DataFrame and report
    """
    report: dict[str, Any] = {
        "status": "ok",
        "rows_total": len(df),
        "rows_valid": 0,
        "rows_rejected": 0,
        "missing_required": [],
        "renamed": {},
        "coerced": {},
        "unknown_columns": [],
    }

    # Step 1: Normalize column names - strip whitespace
    df_columns_stripped = df.columns.str.strip().tolist()
    df = df.copy()
    df.columns = df_columns_stripped

    # Map original names to stripped names
    name_map = dict(zip(df.columns, df_columns_stripped, strict=False))

    # Step 2: Apply aliases - build a mapping from alias -> canonical name
    alias_map: dict[str, str] = {}
    for col_spec in spec.columns.required + spec.columns.optional:
        for alias in col_spec.aliases:
            alias_map[alias.strip().lower()] = col_spec.name.strip().lower()

    # Rename columns based on aliases
    renamed_columns: dict[str, str] = {}
    new_cols = []
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in alias_map:
            canonical = alias_map[col_lower]
            if col != canonical:
                renamed_columns[col] = canonical
                new_cols.append(canonical)
            else:
                new_cols.append(col)
        else:
            new_cols.append(col)

    df.columns = new_cols
    report["renamed"] = {
        name_map.get(k, k): v for k, v in renamed_columns.items()
    }

    # Step 3: Check required columns
    missing_required: list[str] = []
    for req in spec.columns.required:
        col_name = req.name.strip().lower()
        if col_name not in [c.strip().lower() for c in df.columns]:
            # Also try aliases
            found_alias = False
            for alias in req.aliases:
                if alias.strip().lower() in [c.strip().lower() for c in df.columns]:
                    found_alias = True
                    break
            if not found_alias:
                missing_required.append(req.name)

    report["missing_required"] = missing_required

    if missing_required:
        report["status"] = "rejected"
        report["rows_valid"] = 0
        report["rows_rejected"] = report["rows_total"]
        return ValidationResult(
            status="rejected",
            clean_df=df.iloc[:0],
            rows_total=report["rows_total"],
            rows_valid=report["rows_valid"],
            rows_rejected=report["rows_rejected"],
            validation_report=report,
            missing_required=report["missing_required"],
            renamed=report["renamed"],
            coerced=report["coerced"],
            unknown_columns=report["unknown_columns"],
        )

    # Step 4: Handle optional columns - fill defaults
    for opt in spec.columns.optional:
        col_name = opt.name.strip().lower()
        if col_name not in [c.strip().lower() for c in df.columns]:
            # Column missing - fill with default
            default_val = opt.default
            if default_val is None:
                default_val = "unknown"
            df[opt.name] = default_val
            report["coerced"][opt.name] = {
                "default_used": True,
                "default_value": default_val,
            }

    # Step 5: Handle unknown columns
    known_columns: set[str] = set()
    for rc in spec.columns.required:
        known_columns.add(rc.name.strip().lower())
    for oc in spec.columns.optional:
        known_columns.add(oc.name.strip().lower())

    unknown: list[str] = []
    for col in df.columns:
        if col.strip().lower() not in known_columns:
            unknown.append(col.strip())
    report["unknown_columns"] = unknown

    if unknown and spec.behavior.on_unknown_columns == "ignore":
        # Keep unknown columns only if STORE_RAW=true, otherwise drop
        if not getattr(spec, "store_raw", False):
            df = df.drop(columns=unknown)
            report["unknown_columns"] = []

    # Step 6: Build clean DataFrame
    report["rows_valid"] = len(df)
    report["rows_rejected"] = report["rows_total"] - len(df)

    # For now, the clean_df is the validated DataFrame
    # In full implementation, would also remove coerced columns etc.

    return ValidationResult(
        status=report["status"],
        clean_df=df,
        rows_total=report["rows_total"],
        rows_valid=report["rows_valid"],
        rows_rejected=report["rows_rejected"],
        validation_report=report,
        missing_required=report["missing_required"],
        renamed=report["renamed"],
        coerced=report["coerced"],
        unknown_columns=report["unknown_columns"],
    )
