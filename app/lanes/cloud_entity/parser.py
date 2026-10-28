from pathlib import Path

import pandas as pd


def parse(path: Path, fmt: str = "csv") -> pd.DataFrame:
    """Raw file to DataFrame. No cleaning. Validation happens in core.

    For Cloud Entity logs:
    - Parse CSV or JSONL format
    - Strip whitespace from column names
    - Handle cloud-specific fields
    """
    if fmt == "csv":
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()
        return df
    elif fmt == "jsonl":
        df = pd.read_json(path, lines=True, dtype=str)
        df.columns = df.columns.str.strip()
        return df
    else:
        raise ValueError(f"Unsupported format: {fmt}")
