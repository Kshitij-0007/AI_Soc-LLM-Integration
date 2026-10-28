from pathlib import Path

import pandas as pd


def parse(path: Path, fmt: str = "csv") -> pd.DataFrame:
    """Raw file to DataFrame. No cleaning. Validation happens in core.

    For CSV files (CIC-IDS2017 format):
    - Strip header whitespace
    - Drop duplicated columns
    - Replace inf with NaN
    """
    if fmt == "csv":
        df = pd.read_csv(path, dtype=str)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Drop duplicated columns (CIC-IDS2017 has Fwd Header Length duplicated)
        df = df.loc[:,~df.columns.duplicated()]

        # Replace inf with NaN
        df = df.replace("inf", pd.NA)
        df = df.replace(float("inf"), pd.NA)

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except (ValueError, TypeError):
                pass

        return df
    else:
        raise ValueError(f"Unsupported format: {fmt}")
