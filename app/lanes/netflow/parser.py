"""Read a flow export and convert it to the lane's canonical columns.

Supported inputs: CSV (canonical, CICFlowMeter 2017/2018, Zeek, nProbe
NetFlow v9, UNSW-NB15), JSON lines, and Zeek TSV logs (``#fields`` header).
Conversion between exporters lives in ``adapters.py``. Value cleaning and
required-column checks happen in ``app.core.validator``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.lanes.netflow.adapters import detect_format, normalise_header, to_canonical

CHUNK_ROWS = 200_000
_ENCODINGS = ("utf-8", "latin1")


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns
            fmt = detect_format(header)
            usecols = None
            if fmt is not None and fmt.columns is not None:
                # Reading only the needed columns keeps 1 GB CIC files in memory.
                usecols = [c for c in header if normalise_header(c) in fmt.columns]
            chunks = pd.read_csv(
                path,
                encoding=encoding,
                usecols=usecols,
                low_memory=False,
                chunksize=CHUNK_ROWS,
                na_values=["Infinity", "-Infinity", "inf", "-inf", "NaN", "-"],
                keep_default_na=True,
            )
            # Some exports (CIC-IDS2017 Thursday) end with blank rows.
            return pd.concat(
                (chunk.dropna(how="all") for chunk in chunks), ignore_index=True
            )
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not decode {path.name}: {last_error}")


def _read_zeek_tsv(path: Path) -> pd.DataFrame:
    fields: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                break
            if not line.startswith("#"):
                break
    if not fields:
        raise ValueError(f"{path.name} has no Zeek #fields header")
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=fields,
        na_values=["-", "(empty)"],
        low_memory=False,
    )


def _is_zeek_tsv(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().startswith("#separator")


def read_raw(path: Path, fmt: str = "csv") -> pd.DataFrame:
    """Read the file as-is, without converting columns."""
    fmt = fmt.lower()
    if fmt in {"jsonl", "json"}:
        return pd.read_json(path, lines=fmt == "jsonl")
    if fmt in {"csv", "log", "tsv", "txt"}:
        return _read_zeek_tsv(path) if _is_zeek_tsv(path) else _read_csv(path)
    raise ValueError(f"Unsupported format: {fmt}")


def parse(path: Path, fmt: str = "csv") -> pd.DataFrame:
    """Raw file to canonical DataFrame.

    ``df.attrs["source_format"]`` records which exporter was detected.
    """
    raw = read_raw(Path(path), fmt)
    converted, source_format = to_canonical(raw)
    converted.attrs["source_format"] = source_format
    return converted
