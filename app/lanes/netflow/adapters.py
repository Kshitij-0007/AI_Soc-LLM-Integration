"""Convert flow exports from different tools into the lane's canonical columns.

Every exporter names, scales and defines its fields differently. A model
trained on one exporter's raw columns silently breaks on another's, so each
supported format is converted here, once, into one canonical schema with
fixed units and meanings:

==================  ==========================================================
column              meaning
==================  ==========================================================
timestamp           flow start, pandas datetime (NaT when the export has none)
src_ip / dst_ip     strings, "unknown" when the export has none
src_port/dst_port   numbers
protocol            "tcp" | "udp" | "icmp" | other lowercase name
flow_duration_ms    milliseconds
total_fwd_packets   packets source -> destination
total_bwd_packets   packets destination -> source
total_fwd_bytes     IP-layer bytes (headers included) source -> destination
total_bwd_bytes     IP-layer bytes destination -> source
tcp_*_count         >0 when the flag was seen in the flow, 0 when not,
                    NaN when the exporter does not report flags
label               native attack name string (training files only)
==================  ==========================================================

Flag columns are only used as "seen / not seen" by features.py because
exporters disagree on how they count them (CICFlowMeter reports 0/1,
nfdump/NetFlow report a cumulative bitmask, Zeek reports a history string).

Adding a new exporter: write a ``_from_<name>`` converter, add a
``SourceFormat`` entry to ``FORMATS`` and add a test in
``tests/lanes/test_netflow_adapters.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flow_duration_ms",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_rst_count",
    "tcp_fin_count",
    "tcp_psh_count",
]
REQUIRED_COLUMNS = ["timestamp", "src_ip", "dst_ip", "dst_port", "flow_duration_ms"]
FLAG_COLUMNS = [
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_rst_count",
    "tcp_fin_count",
    "tcp_psh_count",
]

_PROTOCOL_NUMBERS = {1: "icmp", 6: "tcp", 17: "udp", 58: "icmp"}

# Minimum IP + transport header sizes, used only when an exporter reports
# payload bytes and no usable header length.
_HEADER_BYTES = {"tcp": 40, "udp": 28, "icmp": 28}
_DEFAULT_HEADER_BYTES = 40


def normalise_header(name: object) -> str:
    """Lowercase, trim and collapse whitespace so header lookups are stable."""
    return " ".join(str(name).replace("﻿", "").strip().lower().split())


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column by normalised name, or an all-NaN series when absent."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(_col(df, name), errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _text(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series("unknown", index=df.index, dtype=object)
    values = df[name].astype("string").str.strip()
    return values.fillna("unknown").replace("", "unknown").astype(object)


def _protocol(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    named = values.astype("string").str.strip().str.lower()
    mapped = numeric.map(_PROTOCOL_NUMBERS)
    result = mapped.fillna(named.where(numeric.isna()))
    return result.fillna("unknown").replace("", "unknown").astype(object)


def _ip_bytes(
    payload: pd.Series,
    header: pd.Series,
    packets: pd.Series,
    protocol: pd.Series,
) -> pd.Series:
    """Payload bytes plus header bytes.

    Uses the exporter's header length when it is plausible (CICFlowMeter
    writes 0 or negative values for some flows; an IP header alone is 20
    bytes per packet) and a per-protocol estimate otherwise.
    """
    estimate = packets * protocol.map(_HEADER_BYTES).fillna(_DEFAULT_HEADER_BYTES)
    header = header.where(header >= packets * 20)
    return payload + header.fillna(estimate)


def _parse_times(values: pd.Series, formats: list[str]) -> pd.Series:
    """Try explicit formats first (fast, unambiguous), then a mixed fallback."""
    text = values.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in formats:
        missing = result.isna() & text.notna()
        if not missing.any():
            break
        parsed = pd.to_datetime(text[missing], format=fmt, errors="coerce")
        result.loc[missing] = parsed
    missing = result.isna() & text.notna()
    if missing.any():
        parsed = pd.to_datetime(
            text[missing], format="mixed", dayfirst=True, errors="coerce"
        )
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
        result.loc[missing] = parsed
    return result


def _epoch_times(values: pd.Series) -> pd.Series:
    seconds = pd.to_numeric(values, errors="coerce")
    # Millisecond epochs are > 1e11; second epochs are ~1e9.
    seconds = seconds.where(seconds < 1e11, seconds / 1000.0)
    return pd.to_datetime(seconds, unit="s", errors="coerce")


def _flag_from_mask(mask: pd.Series, bit: int) -> pd.Series:
    numeric = pd.to_numeric(mask, errors="coerce")
    valid = numeric.notna()
    flags = pd.Series(np.nan, index=mask.index)
    flags[valid] = ((numeric[valid].astype("int64") & bit) > 0).astype(float)
    return flags


# ---------------------------------------------------------------------------
# Converters. Each receives a frame whose headers are already normalised.
# ---------------------------------------------------------------------------


def _from_canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = _parse_times(
            out["timestamp"], ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
        )
    if "protocol" in out.columns:
        out["protocol"] = _protocol(out["protocol"])
    return out


def _cic_working_hours(times: pd.Series) -> pd.Series:
    """CIC-IDS2017 writes afternoon times on a 12-hour clock without AM/PM.

    Captures ran roughly 08:00-18:00, so hours 1-7 are afternoon hours.
    """
    early = times.dt.hour.between(1, 7)
    return times.where(~early, times + pd.Timedelta(hours=12))


def _from_cicflowmeter(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 (long names) and CSE-CIC-IDS2018 (short names)."""

    def pick(*names: str) -> str:
        return next((n for n in names if n in df.columns), names[0])

    out = pd.DataFrame(index=df.index)
    times = _parse_times(
        _col(df, "timestamp"),
        [
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y %I:%M:%S %p",
            "%Y-%m-%d %H:%M:%S",
        ],
    )
    out["timestamp"] = _cic_working_hours(times) if "destination port" in df else times
    out["src_ip"] = _text(df, pick("source ip", "src ip"))
    out["dst_ip"] = _text(df, pick("destination ip", "dst ip"))
    out["src_port"] = _num(df, pick("source port", "src port"))
    out["dst_port"] = _num(df, pick("destination port", "dst port"))
    out["protocol"] = _protocol(_col(df, "protocol"))
    # CICFlowMeter durations are microseconds.
    out["flow_duration_ms"] = _num(df, "flow duration") / 1000.0

    fwd_packets = _num(df, pick("total fwd packets", "tot fwd pkts"))
    bwd_packets = _num(df, pick("total backward packets", "tot bwd pkts"))
    out["total_fwd_packets"] = fwd_packets
    out["total_bwd_packets"] = bwd_packets
    # CICFlowMeter "Total Length" is payload only; add header bytes so the
    # value matches exporters that report IP-layer bytes.
    out["total_fwd_bytes"] = _ip_bytes(
        _num(df, pick("total length of fwd packets", "totlen fwd pkts")),
        _num(df, pick("fwd header length", "fwd header len")),
        fwd_packets,
        out["protocol"],
    )
    out["total_bwd_bytes"] = _ip_bytes(
        _num(df, pick("total length of bwd packets", "totlen bwd pkts")),
        _num(df, pick("bwd header length", "bwd header len")),
        bwd_packets,
        out["protocol"],
    )
    for flag in ("syn", "ack", "rst", "fin", "psh"):
        out[f"tcp_{flag}_count"] = _num(
            df, pick(f"{flag} flag count", f"{flag} flag cnt")
        )
    if "label" in df.columns:
        out["label"] = _text(df, "label")
    return out


def _from_zeek_conn(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = _epoch_times(_col(df, "ts"))
    out["src_ip"] = _text(df, "id.orig_h")
    out["dst_ip"] = _text(df, "id.resp_h")
    out["src_port"] = _num(df, "id.orig_p")
    out["dst_port"] = _num(df, "id.resp_p")
    out["protocol"] = _protocol(_col(df, "proto"))
    out["flow_duration_ms"] = _num(df, "duration") * 1000.0
    out["total_fwd_packets"] = _num(df, "orig_pkts")
    out["total_bwd_packets"] = _num(df, "resp_pkts")
    out["total_fwd_bytes"] = _num(df, "orig_ip_bytes")
    out["total_bwd_bytes"] = _num(df, "resp_ip_bytes")

    # Zeek history letters: upper case = originator, lower case = responder.
    if "history" in df.columns:
        history = df["history"].astype("string").fillna("")
        tcp = out["protocol"] == "tcp"
        letters = {"syn": "Ss", "ack": "Aa", "rst": "Rr", "fin": "Ff", "psh": "Dd"}
        for flag, chars in letters.items():
            seen = history.str.contains(f"[{chars}]", regex=True).astype(float)
            out[f"tcp_{flag}_count"] = seen.where(tcp)
        # 'h' is the responder's SYN-ACK.
        synack = history.str.contains("h", regex=False).astype(float)
        out["tcp_syn_count"] = out["tcp_syn_count"].fillna(0).add(synack).where(tcp)
    else:
        for column in FLAG_COLUMNS:
            out[column] = np.nan
    label = "detailed-label" if "detailed-label" in df.columns else "label"
    if label in df.columns:
        out["label"] = _text(df, label)
    return out


def _from_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = _epoch_times(_col(df, "stime"))
    out["src_ip"] = _text(df, "srcip")
    out["dst_ip"] = _text(df, "dstip")
    out["src_port"] = _num(df, "sport")
    out["dst_port"] = _num(df, "dsport")
    out["protocol"] = _protocol(_col(df, "proto"))
    out["flow_duration_ms"] = _num(df, "dur") * 1000.0
    out["total_fwd_packets"] = _num(df, "spkts")
    out["total_bwd_packets"] = _num(df, "dpkts")
    out["total_fwd_bytes"] = _num(df, "sbytes")
    out["total_bwd_bytes"] = _num(df, "dbytes")
    for column in FLAG_COLUMNS:
        out[column] = np.nan
    if "attack_cat" in df.columns:
        category = _text(df, "attack_cat")
        out["label"] = category.where(category != "unknown", "Normal")
    return out


def _from_netflow_v9(df: pd.DataFrame) -> pd.DataFrame:
    """nProbe NetFlow v9 field names (the UQ "NF-*" datasets use these)."""
    out = pd.DataFrame(index=df.index)
    if "flow_start_milliseconds" in df.columns:
        out["timestamp"] = _epoch_times(_num(df, "flow_start_milliseconds") / 1000)
    else:
        out["timestamp"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    out["src_ip"] = _text(df, "ipv4_src_addr")
    out["dst_ip"] = _text(df, "ipv4_dst_addr")
    out["src_port"] = _num(df, "l4_src_port")
    out["dst_port"] = _num(df, "l4_dst_port")
    out["protocol"] = _protocol(_col(df, "protocol"))
    out["flow_duration_ms"] = _num(df, "flow_duration_milliseconds")
    out["total_fwd_packets"] = _num(df, "in_pkts")
    out["total_bwd_packets"] = _num(df, "out_pkts")
    out["total_fwd_bytes"] = _num(df, "in_bytes")
    out["total_bwd_bytes"] = _num(df, "out_bytes")
    mask = _col(df, "tcp_flags")
    tcp = out["protocol"] == "tcp"
    bits = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16}
    for flag, bit in bits.items():
        out[f"tcp_{flag}_count"] = _flag_from_mask(mask, bit).where(tcp)
    if "attack" in df.columns:
        out["label"] = _text(df, "attack")
    elif "label" in df.columns:
        out["label"] = _text(df, "label")
    return out


@dataclass(frozen=True)
class SourceFormat:
    name: str
    # Every one of these normalised headers must be present to match.
    signature: frozenset[str]
    convert: Callable[[pd.DataFrame], pd.DataFrame]
    # Normalised headers worth reading; None reads every column.
    columns: frozenset[str] | None = None


_CIC_COLUMNS = frozenset(
    {
        "timestamp", "source ip", "src ip", "destination ip", "dst ip",
        "source port", "src port", "destination port", "dst port", "protocol",
        "flow duration", "total fwd packets", "tot fwd pkts",
        "total backward packets", "tot bwd pkts",
        "total length of fwd packets", "totlen fwd pkts",
        "total length of bwd packets", "totlen bwd pkts",
        "fwd header length", "fwd header len",
        "bwd header length", "bwd header len",
        "syn flag count", "syn flag cnt", "ack flag count", "ack flag cnt",
        "rst flag count", "rst flag cnt", "fin flag count", "fin flag cnt",
        "psh flag count", "psh flag cnt", "label",
    }
)

# Order matters: the first matching format wins.
FORMATS: list[SourceFormat] = [
    SourceFormat(
        "canonical",
        frozenset({"flow_duration_ms", "dst_port"}),
        _from_canonical,
    ),
    SourceFormat(
        "cicflowmeter",
        frozenset({"flow duration", "total fwd packets", "destination port"}),
        _from_cicflowmeter,
        _CIC_COLUMNS,
    ),
    SourceFormat(
        "cicflowmeter",
        frozenset({"flow duration", "tot fwd pkts", "dst port"}),
        _from_cicflowmeter,
        _CIC_COLUMNS,
    ),
    SourceFormat(
        "zeek_conn",
        frozenset({"id.orig_h", "id.resp_h", "id.resp_p"}),
        _from_zeek_conn,
    ),
    SourceFormat(
        "netflow_v9",
        frozenset({"ipv4_src_addr", "ipv4_dst_addr", "l4_dst_port", "in_pkts"}),
        _from_netflow_v9,
    ),
    SourceFormat(
        "unsw_nb15",
        frozenset({"dur", "spkts", "dpkts", "sbytes", "dbytes"}),
        _from_unsw_nb15,
    ),
]


def detect_format(columns: list[str] | pd.Index) -> SourceFormat | None:
    headers = {normalise_header(c) for c in columns}
    return next((f for f in FORMATS if f.signature <= headers), None)


def supported_formats() -> list[str]:
    return sorted({f.name for f in FORMATS})


def to_canonical(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Convert a raw export to canonical columns.

    Returns the converted frame and the detected format name. Unknown formats
    are returned unchanged (with stripped headers) so the validator can report
    exactly which required columns are missing.
    """
    original = df.copy()
    original.columns = [str(c).strip() for c in original.columns]
    original = original.loc[:, ~original.columns.duplicated()]

    fmt = detect_format(original.columns)
    if fmt is None:
        return original, "unknown"

    normalised = original.copy()
    normalised.columns = [normalise_header(c) for c in normalised.columns]
    normalised = normalised.loc[:, ~normalised.columns.duplicated()]
    converted = fmt.convert(normalised)
    # Required columns always exist (the validator checks their values);
    # optional columns the export cannot provide are dropped so the validator
    # reports them as missing and the job carries a warning.
    for column in REQUIRED_COLUMNS:
        if column not in converted.columns and fmt.name != "canonical":
            converted[column] = np.nan
    empty = [
        c
        for c in CANONICAL_COLUMNS
        if c not in REQUIRED_COLUMNS
        and c in converted.columns
        and converted[c].isna().all()
    ]
    converted = converted.drop(columns=empty)
    return converted.reset_index(drop=True), fmt.name
