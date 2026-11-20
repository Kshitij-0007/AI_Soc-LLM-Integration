"""Per-flow features for the NetFlow lane (feature set 2).

Design rules, chosen so a model trained on one network keeps working on
another:

* Only canonical columns are used (see adapters.py), so units and meanings
  are identical whichever exporter produced the file.
* Rates are derived here from packets, bytes and duration, never taken from
  the exporter (exporters compute them differently).
* TCP flags are used as "seen / not seen", because exporters count them
  differently (0/1, bitmask, history string).
* No identifiers: IPs, raw ports and absolute timestamps are not features.
  Ports are reduced to service groups.
* Source context (how many flows, destinations and ports the same source
  touched in the same minute) is computed from the uploaded file itself, so
  it exists for every exporter that provides IPs and timestamps. It replaces
  the old ``failed_connections`` / ``unique_destinations`` inputs, which no
  real exporter provides.
* Missing values stay NaN. XGBoost learns a default branch for them, which
  is more honest than imputing a median that was never observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.contracts import FeatureFrame
from app.core.lane_spec import LaneSpec

FEATURE_SET_VERSION = "2"
CONTEXT_WINDOW = "60s"

ADMIN_PORTS = {
    21, 22, 23, 135, 139, 445, 1433, 1521, 3306, 3389, 5432, 5900, 5985, 5986,
}
WEB_PORTS = {80, 443, 8000, 8080, 8443}
INFRA_PORTS = {53, 67, 68, 88, 123, 161, 389, 636}

FLOW_FEATURES = [
    "log_duration_ms",
    "log_fwd_packets",
    "log_bwd_packets",
    "log_fwd_bytes",
    "log_bwd_bytes",
    "log_packets_per_second",
    "log_bytes_per_second",
    "fwd_bytes_per_packet",
    "bwd_bytes_per_packet",
    "fwd_packet_share",
    "fwd_byte_share",
    "no_response",
]
FLAG_FEATURES = ["syn_seen", "ack_seen", "rst_seen", "fin_seen", "psh_seen"]
PROTOCOL_FEATURES = ["proto_tcp", "proto_udp", "proto_icmp"]
PORT_FEATURES = ["port_admin", "port_web", "port_infra", "port_ephemeral"]
CONTEXT_FEATURES = [
    "src_log_flows_per_window",
    "src_log_unique_dst_ips",
    "src_log_unique_dst_ports",
    "src_no_response_share",
    "src_same_port_share",
    "pair_log_flows_per_window",
]

FEATURE_COLUMNS = (
    FLOW_FEATURES + FLAG_FEATURES + PROTOCOL_FEATURES + PORT_FEATURES + CONTEXT_FEATURES
)

# Feature groups an exporter may be unable to provide. Training adds copies
# with these blanked so the model copes when an upload lacks them.
OPTIONAL_FEATURE_GROUPS = {
    "flags": FLAG_FEATURES,
    "context": CONTEXT_FEATURES,
    "ports": PORT_FEATURES,
}

# Placeholder strings written by the validator for absent optional columns.
_MISSING_LABELS = {"", "unknown", "nan", "none", "null", "<na>"}
_MIN_DURATION_S = 0.001


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.where(denominator > 0)


def _log(values: pd.Series) -> pd.Series:
    return np.log1p(values.clip(lower=0))


def _known(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.notna() & ~text.isin(_MISSING_LABELS)


def _context_features(
    df: pd.DataFrame,
    dst_port: pd.Series,
    no_response: pd.Series,
) -> pd.DataFrame:
    """Per source IP and minute: volume, fan-out and failure share."""
    context = pd.DataFrame(np.nan, index=df.index, columns=CONTEXT_FEATURES)
    if not {"timestamp", "src_ip", "dst_ip"} <= set(df.columns):
        return context

    times = pd.to_datetime(df["timestamp"], errors="coerce")
    usable = times.notna() & _known(df["src_ip"]) & _known(df["dst_ip"])
    if not usable.any():
        return context

    frame = pd.DataFrame(
        {
            "src": df.loc[usable, "src_ip"].astype(str),
            "dst": df.loc[usable, "dst_ip"].astype(str),
            "port": dst_port[usable].fillna(-1),
            "no_response": no_response[usable],
            "bucket": times[usable].dt.floor(CONTEXT_WINDOW),
        }
    )
    by_source = frame.groupby(["src", "bucket"], sort=False)
    flows = by_source["dst"].transform("size").astype(float)
    context.loc[usable, "src_log_flows_per_window"] = np.log1p(flows)
    context.loc[usable, "src_log_unique_dst_ips"] = np.log1p(
        by_source["dst"].transform("nunique").astype(float)
    )
    context.loc[usable, "src_log_unique_dst_ports"] = np.log1p(
        by_source["port"].transform("nunique").astype(float)
    )
    context.loc[usable, "src_no_response_share"] = by_source["no_response"].transform(
        "mean"
    )
    same_port = frame.groupby(["src", "bucket", "port"], sort=False)["dst"].transform(
        "size"
    )
    context.loc[usable, "src_same_port_share"] = same_port / flows
    pair = frame.groupby(["src", "bucket", "dst"], sort=False)["port"].transform("size")
    context.loc[usable, "pair_log_flows_per_window"] = np.log1p(pair.astype(float))
    return context


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features for one flow per row, in ``FEATURE_COLUMNS`` order.

    ``df`` must use canonical columns. NaN marks a value the export could not
    provide.
    """
    fwd_packets = _numeric(df, "total_fwd_packets")
    bwd_packets = _numeric(df, "total_bwd_packets")
    fwd_bytes = _numeric(df, "total_fwd_bytes")
    bwd_bytes = _numeric(df, "total_bwd_bytes")
    duration_ms = _numeric(df, "flow_duration_ms").clip(lower=0)
    packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes
    seconds = (duration_ms / 1000.0).clip(lower=_MIN_DURATION_S)

    features = pd.DataFrame(index=df.index)
    features["log_duration_ms"] = _log(duration_ms)
    features["log_fwd_packets"] = _log(fwd_packets)
    features["log_bwd_packets"] = _log(bwd_packets)
    features["log_fwd_bytes"] = _log(fwd_bytes)
    features["log_bwd_bytes"] = _log(bwd_bytes)
    features["log_packets_per_second"] = _log(packets / seconds)
    features["log_bytes_per_second"] = _log(total_bytes / seconds)
    features["fwd_bytes_per_packet"] = _ratio(fwd_bytes, fwd_packets)
    features["bwd_bytes_per_packet"] = _ratio(bwd_bytes, bwd_packets)
    features["fwd_packet_share"] = _ratio(fwd_packets, packets)
    features["fwd_byte_share"] = _ratio(fwd_bytes, total_bytes)
    no_response = (bwd_packets == 0).astype(float).where(bwd_packets.notna())
    features["no_response"] = no_response

    for flag in ("syn", "ack", "rst", "fin", "psh"):
        counts = _numeric(df, f"tcp_{flag}_count")
        features[f"{flag}_seen"] = (counts > 0).astype(float).where(counts.notna())

    protocol = (
        df["protocol"].astype("string").str.strip().str.lower()
        if "protocol" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    protocol_known = _known(protocol)
    for name in ("tcp", "udp", "icmp"):
        features[f"proto_{name}"] = (
            (protocol == name).astype(float).where(protocol_known)
        )

    port = _numeric(df, "dst_port")
    features["port_admin"] = port.isin(ADMIN_PORTS).astype(float).where(port.notna())
    features["port_web"] = port.isin(WEB_PORTS).astype(float).where(port.notna())
    features["port_infra"] = port.isin(INFRA_PORTS).astype(float).where(port.notna())
    features["port_ephemeral"] = (
        ((port >= 1024) & ~port.isin(WEB_PORTS)).astype(float).where(port.notna())
    )

    context = _context_features(df, port, no_response)
    features = pd.concat([features, context], axis=1)
    return features[FEATURE_COLUMNS].astype(float)


def extract_labels(df: pd.DataFrame, label_column: str | None) -> pd.Series | None:
    """Native labels when every row has one, otherwise None (scoring uploads)."""
    if not label_column or label_column not in df.columns:
        return None
    labels = df[label_column].astype("string").str.strip()
    if labels.isna().any() or labels.str.lower().isin(_MISSING_LABELS).any():
        return None
    return labels.astype(str).reset_index(drop=True)


def _text_or_unknown(df: pd.DataFrame, column: str | None) -> np.ndarray:
    if column and column in df.columns:
        return df[column].astype("string").fillna("unknown").astype(str).to_numpy()
    return np.full(len(df), "unknown", dtype=object)


def build_features(df: pd.DataFrame, spec: LaneSpec) -> FeatureFrame:
    """One feature row per flow. Same function for training and scoring."""
    if df.empty:
        raise ValueError(
            "The validated dataset is empty. Check spec.yaml and validator.py."
        )

    df = df.reset_index(drop=True)
    X = compute_features(df)

    timestamps = (
        pd.to_datetime(df["timestamp"], errors="coerce")
        if "timestamp" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    entity_field = spec.entity.get("field", "src_ip") if spec.entity else "src_ip"
    packets = _numeric(df, "total_fwd_packets") + _numeric(df, "total_bwd_packets")
    total_bytes = _numeric(df, "total_fwd_bytes") + _numeric(df, "total_bwd_bytes")

    meta = pd.DataFrame(
        {
            "entity_type": spec.entity.get("type", "ip") if spec.entity else "ip",
            "entity_value": _text_or_unknown(df, entity_field),
            "ip": _text_or_unknown(df, "src_ip"),
            "target_value": _text_or_unknown(df, "dst_ip"),
            "window_start": timestamps.to_numpy(),
            "window_end": timestamps.to_numpy(),
            "event_count": 1,
            # Volume fields kept for later intensity / risk scoring.
            "packets": packets.to_numpy(),
            "bytes": total_bytes.to_numpy(),
            "duration_ms": _numeric(df, "flow_duration_ms").to_numpy(),
        }
    )

    return FeatureFrame(X=X, meta=meta, y=extract_labels(df, spec.label_column))
