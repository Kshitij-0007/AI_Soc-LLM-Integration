from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.validator import validate
from app.lanes.netflow.features import FEATURE_COLUMNS, build_features, compute_features
from app.lanes.netflow.train import canonical_label


def _flows() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01 10:00:00")
    rows = []
    # One scanner touching 20 ports on one host with no responses.
    for port in range(20):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(seconds=port),
                "src_ip": "10.0.0.9", "dst_ip": "10.0.0.1", "dst_port": 1000 + port,
                "protocol": "tcp", "flow_duration_ms": 1.0,
                "total_fwd_packets": 1, "total_bwd_packets": 0,
                "total_fwd_bytes": 44, "total_bwd_bytes": 0,
                "tcp_syn_count": 1, "tcp_ack_count": 0,
            }
        )
    rows.append(
        {
            "timestamp": base, "src_ip": "10.0.0.5", "dst_ip": "10.0.0.80",
            "dst_port": 443, "protocol": "tcp", "flow_duration_ms": 5000.0,
            "total_fwd_packets": 10, "total_bwd_packets": 12,
            "total_fwd_bytes": 1500, "total_bwd_bytes": 15000,
            "tcp_syn_count": 1, "tcp_ack_count": 1,
        }
    )
    return pd.DataFrame(rows)


def test_feature_order_is_fixed():
    features = compute_features(_flows())
    assert list(features.columns) == FEATURE_COLUMNS
    assert all(pd.api.types.is_float_dtype(features[c]) for c in features.columns)


def test_context_features_capture_scanning():
    features = compute_features(_flows())
    scanner, normal = features.iloc[0], features.iloc[-1]
    assert scanner["src_log_unique_dst_ports"] == pytest.approx(np.log1p(20))
    assert scanner["src_no_response_share"] == pytest.approx(1.0)
    assert normal["src_log_flows_per_window"] == pytest.approx(np.log1p(1))
    assert normal["no_response"] == 0


def test_missing_columns_stay_nan_not_zero():
    flows = _flows().drop(columns=["tcp_syn_count", "tcp_ack_count", "timestamp"])
    features = compute_features(flows)
    assert features["syn_seen"].isna().all()
    assert features["src_log_flows_per_window"].isna().all()
    assert features["log_duration_ms"].notna().all()


def test_build_features_meta_and_no_labels_on_upload(netflow_spec):
    validated = validate(_flows(), netflow_spec)
    ff = build_features(validated.clean_df, netflow_spec)
    assert ff.y is None
    assert len(ff.X) == len(ff.meta) == 21
    assert ff.meta.loc[0, "entity_value"] == "10.0.0.9"
    assert ff.meta.loc[0, "target_value"] == "10.0.0.1"


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("BENIGN", "BENIGN"),
        ("Benign", "BENIGN"),
        ("DoS Hulk", "DoS"),
        ("DoS attacks-SlowHTTPTest", "DoS"),
        ("DDoS", "DDoS"),
        ("DDOS attack-HOIC", "DDoS"),
        ("FTP-Patator", "Brute Force"),
        ("SSH-Bruteforce", "Brute Force"),
        ("Brute Force -Web", "Web Attack"),
        ("Web Attack \x96 XSS", "Web Attack"),
        ("SQL Injection", "Web Attack"),
        ("PortScan", "PortScan"),
        ("scanning", "PortScan"),
        ("Reconnaissance", "PortScan"),
        ("Bot", "Bot"),
        ("Infiltration", None),
        ("Heartbleed", None),
        ("unknown", None),
    ],
)
def test_canonical_label(native, expected):
    assert canonical_label(native) == expected
