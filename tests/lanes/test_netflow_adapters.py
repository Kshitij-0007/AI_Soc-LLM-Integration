from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.lanes.netflow.adapters import detect_format, to_canonical
from app.lanes.netflow.parser import parse


def _cic2017_row(**overrides) -> dict:
    row = {
        " Source IP": "192.168.10.5",
        " Destination IP": "192.168.10.50",
        " Source Port": 54865,
        " Destination Port": 22,
        " Protocol": 6,
        " Timestamp": "6/7/2017 1:05",
        " Flow Duration": 2_000_000,  # microseconds
        " Total Fwd Packets": 10,
        " Total Backward Packets": 8,
        "Total Length of Fwd Packets": 1000,
        " Total Length of Bwd Packets": 2000,
        " Fwd Header Length": 320,
        " Bwd Header Length": 256,
        " SYN Flag Count": 1,
        " ACK Flag Count": 0,
        " RST Flag Count": 0,
        "FIN Flag Count": 1,
        " PSH Flag Count": 0,
        " Label": "SSH-Patator",
    }
    row.update(overrides)
    return row


def test_cicflowmeter_2017_units_and_names():
    df, fmt = to_canonical(pd.DataFrame([_cic2017_row()]))
    assert fmt == "cicflowmeter"
    row = df.iloc[0]
    assert row["flow_duration_ms"] == pytest.approx(2000.0)
    assert row["total_fwd_bytes"] == pytest.approx(1000 + 320)  # payload + header
    assert row["total_bwd_bytes"] == pytest.approx(2000 + 256)
    assert row["protocol"] == "tcp"
    assert row["dst_port"] == 22
    assert row["label"] == "SSH-Patator"
    # 12-hour clock without AM/PM: 1:05 during a working-hours capture is 13:05.
    assert row["timestamp"] == pd.Timestamp("2017-07-06 13:05")


def test_cicflowmeter_invalid_header_length_uses_estimate():
    df, _ = to_canonical(pd.DataFrame([_cic2017_row(**{" Fwd Header Length": -5})]))
    assert df.iloc[0]["total_fwd_bytes"] == pytest.approx(1000 + 10 * 40)


def test_cicflowmeter_2018_short_names():
    raw = pd.DataFrame(
        [
            {
                "Dst Port": 443, "Protocol": 6, "Timestamp": "02/03/2018 08:47:38",
                "Flow Duration": 5000, "Tot Fwd Pkts": 3, "Tot Bwd Pkts": 2,
                "TotLen Fwd Pkts": 100, "TotLen Bwd Pkts": 50,
                "SYN Flag Cnt": 0, "Label": "Benign",
            }
        ]
    )
    df, fmt = to_canonical(raw)
    assert fmt == "cicflowmeter"
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2018-03-02 08:47:38")
    assert df.iloc[0]["flow_duration_ms"] == pytest.approx(5.0)
    assert df.iloc[0]["src_ip"] == "unknown"


def test_zeek_conn_history_flags():
    raw = pd.DataFrame(
        [
            {
                "ts": 1_500_000_000.5, "id.orig_h": "10.0.0.1", "id.orig_p": 5555,
                "id.resp_h": "10.0.0.2", "id.resp_p": 80, "proto": "tcp",
                "duration": 1.5, "orig_pkts": 4, "resp_pkts": 3,
                "orig_ip_bytes": 400, "resp_ip_bytes": 900, "history": "ShADadFf",
            },
            {
                "ts": 1_500_000_001, "id.orig_h": "10.0.0.1", "id.orig_p": 5556,
                "id.resp_h": "10.0.0.3", "id.resp_p": 53, "proto": "udp",
                "duration": 0.01, "orig_pkts": 1, "resp_pkts": 1,
                "orig_ip_bytes": 60, "resp_ip_bytes": 120, "history": "Dd",
            },
        ]
    )
    df, fmt = to_canonical(raw)
    assert fmt == "zeek_conn"
    assert "tcp_psh_count" in df.columns
    tcp, udp = df.iloc[0], df.iloc[1]
    assert tcp["flow_duration_ms"] == pytest.approx(1500.0)
    assert tcp["tcp_syn_count"] > 0 and tcp["tcp_fin_count"] > 0
    assert tcp["tcp_rst_count"] == 0
    assert np.isnan(udp["tcp_syn_count"])  # flags do not apply to UDP


def test_netflow_v9_flag_bitmask():
    raw = pd.DataFrame(
        [
            {
                "IPV4_SRC_ADDR": "10.0.0.1", "L4_SRC_PORT": 1234,
                "IPV4_DST_ADDR": "10.0.0.2", "L4_DST_PORT": 22, "PROTOCOL": 6,
                "IN_BYTES": 500, "IN_PKTS": 5, "OUT_BYTES": 300, "OUT_PKTS": 4,
                "TCP_FLAGS": 2 | 16, "FLOW_DURATION_MILLISECONDS": 42,
                "Label": 1, "Attack": "SSH-Bruteforce",
            }
        ]
    )
    df, fmt = to_canonical(raw)
    assert fmt == "netflow_v9"
    row = df.iloc[0]
    assert row["tcp_syn_count"] == 1 and row["tcp_ack_count"] == 1
    assert row["tcp_rst_count"] == 0
    assert row["label"] == "SSH-Bruteforce"
    assert pd.isna(row["timestamp"])


def test_unsw_nb15_seconds_to_ms():
    raw = pd.DataFrame(
        [{"dur": 0.5, "proto": "tcp", "spkts": 4, "dpkts": 2, "sbytes": 300,
          "dbytes": 100, "attack_cat": "Reconnaissance", "label": 1}]
    )
    df, fmt = to_canonical(raw)
    assert fmt == "unsw_nb15"
    assert df.iloc[0]["flow_duration_ms"] == pytest.approx(500.0)
    assert df.iloc[0]["label"] == "Reconnaissance"
    # No flags in this export: optional flag columns are absent, so the
    # validator reports them and the job gets a warning.
    assert "tcp_syn_count" not in df.columns
    assert "timestamp" in df.columns  # required columns always exist


def test_unknown_format_passes_through():
    raw = pd.DataFrame([{"timestamp": "x", "principal_name": "p"}])
    df, fmt = to_canonical(raw)
    assert fmt == "unknown"
    assert list(df.columns) == ["timestamp", "principal_name"]


def test_detect_format_canonical():
    fmt = detect_format(["timestamp", "dst_port", "flow_duration_ms"])
    assert fmt.name == "canonical"


def test_parse_reads_cic_csv_and_drops_blank_rows(tmp_path):
    raw = pd.DataFrame([_cic2017_row(), _cic2017_row(**{" Label": "BENIGN"})])
    path = tmp_path / "cic.csv"
    raw.to_csv(path, index=False)
    with path.open("a") as handle:
        handle.write("," * (len(raw.columns) - 1) + "\n")
    df = parse(path, "csv")
    assert df.attrs["source_format"] == "cicflowmeter"
    assert len(df) == 2
