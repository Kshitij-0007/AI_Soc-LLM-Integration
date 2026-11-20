from __future__ import annotations

from pathlib import Path

from app.core.job_service import missing_columns_message
from app.core.validator import validate
from app.lanes.netflow.parser import parse

LANES_DIR = Path(__file__).resolve().parents[2] / "app" / "lanes"


def test_valid_sample_ok(netflow_spec, netflow_samples):
    result = validate(parse(netflow_samples / "valid_sample.csv"), netflow_spec)
    assert result.status in {"ok", "warning"}
    assert result.rows_valid == result.rows_total > 0


def test_missing_optional_warns(netflow_spec, netflow_samples):
    result = validate(parse(netflow_samples / "missing_optional.csv"), netflow_spec)
    assert result.status == "warning"
    assert "total_fwd_packets" in result.validation_report["missing_optional"]
    assert "label" not in result.validation_report["missing_optional"]


def test_missing_required_rejected(netflow_spec, netflow_samples):
    result = validate(parse(netflow_samples / "missing_required.csv"), netflow_spec)
    assert result.status == "rejected"
    assert result.missing_required == ["dst_port"]


def test_wrong_lane_gives_readable_message(netflow_spec):
    cloud_file = LANES_DIR / "cloud_entity" / "samples" / "valid_sample.csv"
    df = parse(cloud_file)
    assert df.attrs["source_format"] == "unknown"
    result = validate(df, netflow_spec)
    assert result.status == "rejected"
    message = missing_columns_message(result, netflow_spec)
    assert message.startswith("missing required columns: ")
    assert "dst_ip (also tried: dest_ip" in message
    assert "flow_duration_ms" in message
    assert "cicflowmeter" in message
