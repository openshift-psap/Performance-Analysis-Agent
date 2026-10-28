"""Tests for startup loading and caching of RHAIIS performance data."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from psap_mcp_server.src.tools import performance_data_loader as loader
from psap_mcp_server.src.tools import s3_utils


@pytest.fixture(autouse=True)
def clear_rhaiis_cache():
    loader._data_cache.invalidate("rhaiis_data")
    yield
    loader._data_cache.invalidate("rhaiis_data")


def test_startup_validation_warms_cache_for_first_tool_load(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        S3_BUCKET="benchmark-data",
        S3_KEY="main/dashboard.csv",
        S3_REGION="us-east-1",
        S3_CACHE_TTL_SECONDS=300,
    )
    raw_data = pd.DataFrame(
        {
            "run": [" run-a "],
            "accelerator": [" H100 "],
            "model": [" model-a "],
            "version": [" v1 "],
            "TP": ["2"],
            "output_tok/sec": [100.0],
        }
    )
    read_s3 = Mock(return_value=raw_data)
    monkeypatch.setattr(loader, "settings", settings)
    monkeypatch.setattr(loader, "RHAIIS_DATA_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(loader, "_read_csv_from_s3", read_s3)

    loader.validate_data_availability()
    cached = loader._data_cache.get("rhaiis_data", settings.S3_CACHE_TTL_SECONDS)

    assert cached is not None
    assert cached.source == "S3: s3://benchmark-data/main/dashboard.csv"
    assert cached.data.loc[0, "run"] == "run-a"
    assert cached.data.loc[0, "TP"] == 2

    loaded_for_tool = loader.load_rhaiis_data()

    assert loaded_for_tool is not None
    assert loaded_for_tool.equals(cached.data)
    read_s3.assert_called_once_with("benchmark-data", "main/dashboard.csv", "us-east-1")


def test_startup_validation_exits_when_no_data_source_can_be_loaded(monkeypatch):
    monkeypatch.setattr(loader, "load_rhaiis_data", Mock(return_value=None))

    with pytest.raises(SystemExit) as exc_info:
        loader.validate_data_availability()

    assert exc_info.value.code == 1


def test_read_csv_from_s3_parses_bytes_and_closes_response_body(monkeypatch):
    body = Mock()
    body.read.return_value = b"model,throughput\nmodel-a,123\n"
    s3_client = Mock()
    s3_client.get_object.return_value = {"Body": body}
    get_s3_client = Mock(return_value=s3_client)
    monkeypatch.setattr(s3_utils, "get_s3_client", get_s3_client)

    result = loader._read_csv_from_s3("benchmark-data", "dashboard.csv")

    assert result.to_dict(orient="records") == [{"model": "model-a", "throughput": 123}]
    get_s3_client.assert_called_once_with(
        region="us-east-1",
        test_bucket="benchmark-data",
        test_key="dashboard.csv",
    )
    s3_client.get_object.assert_called_once_with(
        Bucket="benchmark-data", Key="dashboard.csv"
    )
    body.read.assert_called_once_with()
    body.close.assert_called_once_with()
