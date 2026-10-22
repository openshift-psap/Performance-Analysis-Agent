"""Tests for aggregate-row and latest-run selection."""

import pandas as pd

from psap_mcp_server.src.tools.benchmark_row_selection import (
    exclude_per_turn_rows,
    select_latest_run_rows,
)


def _benchmark_row(
    *,
    run_id: str,
    concurrency: int,
    start_ms: int,
    throughput: float,
    model: str = "model-a",
) -> dict:
    return {
        "model": model,
        "accelerator": "H200",
        "version": "RHAIIS-3.6",
        "TP": 1,
        "prompt toks": 128,
        "output toks": 128,
        "uuid": run_id,
        "intended concurrency": concurrency,
        "guidellm_start_time_ms": start_ms,
        "output_tok/sec": throughput,
    }


def test_exclude_per_turn_rows_keeps_aggregate_and_legacy_rows():
    df = pd.DataFrame(
        {
            "turn_index": [None, "", 0, "1", "nan"],
            "output_tok/sec": [100, 110, 10, 20, 120],
        }
    )

    result = exclude_per_turn_rows(df)

    assert result["output_tok/sec"].tolist() == [100, 110, 120]
    assert len(df) == 5


def test_exclude_per_turn_rows_is_backward_compatible_without_column():
    df = pd.DataFrame({"output_tok/sec": [100, 110]})

    result = exclude_per_turn_rows(df)

    assert result.equals(df)
    assert result is not df


def test_latest_run_keeps_every_concurrency_row_from_newest_cohort():
    df = pd.DataFrame(
        [
            # The newer run appears first in the CSV to verify timestamp wins
            # over source order, while both of its concurrency rows survive.
            _benchmark_row(
                run_id="new-run", concurrency=1, start_ms=300, throughput=300
            ),
            _benchmark_row(
                run_id="new-run", concurrency=8, start_ms=400, throughput=800
            ),
            _benchmark_row(
                run_id="old-run", concurrency=1, start_ms=100, throughput=100
            ),
            _benchmark_row(
                run_id="old-run", concurrency=8, start_ms=200, throughput=200
            ),
            _benchmark_row(
                run_id="other-config",
                concurrency=1,
                start_ms=250,
                throughput=250,
                model="model-b",
            ),
        ]
    )

    result = select_latest_run_rows(df)

    assert result["uuid"].tolist() == ["new-run", "new-run", "other-config"]
    assert result["intended concurrency"].tolist() == [1, 8, 1]
    assert result["output_tok/sec"].tolist() == [300, 800, 250]


def test_latest_run_uses_mlflow_id_when_available_and_keeps_unidentified_rows():
    df = pd.DataFrame(
        [
            {
                **_benchmark_row(
                    run_id="old-deployment",
                    concurrency=1,
                    start_ms=100,
                    throughput=100,
                ),
                "mlflow_run_id": "old-sweep",
            },
            {
                **_benchmark_row(
                    run_id="new-deployment-1",
                    concurrency=1,
                    start_ms=300,
                    throughput=300,
                ),
                "mlflow_run_id": "new-sweep",
            },
            {
                **_benchmark_row(
                    run_id="new-deployment-8",
                    concurrency=8,
                    start_ms=400,
                    throughput=800,
                ),
                "mlflow_run_id": "new-sweep",
            },
            {
                **_benchmark_row(
                    run_id="",
                    concurrency=32,
                    start_ms=500,
                    throughput=900,
                ),
                "mlflow_run_id": "",
            },
        ]
    )

    result = select_latest_run_rows(df)

    assert result["uuid"].tolist() == [
        "new-deployment-1",
        "new-deployment-8",
        "",
    ]
    assert result["intended concurrency"].tolist() == [1, 8, 32]
