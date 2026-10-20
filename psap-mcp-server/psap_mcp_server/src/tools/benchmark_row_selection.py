"""Selection helpers for aggregate RHAIIS benchmark rows."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_RUN_CONFIGURATION_COLUMNS = (
    "model",
    "accelerator",
    "version",
    "TP",
    "DP",
    "EP",
    "prompt toks",
    "output toks",
    "runtime_args",
    "dataset",
    "spec_decoding",
    "prefix_caching",
    "request_type",
    "turns",
    "prefix_tokens",
    "prefix_count",
)


def exclude_per_turn_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop per-turn rows; current MCP analysis supports aggregate rows only."""
    if df.empty or "turn_index" not in df.columns:
        return df.copy()

    turn_index = df["turn_index"].astype("string").str.strip()
    has_turn_index = (
        turn_index.notna()
        & turn_index.ne("")
        & ~turn_index.str.lower().isin({"nan", "none", "<na>"})
    )
    excluded_rows = int(has_turn_index.sum())
    if excluded_rows:
        logger.info(
            "Excluding %s per-turn RHAIIS rows; aggregate-only analysis is enabled",
            excluded_rows,
        )
    return df.loc[~has_turn_index].copy()


def select_latest_run_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select the newest identifiable run per configuration.

    A run can contain multiple rows, typically one per concurrency level.
    This selects the newest run cohort by its MLflow run ID when available,
    otherwise by deployment UUID, while retaining every row in that cohort.
    Rows without a stable run identifier are retained because they cannot be
    safely assigned to a complete run cohort.
    """
    if df.empty:
        return df.copy()

    run_id = pd.Series(pd.NA, index=df.index, dtype="string")
    if "mlflow_run_id" in df.columns:
        mlflow_ids = df["mlflow_run_id"].astype("string").str.strip()
        mlflow_ids = mlflow_ids.mask(
            mlflow_ids.isna()
            | mlflow_ids.eq("")
            | mlflow_ids.str.lower().isin({"nan", "none", "<na>"})
        )
        run_id = mlflow_ids.map(
            lambda value: f"mlflow:{value}" if pd.notna(value) else pd.NA
        )

    if "uuid" in df.columns:
        uuids = df["uuid"].astype("string").str.strip()
        uuids = uuids.mask(
            uuids.isna()
            | uuids.eq("")
            | uuids.str.lower().isin({"nan", "none", "<na>"})
        )
        use_uuid = run_id.isna() & uuids.notna()
        run_id.loc[use_uuid] = uuids.loc[use_uuid].map(lambda value: f"uuid:{value}")

    identified_mask = run_id.notna()
    if not identified_mask.any():
        logger.warning("Cannot select latest benchmark run: no stable run IDs found")
        return df.copy()

    configuration_columns = [
        column for column in _RUN_CONFIGURATION_COLUMNS if column in df.columns
    ]
    work = df.copy()
    if not configuration_columns:
        work["_latest_configuration"] = "all"
        configuration_columns = ["_latest_configuration"]

    work["_latest_run_id"] = run_id
    work["_latest_run_order"] = range(len(work))
    if "guidellm_start_time_ms" in work.columns:
        work["_latest_run_timestamp"] = pd.to_numeric(
            work["guidellm_start_time_ms"], errors="coerce"
        )
    else:
        work["_latest_run_timestamp"] = float("nan")

    identified = work.loc[identified_mask]
    run_keys = configuration_columns + ["_latest_run_id"]
    run_ranks = (
        identified.groupby(run_keys, dropna=False, sort=False)
        .agg(
            _latest_timestamp=("_latest_run_timestamp", "max"),
            _last_row=("_latest_run_order", "max"),
        )
        .reset_index()
    )
    # Timestamp is primary. Source order is a deterministic tie-breaker and
    # fallback when candidate runs have no timestamp.
    run_ranks["_latest_timestamp"] = run_ranks["_latest_timestamp"].fillna(
        float("-inf")
    )
    run_ranks = run_ranks.sort_values(["_latest_timestamp", "_last_row"], kind="stable")
    latest_runs = run_ranks.drop_duplicates(configuration_columns, keep="last")

    identified_keys = pd.MultiIndex.from_frame(identified[run_keys])
    latest_keys = pd.MultiIndex.from_frame(latest_runs[run_keys])
    selected = work.loc[identified_mask].loc[identified_keys.isin(latest_keys)]
    unidentified = work.loc[~identified_mask]
    if not unidentified.empty:
        logger.warning(
            "Retaining %s benchmark rows without a stable run ID; "
            "latest-run selection cannot be applied to those rows",
            len(unidentified),
        )

    result = pd.concat([selected, unidentified]).sort_values(
        "_latest_run_order", kind="stable"
    )
    return result.drop(
        columns=[
            "_latest_configuration",
            "_latest_run_id",
            "_latest_run_order",
            "_latest_run_timestamp",
        ],
        errors="ignore",
    )
