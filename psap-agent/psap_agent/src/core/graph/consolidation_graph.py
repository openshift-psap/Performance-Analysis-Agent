"""Consolidation graph — Dreams-like 4-phase memory cleanup.

Runs after a full version pipeline completes. Reads all facts + red flags,
asks the LLM to orient / gather signal / consolidate / prune, then applies
the resulting mutations to the database and archives the red flags file.

Pipeline:
  load_all_facts → consolidate → fold_red_flags → write_back → log_summary → END
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.memory.db import (
    get_connection,
    get_sqlite_conn_and_lock,
    is_postgres,
)
from psap_agent.src.core.memory.queries import (
    get_config_facts,
    get_model_summary,
    get_version_summary,
    update_fact_status,
    upsert_config_finding,
)
from psap_agent.src.core.memory.red_flags import RedFlagsManager
from psap_agent.src.prompts.consolidation import (
    CONSOLIDATION_SYSTEM,
    build_consolidation_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# Node 1: load_all_facts
# ---------------------------------------------------------------------------


async def load_all_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Read all facts from every table + the red flags file for this version."""
    version = state.get("rhaiis_version", "")

    config_facts = await get_config_facts(version, limit=200)
    model_summaries = await get_model_summary(version, limit=50)
    version_summaries = await get_version_summary(version, limit=10)

    red_flags_dir = config.get("configurable", {}).get("red_flags_dir")
    rfm = RedFlagsManager(base_dir=red_flags_dir)
    red_flags_text = rfm.read_file(version)

    total = len(config_facts) + len(model_summaries) + len(version_summaries)
    logger.info(
        "load_all_facts: %d config, %d model, %d version facts; "
        "red flags %d chars",
        len(config_facts),
        len(model_summaries),
        len(version_summaries),
        len(red_flags_text),
    )

    facts_blob = build_consolidation_user_prompt(
        current_version=version,
        config_facts=config_facts,
        model_summaries=model_summaries,
        version_summaries=version_summaries,
        red_flags_text=red_flags_text,
    )

    return {
        "prior_facts": facts_blob,
        "red_flags": red_flags_text,
        # Stash raw lists in config for downstream nodes
        "config": {
            **state.get("config", {}),
            "_config_facts": config_facts,
            "_model_summaries": model_summaries,
            "_version_summaries": version_summaries,
            "_facts_total": total,
        },
    }


# ---------------------------------------------------------------------------
# Node 2: consolidate (LLM call)
# ---------------------------------------------------------------------------


def _parse_consolidation_response(text: str) -> dict:
    """Extract JSON from the LLM response, tolerating markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        logger.warning("consolidate: failed to parse LLM JSON, returning empty plan")
        return {"orientation": {}, "actions": [], "red_flag_actions": [], "summary": ""}


async def consolidate(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM call with the Dreams-like 4-phase consolidation prompt."""
    fact_model = config["configurable"]["fact_model"]
    user_prompt = state.get("prior_facts", "")

    response = await fact_model.ainvoke(
        [
            SystemMessage(content=CONSOLIDATION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    plan = _parse_consolidation_response(response.content)
    logger.info(
        "consolidate: %d actions, %d red_flag_actions",
        len(plan.get("actions", [])),
        len(plan.get("red_flag_actions", [])),
    )

    return {
        "extracted_facts": plan.get("actions", []),
        "validated_facts": plan.get("red_flag_actions", []),
        "report": plan.get("summary", ""),
        "reconciled_facts": [plan],
    }


# ---------------------------------------------------------------------------
# Node 3: fold_red_flags
# ---------------------------------------------------------------------------


async def fold_red_flags(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Promote valuable red flags to persistent config_findings."""
    plans = state.get("reconciled_facts", [])
    plan = plans[0] if plans else {}
    rf_actions = plan.get("red_flag_actions", [])
    version = state.get("rhaiis_version", "")
    baseline = state.get("baseline_version", "")
    run_id = state.get("run_id", "")
    folded_count = 0

    for rf in rf_actions:
        if rf.get("action") != "insert":
            continue
        try:
            await upsert_config_finding(
                {
                    "fact_text": rf.get("fact_text", ""),
                    "category": rf.get("category", "cross_config"),
                    "rhaiis_version": version,
                    "baseline_version": baseline,
                    "model_name": rf.get("model_name", "ALL MODELS"),
                    "accelerator": rf.get("accelerator", "all"),
                    "profile": rf.get("profile", "all"),
                    "status": "active",
                    "confidence": "medium",
                    "source": "red_flag_consolidation",
                    "run_id": run_id,
                }
            )
            folded_count += 1
        except Exception:
            logger.exception("fold_red_flags: failed to insert promoted red flag")

    logger.info("fold_red_flags: promoted %d red flags to persistent facts", folded_count)
    return {"red_flags_added_count": folded_count}


# ---------------------------------------------------------------------------
# Node 4: write_back
# ---------------------------------------------------------------------------

TABLE_FOR_ACTION = {
    "config_findings": "config_findings",
    "model_summaries": "model_summaries",
    "version_summaries": "version_summaries",
}


async def _delete_fact(fact_id: str, table: str) -> bool:
    """Hard-delete a single fact by ID."""
    if is_postgres():
        conn = await get_connection()
        try:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE id=$1", fact_id
            )
            return result == "DELETE 1"
        finally:
            await conn.close()
    else:
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            cur = c.execute(f"DELETE FROM {table} WHERE id=?", (fact_id,))
            deleted = cur.rowcount > 0
            c.commit()
            return deleted


async def _update_fact_text(fact_id: str, new_text: str, table: str) -> bool:
    """Update the text content of a fact (merge / compress)."""
    now = datetime.now(timezone.utc)
    text_col = "fact_text" if table == "config_findings" else "summary_text"
    if is_postgres():
        conn = await get_connection()
        try:
            result = await conn.execute(
                f"UPDATE {table} SET {text_col}=$1, updated_at=$2 WHERE id=$3",
                new_text,
                now,
                fact_id,
            )
            return result == "UPDATE 1"
        finally:
            await conn.close()
    else:
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            cur = c.execute(
                f"UPDATE {table} SET {text_col}=?, updated_at=? WHERE id=?",
                (new_text, now.isoformat(), fact_id),
            )
            ok = cur.rowcount > 0
            c.commit()
            return ok


def _infer_table(fact_id: str, config_facts: list, model_sums: list, version_sums: list) -> str:
    """Figure out which table a fact_id belongs to."""
    cf_ids = {f.get("id") for f in config_facts}
    ms_ids = {f.get("id") for f in model_sums}
    vs_ids = {f.get("id") for f in version_sums}
    if fact_id in cf_ids:
        return "config_findings"
    if fact_id in ms_ids:
        return "model_summaries"
    if fact_id in vs_ids:
        return "version_summaries"
    return "config_findings"


async def write_back(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Apply consolidation mutations to the database and archive red flags."""
    plans = state.get("reconciled_facts", [])
    plan = plans[0] if plans else {}
    actions = plan.get("actions", [])
    version = state.get("rhaiis_version", "")
    cfg = state.get("config", {})
    config_facts = cfg.get("_config_facts", [])
    model_sums = cfg.get("_model_summaries", [])
    version_sums = cfg.get("_version_summaries", [])

    pruned = 0
    merged = 0

    for action in actions:
        action_type = action.get("action", "")
        try:
            if action_type == "delete":
                tid = action.get("target_id", "")
                table = _infer_table(tid, config_facts, model_sums, version_sums)
                if await _delete_fact(tid, table):
                    pruned += 1

            elif action_type == "mark_stale":
                tid = action.get("target_id", "")
                table = _infer_table(tid, config_facts, model_sums, version_sums)
                await update_fact_status(tid, "stale", table)
                pruned += 1

            elif action_type == "merge":
                target_ids = action.get("target_ids", [])
                merged_text = action.get("merged_text", "")
                new_status = action.get("status", "active")
                if not target_ids:
                    continue
                keep_id = target_ids[0]
                keep_table = _infer_table(keep_id, config_facts, model_sums, version_sums)
                await _update_fact_text(keep_id, merged_text, keep_table)
                await update_fact_status(keep_id, new_status, keep_table)
                for remove_id in target_ids[1:]:
                    rm_table = _infer_table(remove_id, config_facts, model_sums, version_sums)
                    await _delete_fact(remove_id, rm_table)
                merged += 1

            elif action_type == "compress":
                tid = action.get("target_id", "")
                table = _infer_table(tid, config_facts, model_sums, version_sums)
                await _update_fact_text(tid, action.get("compressed_text", ""), table)
                merged += 1

        except Exception:
            logger.exception("write_back: failed to apply action %s", action_type)

    red_flags_dir = config.get("configurable", {}).get("red_flags_dir")
    rfm = RedFlagsManager(base_dir=red_flags_dir)
    rfm.archive(version)

    logger.info("write_back: pruned=%d, merged=%d", pruned, merged)
    return {
        "facts_stored_count": merged,
        "config": {
            **cfg,
            "_pruned": pruned,
            "_merged": merged,
        },
    }


# ---------------------------------------------------------------------------
# Node 5: log_summary
# ---------------------------------------------------------------------------


async def _insert_consolidation_log(
    version: str,
    facts_pruned: int,
    facts_merged: int,
    red_flags_folded: int,
    summary_text: str,
) -> str:
    """Write a row to the consolidation_log table."""
    log_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    if is_postgres():
        conn = await get_connection()
        try:
            await conn.execute(
                """INSERT INTO consolidation_log
                (id, rhaiis_version, facts_pruned, facts_merged,
                 red_flags_folded, summary_text, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                log_id,
                version,
                facts_pruned,
                facts_merged,
                red_flags_folded,
                summary_text,
                now,
            )
        finally:
            await conn.close()
    else:
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            c.execute(
                """INSERT INTO consolidation_log
                (id, rhaiis_version, facts_pruned, facts_merged,
                 red_flags_folded, summary_text, created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (
                    log_id,
                    version,
                    facts_pruned,
                    facts_merged,
                    red_flags_folded,
                    summary_text,
                    now.isoformat(),
                ),
            )
            c.commit()

    return log_id


async def log_summary(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Write to consolidation_log and produce a human-readable report."""
    version = state.get("rhaiis_version", "")
    cfg = state.get("config", {})
    pruned = cfg.get("_pruned", 0)
    merged = cfg.get("_merged", 0)
    folded = state.get("red_flags_added_count", 0)
    llm_summary = state.get("report", "")

    report = (
        f"# Consolidation Report — {version}\n\n"
        f"- Facts pruned/marked stale: {pruned}\n"
        f"- Facts merged/compressed: {merged}\n"
        f"- Red flags promoted to persistent facts: {folded}\n"
        f"- Total facts at start: {cfg.get('_facts_total', '?')}\n\n"
        f"## LLM Summary\n\n{llm_summary}\n"
    )

    log_id = await _insert_consolidation_log(
        version=version,
        facts_pruned=pruned,
        facts_merged=merged,
        red_flags_folded=folded,
        summary_text=report,
    )

    logger.info("log_summary: consolidation log %s written for %s", log_id, version)
    return {
        "report": report,
        "facts_stored_count": merged + folded,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_consolidation_graph():
    """Compile the consolidation-level StateGraph."""
    graph = StateGraph(AnalysisState)

    graph.add_node("load_all_facts", load_all_facts)
    graph.add_node("consolidate", consolidate)
    graph.add_node("fold_red_flags", fold_red_flags)
    graph.add_node("write_back", write_back)
    graph.add_node("log_summary", log_summary)

    graph.add_edge(START, "load_all_facts")
    graph.add_edge("load_all_facts", "consolidate")
    graph.add_edge("consolidate", "fold_red_flags")
    graph.add_edge("fold_red_flags", "write_back")
    graph.add_edge("write_back", "log_summary")
    graph.add_edge("log_summary", END)

    return graph.compile()
