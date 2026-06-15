"""MCP tools for key facts memory layer (interactive use).

3-table design: config_findings, model_summaries, version_summaries.
These tools are exposed via MCP for interactive chat sessions.
"""

from typing import Any, Dict, Optional

from psap_mcp_server.src.tools.key_facts_db import (
    ensure_schema,
    query_config_findings,
    query_model_summaries,
    query_version_summaries,
    upsert_config_finding,
)
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()
_schema_ensured = False


async def _ensure_schema_once() -> None:
    global _schema_ensured
    if not _schema_ensured:
        await ensure_schema()
        _schema_ensured = True


async def search_key_facts(
    rhaiis_version: Optional[str] = None, model_name: Optional[str] = None,
    accelerator: Optional[str] = None, profile: Optional[str] = None,
    category: Optional[str] = None, limit: int = 30,
) -> Dict[str, Any]:
    """Search config-level findings from prior analysis runs.

    TOOL_NAME=search_key_facts
    DISPLAY_NAME=Search Key Facts
    USECASE=Retrieve previously discovered performance findings
    INSTRUCTIONS=Provide one or more filters. All use partial matching.
    INPUT_DESCRIPTION=rhaiis_version, model_name, accelerator, profile, category, limit
    OUTPUT_DESCRIPTION=Dictionary with matching findings
    EXAMPLES=search_key_facts(rhaiis_version="BENCH-2.5", model_name="Helios")
    PREREQUISITES=Database must be accessible
    RELATED_TOOLS=get_version_summary, get_model_performance_history
    """
    try:
        await _ensure_schema_once()
        facts = await query_config_findings(
            rhaiis_version=rhaiis_version, model_name=model_name,
            accelerator=accelerator, profile=profile, category=category, limit=limit,
        )
        for f in facts:
            for k in ("created_at", "updated_at"):
                if k in f and f[k] is not None:
                    f[k] = str(f[k])
        return {"status": "success", "facts": facts, "count": len(facts)}
    except Exception as e:
        logger.error(f"Error searching key facts: {e}")
        return {"status": "error", "error": str(e)}


async def get_version_summary(version: str) -> Dict[str, Any]:
    """Get the version-level summary and all model summaries for a version.

    TOOL_NAME=get_version_summary
    DISPLAY_NAME=Get Version Summary
    USECASE=Retrieve cross-model rollup and per-model summaries for a version
    INSTRUCTIONS=Provide a version string
    INPUT_DESCRIPTION=version (str): e.g. "BENCH-2.5"
    OUTPUT_DESCRIPTION=Dictionary with version_summary and model_summaries
    EXAMPLES=get_version_summary("BENCH-2.5")
    PREREQUISITES=Database must be accessible
    RELATED_TOOLS=search_key_facts, get_model_performance_history
    """
    try:
        await _ensure_schema_once()
        vs = await query_version_summaries(rhaiis_version=version)
        ms = await query_model_summaries(rhaiis_version=version)
        for item in vs + ms:
            for k in ("created_at", "updated_at"):
                if k in item and item[k] is not None:
                    item[k] = str(item[k])
        return {
            "status": "success",
            "version": version,
            "version_summaries": vs,
            "model_summaries": ms,
            "total": len(vs) + len(ms),
        }
    except Exception as e:
        logger.error(f"Error getting version summary: {e}")
        return {"status": "error", "error": str(e)}


async def get_model_performance_history(
    model_name: str, last_n_versions: int = 5,
) -> Dict[str, Any]:
    """Get model summaries across recent versions.

    TOOL_NAME=get_model_performance_history
    DISPLAY_NAME=Get Model Performance History
    USECASE=Track model performance trends across versions
    INSTRUCTIONS=Provide a model name (partial match)
    INPUT_DESCRIPTION=model_name (str), last_n_versions (int)
    OUTPUT_DESCRIPTION=Dictionary with per-version model summaries
    EXAMPLES=get_model_performance_history("Helios")
    PREREQUISITES=Database must be accessible
    RELATED_TOOLS=search_key_facts, get_version_summary
    """
    try:
        await _ensure_schema_once()
        summaries = await query_model_summaries(
            model_name=model_name, limit=last_n_versions * 5
        )
        versions: dict[str, list] = {}
        for s in summaries:
            v = s.get("rhaiis_version", "unknown")
            for k in ("created_at", "updated_at"):
                if k in s and s[k] is not None:
                    s[k] = str(s[k])
            versions.setdefault(v, []).append(s)
        sorted_v = sorted(versions.keys(), reverse=True)[:last_n_versions]
        return {
            "status": "success",
            "model": model_name,
            "versions_covered": sorted_v,
            "history": {v: versions[v] for v in sorted_v},
        }
    except Exception as e:
        logger.error(f"Error getting model history: {e}")
        return {"status": "error", "error": str(e)}


async def store_key_facts(
    facts_json: str, run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Store config-level findings.

    TOOL_NAME=store_key_facts
    DISPLAY_NAME=Store Key Facts
    USECASE=Persist performance findings for future analysis runs
    INSTRUCTIONS=Pass findings as a JSON string (array of objects)
    INPUT_DESCRIPTION=facts_json (str): JSON array of finding dicts, run_id (str, optional)
    OUTPUT_DESCRIPTION=Dictionary with per-fact results
    EXAMPLES=store_key_facts(facts_json='[{"text": "...", "category": "metrics_comparison", ...}]')
    PREREQUISITES=Database must be accessible
    RELATED_TOOLS=search_key_facts
    """
    try:
        await _ensure_schema_once()
        import json
        try:
            facts = json.loads(facts_json)
        except (json.JSONDecodeError, TypeError):
            return {"status": "error", "error": "Could not parse facts_json — expected JSON array"}
        if not isinstance(facts, list):
            facts = [facts]
        results, inserted, updated, noop = [], 0, 0, 0
        for f in facts:
            if run_id:
                f["run_id"] = run_id
            if "fact_text" not in f and "text" in f:
                f["fact_text"] = f["text"]
            result = await upsert_config_finding(f)
            results.append(result)
            if result["action"] == "inserted":
                inserted += 1
            elif result["action"] == "updated":
                updated += 1
            else:
                noop += 1
        return {
            "status": "success", "results": results,
            "summary": {"total": len(facts), "inserted": inserted, "updated": updated, "skipped": noop},
        }
    except Exception as e:
        logger.error(f"Error storing key facts: {e}")
        return {"status": "error", "error": str(e)}
