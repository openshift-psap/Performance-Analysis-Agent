-- Migration: Memory Layer Phase 1 (SQLite version)
-- SQLite does not support ADD COLUMN IF NOT EXISTS, so these use a
-- try-and-ignore approach (run via the migrate.py script).

-- =========================================================================
-- 1. config_findings — add status + issue_thread_id columns
-- =========================================================================

ALTER TABLE config_findings ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE config_findings ADD COLUMN issue_thread_id TEXT;

CREATE INDEX IF NOT EXISTS idx_cf_status ON config_findings (status);
CREATE INDEX IF NOT EXISTS idx_cf_thread ON config_findings (issue_thread_id);


-- =========================================================================
-- 2. model_summaries — add status + issue_thread_id
-- =========================================================================

ALTER TABLE model_summaries ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE model_summaries ADD COLUMN issue_thread_id TEXT;

DROP INDEX IF EXISTS idx_ms_dedup;
CREATE INDEX IF NOT EXISTS idx_ms_dedup ON model_summaries (rhaiis_version, model_name);


-- =========================================================================
-- 3. version_summaries — add status
-- =========================================================================

ALTER TABLE version_summaries ADD COLUMN status TEXT NOT NULL DEFAULT 'active';


-- =========================================================================
-- 4. analysis_reports (new table)
-- =========================================================================

CREATE TABLE IF NOT EXISTS analysis_reports (
    id              TEXT PRIMARY KEY,
    analysis_level  TEXT NOT NULL,
    report_text     TEXT NOT NULL,
    rhaiis_version  TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    model_name      TEXT,
    accelerator     TEXT,
    tp_config       INTEGER,
    profile         TEXT,
    run_id          TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ar_version ON analysis_reports (rhaiis_version);
CREATE INDEX IF NOT EXISTS idx_ar_level ON analysis_reports (analysis_level);
CREATE INDEX IF NOT EXISTS idx_ar_model ON analysis_reports (model_name);


-- =========================================================================
-- 5. consolidation_log (new table)
-- =========================================================================

CREATE TABLE IF NOT EXISTS consolidation_log (
    id              TEXT PRIMARY KEY,
    rhaiis_version  TEXT NOT NULL,
    facts_pruned    INTEGER DEFAULT 0,
    facts_merged    INTEGER DEFAULT 0,
    red_flags_folded INTEGER DEFAULT 0,
    summary_text    TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cl_version ON consolidation_log (rhaiis_version);
