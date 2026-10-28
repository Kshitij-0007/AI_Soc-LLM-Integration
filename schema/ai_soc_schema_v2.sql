-- ai_soc_schema_v2.sql
-- Authoritative DDL for the AI-SOC platform (12 tables)
-- Run with: python scripts/reset_db.py --yes or psql -f schema/ai_soc_schema_v2.sql

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pg_catalog"."uuid-ossp";

-- Enum types
CREATE TYPE IF NOT EXISTS lane_id_enum AS TEXT;
CREATE TYPE IF NOT EXISTS run_status_enum AS TEXT;
CREATE TYPE IF NOT EXISTS job_status_enum AS TEXT;
CREATE TYPE IF NOT EXISTS incident_status_enum AS TEXT;
CREATE TYPE IF NOT EXISTS severity_enum AS TEXT;
CREATE TYPE IF NOT EXISTS unified_class_enum AS TEXT;
CREATE TYPE IF NOT EXISTS predicted_class_enum AS TEXT;
CREATE TYPE IF NOT EXISTS tactic_enum AS TEXT;

-- lanes table
CREATE TABLE IF NOT EXISTS lanes (
    lane_id lane_id_enum PRIMARY KEY,
    display_name TEXT NOT NULL,
    is_enabled boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

-- taxonomy_map table
CREATE TABLE IF NOT EXISTS taxonomy_map (
    lane_id lane_id_enum,
    native_label text NOT NULL,
    unified_class text,
    tactic text,
    weight numeric DEFAULT 0,
    PRIMARY KEY (lane_id, native_label)
);

-- runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status run_status_enum DEFAULT 'queued',
    started_at timestamptz DEFAULT now(),
    finished_at timestamptz NULL,
    created_by text NULL
);

-- ingestion_jobs table
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES runs(run_id) ON DELETE CASCADE,
    lane_id lane_id_enum NOT NULL,
    filename text NOT NULL,
    status job_status_enum DEFAULT 'queued',
    rows_total integer DEFAULT 0,
    rows_valid integer DEFAULT 0,
    rows_rejected integer DEFAULT 0,
    validation_report jsonb DEFAULT '{}',
    error text NULL,
    created_at timestamptz DEFAULT now(),
    completed_at timestamptz NULL
);

-- raw_records table
CREATE TABLE IF NOT EXISTS raw_records (
    record_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    raw_payload jsonb NOT NULL,
    source_file text NOT NULL
);

-- rejected_records table
CREATE TABLE IF NOT EXISTS rejected_records (
    record_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    raw_payload jsonb NOT NULL,
    reason text NOT NULL,
    source_file text NOT NULL
);

-- events table
CREATE TABLE IF NOT EXISTS events (
    event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    entity_type text NOT NULL,
    entity_value text NOT NULL,
    ip inet,
    window_start timestamptz NOT NULL,
    window_end timestamptz NOT NULL,
    event_count integer DEFAULT 1
);

-- model_versions table
CREATE TABLE IF NOT EXISTS model_versions (
    model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    lane_id lane_id_enum NOT NULL,
    version text NOT NULL,
    model_path text NOT NULL,
    class_labels jsonb NOT NULL,
    feature_list jsonb NOT NULL,
    train_stats jsonb,
    metrics jsonb,
    data_source text NULL,
    is_active boolean DEFAULT false,
    activated_at timestamptz NULL,
    UNIQUE (lane_id, is_active) WHERE is_active = true
);

-- predictions table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id uuid REFERENCES events(event_id) ON DELETE CASCADE,
    model_id uuid REFERENCES model_versions(model_id) ON DELETE SET NULL,
    predicted_class text NOT NULL,
    confidence numeric NOT NULL,
    class_probabilities jsonb NOT NULL,
    unified_class text,
    tactic text,
    shap_top_features jsonb DEFAULT '[]'
);

-- incidents table
CREATE TABLE IF NOT EXISTS incidents (
    incident_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES runs(run_id) ON DELETE CASCADE,
    severity severity_enum NOT NULL,
    score numeric NOT NULL,
    entity text NOT NULL,
    tactics text[] DEFAULT '{}',
    lanes text[] DEFAULT '{}',
    flags text[] DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);

-- incident_evidence table
CREATE TABLE IF NOT EXISTS incident_evidence (
    evidence_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id uuid REFERENCES incidents(incident_id) ON DELETE CASCADE,
    event_id uuid REFERENCES events(event_id) ON DELETE CASCADE,
    predicted_class text NOT NULL,
    confidence numeric NOT NULL,
    unified_class text,
    tactic text
);

-- reports table
CREATE TABLE IF NOT EXISTS reports (
    report_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES runs(run_id) ON DELETE CASCADE,
    status text DEFAULT 'pending',
    report_json jsonb DEFAULT '{}',
    llm_provider text NULL,
    llm_model text NULL,
    prompt_version text NULL,
    used_fallback boolean DEFAULT false,
    generated timestamptz NULL,
    file_path text NULL
);

-- schema_migrations table (for optional additive changes)
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename text PRIMARY KEY,
    applied_at timestamptz DEFAULT now()
);

-- Index for one active model per lane
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_model_per_lane ON model_versions(lane_id) WHERE is_active = true;