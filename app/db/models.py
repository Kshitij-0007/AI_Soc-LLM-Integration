from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ----- lanes -----
class Lanes(Base):
    __tablename__ = "lanes"
    lane_id = Column(Text, primary_key=True)
    display_name = Column(Text, nullable=False)
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=None)


# ----- taxonomy_map -----
class TaxonomyMap(Base):
    __tablename__ = "taxonomy_map"
    lane_id = Column(Text, primary_key=True)
    native_label = Column(Text, primary_key=True)
    unified_class = Column(Text)
    tactic = Column(Text)
    weight = Column(Numeric(5, 2), default=0)


# ----- runs -----
class Runs(Base):
    __tablename__ = "runs"
    run_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    status = Column(Text, default="queued")
    started_at = Column(DateTime, default=None)
    finished_at = Column(DateTime, nullable=True)
    created_by = Column(Text, nullable=True)


# ----- ingestion_jobs -----
class IngestionJobs(Base):
    __tablename__ = "ingestion_jobs"
    job_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    run_id = Column(UUID, ForeignKey("runs.run_id"), nullable=False)
    lane_id = Column(Text, nullable=False)
    filename = Column(Text, nullable=False)
    status = Column(Text, default="queued")
    rows_total = Column(Integer, default=0)
    rows_valid = Column(Integer, default=0)
    rows_rejected = Column(Integer, default=0)
    validation_report = Column(JSON, default="{}")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=None)
    completed_at = Column(DateTime, nullable=True)


# ----- raw_records -----
class RawRecords(Base):
    __tablename__ = "raw_records"
    record_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    job_id = Column(
        UUID,
        ForeignKey("ingestion_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    raw_payload = Column(JSON, nullable=False)
    source_file = Column(Text, nullable=False)


# ----- rejected_records -----
class RejectedRecords(Base):
    __tablename__ = "rejected_records"
    record_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    job_id = Column(
        UUID,
        ForeignKey("ingestion_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    raw_payload = Column(JSON, nullable=False)
    reason = Column(Text, nullable=False)
    source_file = Column(Text, nullable=False)


# ----- events -----
class Events(Base):
    __tablename__ = "events"
    event_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    job_id = Column(
        UUID,
        ForeignKey("ingestion_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_type = Column(Text, nullable=False)
    entity_value = Column(Text, nullable=False)
    ip = Column(Text)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    event_count = Column(Integer, default=1)


# ----- model_versions -----
class ModelVersions(Base):
    __tablename__ = "model_versions"
    model_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    lane_id = Column(Text, nullable=False)
    version = Column(Text, nullable=False)
    model_path = Column(Text, nullable=False)
    class_labels = Column(JSON, nullable=False)
    feature_list = Column(JSON, nullable=False)
    train_stats = Column(JSON)
    metrics = Column(JSON)
    data_source = Column(Text)
    is_active = Column(Boolean, default=False)
    activated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("lane_id", "is_active", name="uq_one_active_model_per_lane"),
    )


# ----- predictions -----
class Predictions(Base):
    __tablename__ = "predictions"
    prediction_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    event_id = Column(
        UUID,
        ForeignKey("events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id = Column(
        UUID,
        ForeignKey("model_versions.model_id", ondelete="SET NULL"),
    )
    predicted_class = Column(Text, nullable=False)
    confidence = Column(Numeric(5, 2), nullable=False)
    class_probabilities = Column(JSON, nullable=False)
    unified_class = Column(Text)
    tactic = Column(Text)
    shap_top_features = Column(JSON, default="[]")


# ----- incidents -----
class Incidents(Base):
    __tablename__ = "incidents"
    incident_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    run_id = Column(
        UUID,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    severity = Column(Text, nullable=False)
    score = Column(Numeric(5, 2), nullable=False)
    entity = Column(Text, nullable=False)
    tactics = Column(JSON, default="[]")
    lanes = Column(JSON, default="[]")
    flags = Column(JSON, default="[]")
    created_at = Column(DateTime, default=None)


# ----- incident_evidence -----
class IncidentEvidence(Base):
    __tablename__ = "incident_evidence"
    evidence_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    incident_id = Column(
        UUID,
        ForeignKey("incidents.incident_id", ondelete="CASCADE"),
        nullable=False,
    )
    event_id = Column(
        UUID,
        ForeignKey("events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    predicted_class = Column(Text, nullable=False)
    confidence = Column(Numeric(5, 2), nullable=False)
    unified_class = Column(Text)
    tactic = Column(Text)


# ----- reports -----
class Reports(Base):
    __tablename__ = "reports"
    report_id = Column(UUID, primary_key=True, server_default="gen_random_uuid()")
    run_id = Column(
        UUID,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    status = Column(Text, default="pending")
    report_json = Column(JSON, default="{}")
    llm_provider = Column(Text)
    llm_model = Column(Text)
    prompt_version = Column(Text)
    used_fallback = Column(Boolean, default=False)
    generated = Column(DateTime, nullable=True)
    file_path = Column(Text)
