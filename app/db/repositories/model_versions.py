from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..models import ModelVersions


def get_active_model_per_lane(session: Session) -> dict:
    rows = session.query(ModelVersions).filter(ModelVersions.is_active).all()
    return {r.lane_id: r for r in rows}


def get_model_by_id(session: Session, model_id: str) -> ModelVersions | None:
    return session.query(ModelVersions).filter(ModelVersions.model_id == model_id).first()


def get_model_by_version(
    session: Session, lane_id: str, version: str
) -> ModelVersions | None:
    return (
        session.query(ModelVersions)
        .filter(ModelVersions.lane_id == lane_id, ModelVersions.version == version)
        .first()
    )


def list_models(session: Session, lane_id: str | None = None) -> list[ModelVersions]:
    query = session.query(ModelVersions)
    if lane_id:
        query = query.filter(ModelVersions.lane_id == lane_id)
    return query.order_by(ModelVersions.lane_id, ModelVersions.version).all()


def deactivate_lane_models(session: Session, lane_id: str) -> None:
    session.query(ModelVersions).filter(
        ModelVersions.lane_id == lane_id, ModelVersions.is_active
    ).update({"is_active": False}, synchronize_session=False)
    # Flush so the partial unique index sees the old row inactive first.
    session.flush()


def activate_model(session: Session, model_id: str) -> ModelVersions:
    """Make one model the lane's active model, in the caller's transaction."""
    model = get_model_by_id(session, model_id)
    if model is None:
        raise ValueError(f"Model {model_id} not found")
    deactivate_lane_models(session, model.lane_id)
    model.is_active = True
    model.activated_at = datetime.now(UTC)
    session.flush()
    return model


def create_model_version(
    session: Session,
    lane_id: str,
    version: str,
    model_path: str,
    class_labels: list,
    feature_list: list,
    train_stats: dict | None = None,
    metrics: dict | None = None,
    data_source: str | None = None,
) -> ModelVersions:
    """Insert an inactive model row. Use activate_model to make it live."""
    model = ModelVersions(
        lane_id=lane_id,
        version=version,
        model_path=model_path,
        class_labels=class_labels,
        feature_list=feature_list,
        train_stats=train_stats,
        metrics=metrics,
        data_source=data_source,
        is_active=False,
    )
    session.add(model)
    session.flush()
    return model
