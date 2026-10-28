from sqlalchemy.orm import Session

from ..models import ModelVersions


def get_active_model_per_lane(session: Session) -> dict:
    rows = session.query(ModelVersions).filter(ModelVersions.is_active).all()
    return {r.lane_id: r for r in rows}


def get_model_by_id(session: Session, model_id: str) -> ModelVersions | None:
    return session.query(ModelVersions).filter(ModelVersions.model_id == model_id).first()


def deactivate_lane_models(session: Session, lane_id: str) -> None:
    session.query(ModelVersions).filter(ModelVersions.lane_id == lane_id).update(
        {"is_active": False}, synchronize_session=False
    )


def activate_model(session: Session, model_id: str) -> ModelVersions:
    deactivate_lane_models(session, model_id)
    model = session.query(ModelVersions).filter(ModelVersions.model_id == model_id).first()
    if model:
        model.is_active = True
    return model


def create_model_version(
    session: Session,
    lane_id: str,
    version: str,
    model_path: str,
    class_labels: dict,
    feature_list: dict,
    train_stats: dict | None = None,
    metrics: dict | None = None,
    data_source: str | None = None,
    is_active: bool = False,
) -> ModelVersions:
    model = ModelVersions(
        lane_id=lane_id,
        version=version,
        model_path=model_path,
        class_labels=class_labels,
        feature_list=feature_list,
        train_stats=train_stats,
        metrics=metrics,
        data_source=data_source,
        is_active=is_active,
    )
    session.add(model)
    if is_active:
        deactivate_lane_models(session, lane_id)
    return model
