from fastapi import APIRouter, Request

from app.api.deps import Database

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def get_active_models(request: Request, db: Database) -> dict:
    """Get active model per lane with version and metrics."""
    from app.db.repositories.model_versions import get_active_model_per_lane

    active = get_active_model_per_lane(db)

    result = {}
    for lane_id, model in active.items():
        if lane_id not in request.app.state.lanes:
            continue
        result[lane_id] = {
            "version": model.version,
            "model_path": model.model_path,
            "class_labels": model.class_labels,
            "metrics": model.metrics,
            "data_source": model.data_source,
        }
    return result
