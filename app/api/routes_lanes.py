from fastapi import APIRouter, Request

from app.api.deps import Database
from app.db.repositories.lanes import get_all_lanes

router = APIRouter(prefix="/api", tags=["lanes"])


@router.get("/lanes")
def get_lanes(request: Request, db: Database):
    specs = request.app.state.lanes
    return [
        {
            "lane_id": row.lane_id,
            "display_name": row.display_name,
            "formats": specs[row.lane_id].formats,
            "is_enabled": row.is_enabled,
        }
        for row in get_all_lanes(db)
        if row.lane_id in specs
    ]
