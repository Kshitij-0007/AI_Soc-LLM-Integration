from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.discovery import discover_lanes
from app.db.session import get_db

router = APIRouter(prefix="/api", tags=["lanes"])


@router.get("/lanes")
def get_lanes(db: Session = Depends(get_db)) -> list[dict]:
    """Get all discovered lanes."""
    lanes = discover_lanes()
    result = []
    for lane_id, spec in lanes.items():
        result.append(
            {
                "lane_id": lane_id,
                "display_name": spec.display_name,
                "formats": spec.formats,
                "is_enabled": spec.is_enabled,
            }
        )
    return result
