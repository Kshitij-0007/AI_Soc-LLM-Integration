from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models import Lanes


def get_lane(session: Session, lane_id: str) -> Lanes | None:
    return session.get(Lanes, lane_id)


def get_all_lanes(session: Session) -> list[Lanes]:
    return session.query(Lanes).order_by(Lanes.lane_id).all()


def upsert_lane(session: Session, lane_id: str, display_name: str) -> Lanes:
    stmt = insert(Lanes).values(lane_id=lane_id, display_name=display_name)
    return session.scalars(
        stmt.on_conflict_do_update(
            index_elements=[Lanes.lane_id], set_={"display_name": display_name}
        ).returning(Lanes)
    ).one()
