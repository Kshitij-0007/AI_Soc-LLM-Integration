from sqlalchemy.orm import Session

from ..models import Lanes


def get_lane(session: Session, lane_id: str) -> Lanes | None:
    return session.query(Lanes).filter(Lanes.lane_id == lane_id).first()


def get_all_lanes(session: Session) -> list[Lanes]:
    return session.query(Lanes).all()


def upsert_lane(session: Session, lane_id: str, display_name: str) -> Lanes:
    lane = session.query(Lanes).filter(Lanes.lane_id == lane_id).first()
    if lane is None:
        lane = Lanes(lane_id=lane_id, display_name=display_name)
        session.add(lane)
    else:
        lane.display_name = display_name
    return lane
