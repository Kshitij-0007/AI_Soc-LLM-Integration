from sqlalchemy.orm import Session

from ..models import TaxonomyMap


def get_taxonomy_map(session: Session, lane_id: str) -> list[TaxonomyMap]:
    return (
        session.query(TaxonomyMap)
        .filter(TaxonomyMap.lane_id == lane_id)
        .order_by(TaxonomyMap.native_label)
        .all()
    )


def upsert_taxonomy_map(
    session: Session,
    lane_id: str,
    taxonomy_entries: list[dict],
) -> None:
    for entry in taxonomy_entries:
        obj = (
            session.query(TaxonomyMap)
            .filter_by(lane_id=lane_id, native_label=entry["native_label"])
            .first()
        )
        if obj is None:
            obj = TaxonomyMap(
                lane_id=lane_id,
                native_label=entry["native_label"],
                unified_class=entry.get("unified_class"),
                tactic=entry.get("tactic"),
                weight=entry.get("weight", 0.0),
            )
            session.add(obj)
        else:
            if "unified_class" in entry:
                obj.unified_class = entry["unified_class"]
            if "tactic" in entry:
                obj.tactic = entry["tactic"]
            if "weight" in entry:
                obj.weight = entry["weight"]
