from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_lanes, routes_models, routes_runs
from app.config import settings
from app.core.discovery import discover_lanes
from app.core.taxonomy import load_taxonomy_yaml, upsert_lane_taxonomy
from app.db.repositories.lanes import upsert_lane
from app.db.session import SessionLocal, healthcheck
from app.execution.runner import LaneExecutor


@asynccontextmanager
async def lifespan(app: FastAPI):
    lanes = discover_lanes()
    if settings.ENABLED_LANES is not None:
        missing = set(settings.ENABLED_LANES) - lanes.keys()
        if missing:
            raise RuntimeError(f"Unknown configured lanes: {sorted(missing)}")
        lanes = {key: lanes[key] for key in settings.ENABLED_LANES}
    with SessionLocal.begin() as session:
        for lane_id, spec in lanes.items():
            upsert_lane(session, lane_id, spec.display_name)
            upsert_lane_taxonomy(session, lane_id, load_taxonomy_yaml(lane_id))
    app.state.lanes = lanes
    app.state.pools = {key: LaneExecutor(key) for key in lanes}
    try:
        yield
    finally:
        for pool in app.state.pools.values():
            pool.shutdown()


def create_app():
    app = FastAPI(title="AI-SOC", version="0.1.0", lifespan=lifespan)
    if settings.APP_ENV == "dev":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    for router in (routes_lanes.router, routes_runs.router, routes_models.router):
        app.include_router(router)

    @app.get("/api/health")
    def health(request: Request):
        if not healthcheck()["ok"]:
            raise HTTPException(503, "Database unavailable")
        return {"db": "ok", "lanes": len(request.app.state.lanes)}

    @app.post("/api/runs/{run_id}/report")
    @app.get("/api/reports/{report_id}")
    def report_unavailable():
        raise HTTPException(503, "Report generation is not configured")

    return app


app = create_app()
