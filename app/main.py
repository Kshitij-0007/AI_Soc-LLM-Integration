from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def lifespan(app: FastAPI):
    # Startup
    print("Starting AI-SOC")
    # Check DB, discover lanes, create executors, load models
    # For now, just print
    yield  # App runs here
    # Shutdown
    print("Shutdown complete.")


# Create FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-SOC",
        version="0.1.0",
        description="AI Security Operations Center - Multi-model security analytics",
        lifespan=lifespan,
    )

    # Add CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simple hello world route
    @app.get("/api/health")
    def health():
        return {"db": "ok", "lanes": 3}

    @app.get("/api/lanes")
    def get_lanes():
        return [
            {"lane_id": "netflow", "display_name": "Netflow", "formats": ["csv"], "is_enabled": True},
            {"lane_id": "siem_edr", "display_name": "SIEM/EDR", "formats": ["csv", "jsonl"], "is_enabled": True},
            {"lane_id": "cloud_entity", "display_name": "Cloud Entity", "formats": ["csv", "jsonl"], "is_enabled": True},
        ]

    @app.post("/api/runs")
    def create_run():
        run_id = "mock-run-123"
        return {"run_id": run_id, "jobs": [{"job_id": "j1", "lane_id": "netflow", "filename": "sample.csv", "status": "queued"}]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return {
            "status": "scored",
            "jobs": [
                {"job_id": "j1", "lane_id": "netflow", "filename": "sample.csv", "status": "done", "rows_total": 100, "rows_valid": 95, "rows_rejected": 5, "validation_report": {}},
            ],
        }

    @app.get("/api/runs/{run_id}/results")
    def get_results(
        run_id: str,
        lane: str = "",
        min_conf: float = 0.0,
        limit: int = 100,
    ):
        return {"rows": []}

    @app.get("/api/runs/{run_id}/incidents")
    def get_incidents(run_id: str):
        return {"incidents": []}

    @app.post("/api/runs/{run_id}/report")
    def create_report(run_id: str):
        return {"report_id": "mock-report-123", "status": "pending"}

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str):
        return {"report_json": {}, "generated": None}

    @app.get("/api/models")
    def get_models():
        return {
            "netflow": {"version": "1.0.0", "macro_f1": 0.92, "data_source": "synthetic"},
            "siem_edr": {"version": "1.0.0", "macro_f1": 0.88, "data_source": "synthetic"},
            "cloud_entity": {"version": "1.0.0", "macro_f1": 0.91, "data_source": "synthetic"},
        }

    return app


app = create_app()
