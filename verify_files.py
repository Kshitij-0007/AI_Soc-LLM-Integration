import os

base = r"C:\ai_soc"
key_files = [
    "AGENTS.md",
    "app/main.py",
    "app/core/contracts.py",
    "app/core/lane_spec.py",
    "app/core/validator.py",
    "app/core/training.py",
    "app/core/job_service.py",
    "app/core/finalizer.py",
    "app/core/discovery.py",
    "app/execution/runner.py",
    "app/api/routes_lanes.py",
    "app/api/routes_runs.py",
    "app/api/routes_reports.py",
    "app/api/routes_models.py",
    "app/lanes/netflow/spec.yaml",
    "app/lanes/netflow/parser.py",
    "app/lanes/netflow/features.py",
    "app/lanes/netflow/train.py",
    "app/lanes/netflow/taxonomy.yaml",
    "app/lanes/siem_edr/spec.yaml",
    "app/lanes/siem_edr/parser.py",
    "app/lanes/siem_edr/features.py",
    "app/lanes/siem_edr/train.py",
    "app/lanes/siem_edr/taxonomy.yaml",
    "app/lanes/cloud_entity/spec.yaml",
    "app/lanes/cloud_entity/parser.py",
    "app/lanes/cloud_entity/features.py",
    "app/lanes/cloud_entity/train.py",
    "app/lanes/cloud_entity/taxonomy.yaml",
    "scripts/reset_db.py",
    "scripts/init_db.py",
    "scripts/make_demo_data.py",
    "scripts/smoke_test.py",
    "knowledge/attack_snippets.yaml",
    "configs/settings.yaml",
    "configs/orchestrator.yaml",
    "configs/llm.yaml",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
]

all_ok = True
for f in key_files:
    full = os.path.join(base, f)
    exists = os.path.exists(full)
    if not exists:
        all_ok = False
    print(f"{'OK' if exists else 'MISSING'}  {f}")

print(f"\nAll files present: {all_ok}")
