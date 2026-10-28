from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str = Field(default="postgresql+psycopg://postgres:postgres@localhost:5432/ai_soc")
    APP_ENV: str = Field(default="dev")
    STORE_RAW: bool = Field(default=False)
    MAX_UPLOAD_MB: int = Field(default=200)

    LLM_BASE_URL: str = Field(default="http://localhost:1234/v1")
    LLM_MODEL: str = Field(default="local-model")
    LLM_API_KEY: str = Field(default="not-needed")
    LLM_TIMEOUT_SECONDS: int = Field(default=120)

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def executor_type(self) -> str:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return d["executor"]["type"]

    @property
    def default_workers_per_lane(self) -> int:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return d["executor"]["default_workers_per_lane"]

    @property
    def workers_per_lane(self) -> dict:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return d["executor"]["lanes"]

    @property
    def uploads_path(self) -> Path:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return PROJECT_DIR / d["paths"]["uploads"]

    @property
    def models_path(self) -> Path:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return PROJECT_DIR / d["paths"]["models"]

    @property
    def reports_path(self) -> Path:
        d = self._load_yaml(CONFIG_DIR / "settings.yaml")
        return PROJECT_DIR / d["paths"]["reports"]

    @property
    def min_confidence_default(self) -> float:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["filter"]["min_confidence_default"]

    @property
    def dedupe_minutes(self) -> int:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["windows"]["dedupe_minutes"]

    @property
    def multistage_minutes(self) -> int:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["windows"]["multistage_minutes"]

    @property
    def class_weight_boost(self) -> float:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["boosts"]["extra_lane"]

    @property
    def tactic_boost(self) -> float:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["boosts"]["extra_tactic"]

    @property
    def weak_signal_max_confidence(self) -> float:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["weak_signal_max_confidence"]

    @property
    def severity_bands(self) -> dict:
        d = self._load_yaml(CONFIG_DIR / "orchestrator.yaml")
        return d["severity_bands"]

    @property
    def llm_prompt_version(self) -> str:
        d = self._load_yaml(CONFIG_DIR / "llm.yaml")
        return d["prompt_version"]

    @property
    def llm_max_retries(self) -> int:
        d = self._load_yaml(CONFIG_DIR / "llm.yaml")
        return d["max_retries"]

    @property
    def llm_temperature(self) -> float:
        d = self._load_yaml(CONFIG_DIR / "llm.yaml")
        return d["temperature"]

    @property
    def llm_max_incidents_in_prompt(self) -> int:
        d = self._load_yaml(CONFIG_DIR / "llm.yaml")
        return d["max_incidents_in_prompt"]

# Resolve paths relative to this file's project root
PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"

settings = Settings()
