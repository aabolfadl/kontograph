"""Runtime configuration.

Every setting is overridable by a ``KG_``-prefixed environment variable, so the
same image runs locally, in CI and in the compose stack without code changes.
Paths are resolved relative to the repository root, discovered by walking up
from this file — which keeps ``kg`` working regardless of the caller's cwd.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards for the repository marker. Falls back to the cwd."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "uv.lock").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = find_repo_root()


class Settings(BaseSettings):
    """Process-wide configuration, populated from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="KG_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- paths --
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    warehouse_dir: Path = REPO_ROOT / "data" / "warehouse"

    # --------------------------------------------------------- determinism --
    # One seed governs the entire simulated ecosystem. Changing it changes the
    # dataset fingerprint, which changes every downstream artifact's identity.
    seed: int = 20251009  # the day SEPA Instant became mandatory in the euro area

    env: Literal["local", "ci", "docker"] = "local"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # ----------------------------------------------------------- streaming --
    kafka_bootstrap: str = "localhost:19092"
    schema_registry_url: str = "http://localhost:18081"
    topic_payments: str = "kontograph.payments.sctinst.v1"
    topic_labels: str = "kontograph.labels.outcome.v1"
    topic_alerts: str = "kontograph.alerts.v1"
    topic_feedback: str = "kontograph.feedback.analyst.v1"

    # ----------------------------------------------------------- lakehouse --
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "kontograph"
    s3_secret_key: str = "kontograph"
    s3_bucket: str = "kontograph"
    iceberg_catalog_uri: str = ""  # defaults to a sqlite catalog under data/

    # --------------------------------------------------------------- redis --
    redis_url: str = "redis://localhost:6379/0"

    # -------------------------------------------------------------- mlflow --
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "kontograph"

    # ------------------------------------------------------------- serving --
    # Design principle P4 — the SLA is a design input, so it lives in config and is
    # asserted by the load test rather than described in a README.
    sla_p99_millis: int = 200
    alert_threshold: float = 0.5
    analyst_capacity_per_day: int = 200  # the k in precision@k

    # ------------------------------------------------------------- kaggle ---
    kaggle_dataset_slug: str = "kontograph-sctinst"
    kaggle_username: str = ""

    @field_validator("data_dir", "artifacts_dir", "warehouse_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def iceberg_catalog_path(self) -> Path:
        """Location of the sqlite Iceberg catalog (see ADR-0004)."""
        return self.warehouse_dir / "catalog.db"

    def ensure_dirs(self) -> None:
        """Create the writable directories. Safe to call repeatedly."""
        for path in (self.data_dir, self.artifacts_dir, self.warehouse_dir):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so the fingerprint cannot drift mid-run."""
    return Settings()
