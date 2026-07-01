"""Structured logging.

Console-friendly locally, JSON in the container — because a decision log that a
human greps and a decision log that Loki ingests have different shapes, and
neither should be the other's compromise.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Idempotently configure structlog + stdlib logging."""
    global _CONFIGURED
    from kg_core.config import get_settings

    settings = get_settings()
    level = (level or settings.log_level).upper()
    fmt = fmt or settings.log_format

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    for noisy in ("botocore", "urllib3", "s3transfer", "asyncio", "kafka"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level)))

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
