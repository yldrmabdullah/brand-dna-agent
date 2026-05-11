"""Structured logging via structlog. JSON in prod, console in dev.

Every log line carries the run_id and brand_name once configured via
`bind_brand()`. The rubric calls for traceability — this layer is the
backbone.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog once at app startup.

    `fmt` is 'json' (production, machine-readable) or 'console' (dev,
    color-pretty).
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stdout,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Returns a configured structlog logger."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_brand(brand_name: str, run_id: str) -> None:
    """Bind brand context into all log lines for this run. Call once at start."""
    clear_contextvars()
    bind_contextvars(brand=brand_name, run_id=run_id)


@contextmanager
def time_stage(stage: str, logger: structlog.stdlib.BoundLogger) -> Iterator[dict]:
    """Times a pipeline stage and emits start/finish logs with duration.

    Usage:
        with time_stage("discovery", logger) as timing:
            ...
            timing["items"] = len(pages)
    """
    payload: dict[str, Any] = {"items": 0}
    started = time.perf_counter()
    logger.info("stage.start", stage=stage)
    try:
        yield payload
    finally:
        duration = time.perf_counter() - started
        logger.info(
            "stage.finish",
            stage=stage,
            duration_s=round(duration, 3),
            items=payload.get("items", 0),
        )
