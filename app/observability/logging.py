"""Structured logging for the whole service.

``configure_logging`` is called once when the application is created. After that:

* Every log line is a JSON object in production and a coloured, readable line in development,
  chosen by ``Settings.log_as_json``.
* Values bound with ``structlog.contextvars.bind_contextvars`` (the request id, for example)
  appear on every line logged during that request without being passed through function
  arguments.
* Standard-library loggers used by uvicorn, httpx and LangChain are routed through the same
  pipeline, so third-party output has the same shape as ours.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import orjson
import structlog


def _orjson_dumps(value: Any, **_: Any) -> str:
    """Serialise a log record with orjson; anything exotic falls back to ``str``."""
    return orjson.dumps(value, default=str).decode()


def configure_logging(level: str = "INFO", as_json: bool = False) -> None:
    """Install the structlog pipeline on the root logger.

    Args:
        level: Minimum level name, for example ``"INFO"``.
        as_json: Emit one JSON object per line instead of coloured console output.
    """
    min_level = logging.getLevelNamesMapping()[level.upper()]

    # Processors that run for both structlog and stdlib log calls.
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # ConsoleRenderer prints exceptions itself; JSONRenderer needs them pre-formatted.
    final: list[structlog.types.Processor] = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]
    if as_json:
        final += [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=_orjson_dumps),
        ]
    else:
        final.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processors=final)
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(min_level)

    # Our middleware writes one access line per request; uvicorn's would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True
