"""Structured JSON logging utility with ISO-8601 timestamps and payload support."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config import settings


class StructuredJSONFormatter(logging.Formatter):
    """Formats Python log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        payload = getattr(record, "payload", None)
        if isinstance(payload, dict) and payload:
            log_object["payload"] = payload

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object, ensure_ascii=False, default=str)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a configured structured JSON logger."""
    logger = logging.getLogger(name)
    resolved_level = getattr(
        logging, (level or settings.log_level).upper(), logging.INFO
    )
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(resolved_level)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
