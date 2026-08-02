"""
TalentLens Debug Logger — Production-style structured logging helper.

Provides consistent, banner-style logging for every pipeline stage using the
standard library `logging` module.  Full exception tracebacks are emitted only
when `DEBUG=1` or the `LOG_LEVEL` environment variable is set to `DEBUG`.

Usage:
    from src.debug_logger import log_stage_start, log_stage_end, log_error, StageTimer

    log_stage_start(3, "EMBEDDING", query="java developer", model="bge-small")
    ... do work ...
    log_stage_end(3, "EMBEDDING", status="SUCCESS", time_ms=34, output_count=1,
                  sample={"shape": "(384,)", "first_values": [...]})

Or use the context-manager style:
    with StageTimer(3, "EMBEDDING", query="java developer") as t:
        result = embed(query)
        t.set_output(count=1, sample={"shape": "(384,)"})
    # END banner logged automatically on exit
"""

import logging
import os
import re
import sys
import time
from typing import Any

# ── Logger setup ─────────────────────────────────────────────────────────────

# Default to INFO so stage banners and warnings are visible in production.
_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
_DEBUG = _LOG_LEVEL == logging.DEBUG or os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

logger = logging.getLogger("talentlens")
logger.setLevel(logging.DEBUG)  # let the handlers decide

# Keep a single stream handler pointed at stdout.  The formatter uses standard
# key-value pairs so logs are machine-readable and safe for production.
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(_LOG_LEVEL)
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


# ── Sensitive-data redaction ─────────────────────────────────────────────────

# Redact values whose keys look like they could be a credential.  This applies
# to the arbitrary key/value pairs passed to log_stage_start/log_stage_end so
# that no API key, token, or password is accidentally written to logs.
_SENSITIVE_PATTERN = re.compile(
    r"(api_?key|password|passwd|secret|token|credential|auth|private_key|"
    r"access_key|openai_api_key|pinecone_api_key|qdrant_api_key)",
    re.IGNORECASE,
)


def _redact_value(key: str, value: Any) -> Any:
    """Return a redacted placeholder for values that look sensitive."""
    if _SENSITIVE_PATTERN.search(key):
        if value is None:
            return None
        if isinstance(value, bool):
            return "***REDACTED***" if value else value
        return "***REDACTED***"
    return value


# ── Core Helpers ─────────────────────────────────────────────────────────────

_SEP = "=" * 60


def _format_value(value: Any, max_len: int = 120) -> str:
    """Format a value for display, truncating long representations."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        if len(value) > 5:
            preview = ", ".join(str(v) for v in value[:5])
            return f"[{preview}, ... ({len(value)} items)]"
        return str(value)
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _format_fields(fields: dict[str, Any], indent: int = 2) -> str:
    """Format key=value pairs with sensitive values redacted."""
    if not fields:
        return ""
    spaces = " " * indent
    lines = []
    for key, value in fields.items():
        safe_value = _redact_value(key, value)
        formatted = _format_value(safe_value)
        lines.append(f"{spaces}{key:<30} = {formatted}")
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

def log_stage_start(
    stage_num: int,
    stage_name: str,
    **fields: Any,
) -> None:
    """
    Log a START banner for a pipeline stage.

    Args:
        stage_num: Stage number (1-12)
        stage_name: Human-readable stage name (e.g. "EMBEDDING")
        **fields: Arbitrary key=value pairs to display as input
    """
    message = (
        f"\n{_SEP}\n"
        f"STAGE {stage_num} — {stage_name}  [START]\n"
        f"{_SEP}"
    )
    if fields:
        message += "\nInput:\n" + _format_fields(fields)
    message += "\n"
    logger.info(message)


def log_stage_end(
    stage_num: int,
    stage_name: str,
    status: str = "SUCCESS",
    time_ms: float | None = None,
    output_count: int | None = None,
    sample: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Log an END banner for a pipeline stage.

    Args:
        stage_num: Stage number
        stage_name: Stage name
        status: "SUCCESS" | "WARNING" | "FAILED"
        time_ms: Execution time in milliseconds
        output_count: Number of output items
        sample: Sample output object to display
        extra: Additional key=value pairs
    """
    level = logging.INFO
    if status == "WARNING":
        level = logging.WARNING
    elif status == "FAILED":
        level = logging.ERROR

    lines = [
        _SEP,
        f"STAGE {stage_num} — {stage_name}  [END]",
        _SEP,
        f"  {'Status':<30} = {status}",
    ]

    if time_ms is not None:
        lines.append(f"  {'Time':<30} = {time_ms:.1f} ms")

    if output_count is not None:
        lines.append(f"  {'Output Count':<30} = {output_count}")

    if sample is not None:
        lines.append("  Sample:")
        if isinstance(sample, dict):
            for k, v in sample.items():
                safe_v = _redact_value(k, v)
                lines.append(f"    {k:<28} = {_format_value(safe_v)}")
        else:
            lines.append(f"    {_format_value(sample)}")

    if extra:
        lines.append(_format_fields(extra, indent=2))

    lines.extend([_SEP, ""])
    logger.log(level, "\n" + "\n".join(lines))


def log_error(
    stage_num: int,
    stage_name: str,
    error: BaseException,
    reraise: bool = True,
) -> None:
    """
    Log a full error trace for a stage, then optionally re-raise.

    Only the exception message is emitted at ERROR level.  The full traceback
    is sent at DEBUG level so it is available when `LOG_LEVEL=DEBUG` or
    `DEBUG=1`, but never printed to production logs by default.

    Args:
        stage_num: Stage number
        stage_name: Stage name
        error: The exception that occurred
        reraise: Whether to re-raise after logging (default True)
    """
    logger.error(
        f"\n{_SEP}\n"
        f"STAGE {stage_num} — {stage_name}  [FAILED]\n"
        f"{_SEP}\n"
        f"  Error: {error}\n"
        f"{_SEP}"
    )

    if _DEBUG and sys.exc_info()[0] is not None:
        # Full internal traceback is only emitted in DEBUG mode and when an
        # active exception is available.
        logger.debug("Full traceback:", exc_info=True)
    else:
        # In production, log only the exception type for diagnostics.
        logger.info(f"Exception type: {type(error).__name__}")

    if reraise:
        raise error


# ── Context Manager ──────────────────────────────────────────────────────────

class StageTimer:
    """
    Context manager that wraps a pipeline stage with START/END banners.

    Usage:
        with StageTimer(3, "EMBEDDING", query="java developer") as t:
            result = embed(query)
            t.set_output(count=1, sample={"shape": "(384,)"})
        # END banner logged automatically on exit
    """

    def __init__(self, stage_num: int, stage_name: str, **input_fields: Any):
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.input_fields = input_fields
        self._start_time: float = 0.0
        self._output_count: int | None = None
        self._sample: Any = None
        self._extra: dict[str, Any] = {}
        self._status: str = "SUCCESS"

    def __enter__(self) -> "StageTimer":
        log_stage_start(self.stage_num, self.stage_name, **self.input_fields)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        if exc_type is not None:
            self._status = "FAILED"
            log_error(self.stage_num, self.stage_name, exc_val, reraise=True)
            return False  # do not suppress exception

        log_stage_end(
            self.stage_num,
            self.stage_name,
            status=self._status,
            time_ms=elapsed_ms,
            output_count=self._output_count,
            sample=self._sample,
            extra=self._extra if self._extra else None,
        )
        return False

    def set_output(
        self,
        count: int | None = None,
        sample: Any = None,
        status: str = "SUCCESS",
        **extra: Any,
    ) -> None:
        """Set output info to display in the END banner."""
        if count is not None:
            self._output_count = count
        if sample is not None:
            self._sample = sample
        if extra:
            self._extra.update(extra)
        self._status = status
