"""
TalentLens Debug Logger — Production-style observability helper.

Provides consistent, colored, banner-style logging for every pipeline stage.
Every stage prints:
  - Stage name & number
  - Input summary
  - Output summary
  - Count
  - Sample object
  - Execution time
  - Status (SUCCESS / WARNING / FAILED)

Usage:
    from src.debug_logger import log_stage_start, log_stage_end, log_error, StageTimer

    log_stage_start(3, "EMBEDDING", query="java developer", model="bge-small")
    ... do work ...
    log_stage_end(3, "EMBEDDING", status="SUCCESS", time_ms=34, output_count=1,
                  sample={"shape": "(384,)", "first_values": [...]})

Or use the context-manager style:
    with StageTimer(3, "EMBEDDING", query="java developer") as t:
        ... do work ...
        t.set_output(count=1, sample={"shape": "(384,)"})
    # automatically prints START + END banners
"""

import sys
import os
import time
import traceback
from typing import Any, Dict, Optional

# ── ANSI Colors ──────────────────────────────────────────────────────────────

_COLOR_SUPPORT = (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
) or os.getenv("FORCE_COLOR") == "1"

if _COLOR_SUPPORT:
    _BLUE   = "\033[94m"
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _RED    = "\033[91m"
    _CYAN   = "\033[96m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"
else:
    _BLUE = _GREEN = _YELLOW = _RED = _CYAN = _BOLD = _RESET = ""


def _c(color: str, text: str) -> str:
    """Wrap text in ANSI color if supported."""
    if not color:
        return text
    return f"{color}{text}{_RESET}"


# ── Core Helpers ─────────────────────────────────────────────────────────────

_SEP = "=" * 60


def log_stage_start(
    stage_num: int,
    stage_name: str,
    **fields: Any,
) -> None:
    """
    Print a START banner for a pipeline stage.

    Args:
        stage_num: Stage number (1-12)
        stage_name: Human-readable stage name (e.g. "EMBEDDING")
        **fields: Arbitrary key=value pairs to display as input
    """
    print()
    print(_c(_BLUE, _SEP))
    print(_c(_BLUE, f"STAGE {stage_num} — {stage_name}  [START]"))
    print(_c(_BLUE, _SEP))

    if fields:
        print(_c(_CYAN, "Input:"))
        for key, value in fields.items():
            formatted = _format_value(value)
            print(f"  {key:<30} = {formatted}")
    print()


def log_stage_end(
    stage_num: int,
    stage_name: str,
    status: str = "SUCCESS",
    time_ms: Optional[float] = None,
    output_count: Optional[int] = None,
    sample: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print an END banner for a pipeline stage.

    Args:
        stage_num: Stage number (1-12)
        stage_name: Human-readable stage name
        status: "SUCCESS" | "WARNING" | "FAILED"
        time_ms: Execution time in milliseconds
        output_count: Number of output items
        sample: Sample output object to display
        extra: Additional key=value pairs
    """
    status_color = {"SUCCESS": _GREEN, "WARNING": _YELLOW, "FAILED": _RED}.get(status, _RESET)

    print(_c(_BLUE, _SEP))
    print(_c(_BLUE, f"STAGE {stage_num} — {stage_name}  [END]"))
    print(_c(_BLUE, _SEP))

    print(f"  {'Status':<30} = {_c(status_color, status)}")

    if time_ms is not None:
        print(f"  {'Time':<30} = {time_ms:.1f} ms")

    if output_count is not None:
        print(f"  {'Output Count':<30} = {output_count}")

    if sample is not None:
        print(_c(_CYAN, "  Sample:"))
        if isinstance(sample, dict):
            for k, v in sample.items():
                print(f"    {k:<28} = {_format_value(v)}")
        else:
            print(f"    {_format_value(sample)}")

    if extra:
        for k, v in extra.items():
            print(f"  {k:<30} = {_format_value(v)}")

    print(_c(_BLUE, _SEP))
    print()


def log_error(
    stage_num: int,
    stage_name: str,
    error: BaseException,
    reraise: bool = True,
) -> None:
    """
    Print a full error trace for a stage, then optionally re-raise.

    Args:
        stage_num: Stage number
        stage_name: Stage name
        error: The exception that occurred
        reraise: Whether to re-raise after logging (default True)
    """
    print()
    print(_c(_RED, _SEP))
    print(_c(_RED, f"STAGE {stage_num} — {stage_name}  [FAILED]"))
    print(_c(_RED, _SEP))
    print(_c(_RED, f"  Error: {error}"))
    print()
    print(_c(_RED, "  Traceback:"))
    traceback.print_exc()
    print(_c(_RED, _SEP))
    print()

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
        # END banner printed automatically on exit
    """

    def __init__(self, stage_num: int, stage_name: str, **input_fields: Any):
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.input_fields = input_fields
        self._start_time: float = 0.0
        self._output_count: Optional[int] = None
        self._sample: Any = None
        self._extra: Dict[str, Any] = {}
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
        count: Optional[int] = None,
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


# ── Formatting ───────────────────────────────────────────────────────────────

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
