"""UI display-name resolution for ResumeMetadata.

This module provides a presentation-layer helper that falls back through
multiple sources without overwriting the canonical ``candidate_name`` field
in ``ResumeMetadata``.  The original extracted name is preserved for future
parser improvements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.resume_parser.name_validator import is_valid_candidate_name, normalize_candidate_name

from .resume_metadata import ResumeMetadata


def get_display_name(
    metadata: ResumeMetadata,
    source_filename: str | None = None,
) -> str:
    """Return the best display name for a candidate.

    Priority:
      1. A validated ``candidate_name`` from metadata extraction.
      2. The source resume filename (without extension).
      3. A stable ``Resume #<resume_id>`` label.

    The canonical ``metadata.candidate_name`` is left unchanged.
    """
    if isinstance(metadata.candidate_name, str) and is_valid_candidate_name(metadata.candidate_name):
        return normalize_candidate_name(metadata.candidate_name)

    if source_filename:
        stem = Path(source_filename).stem
        if is_valid_candidate_name(stem):
            return normalize_candidate_name(stem)

    return f"Resume #{metadata.resume_id}"
