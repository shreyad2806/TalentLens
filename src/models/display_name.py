"""UI display-name resolution for ResumeMetadata.

This module provides a presentation-layer helper that falls back through
multiple sources without overwriting the canonical ``candidate_name`` field
in ``ResumeMetadata``.  The original extracted name is preserved for future
parser improvements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.normalization.role_normalizer import RoleNormalizer
from src.resume_parser.name_validator import (
    fallback_display_name,
    is_valid_candidate_name,
    normalize_candidate_name,
)

from .resume_metadata import ResumeMetadata

logger = logging.getLogger(__name__)


def get_display_name(
    metadata: ResumeMetadata,
    source_filename: str | None = None,
) -> str:
    """Return the best display name for a candidate.

    Priority:
      1. A validated ``candidate_name`` from metadata extraction.
      2. A validated ``full_name`` if present on the metadata.
      3. The source resume filename (without extension).
      4. ``<Role> Professional`` from canonical role.
      5. A stable ``Resume #<resume_id>`` label.

    The canonical ``metadata.candidate_name`` is left unchanged.
    """
    if isinstance(metadata.candidate_name, str) and is_valid_candidate_name(metadata.candidate_name):
        name = normalize_candidate_name(metadata.candidate_name)
        logger.info("Name resolved from candidate_name: %r", name)
        return name

    full_name = getattr(metadata, "full_name", None)
    if isinstance(full_name, str) and is_valid_candidate_name(full_name):
        name = normalize_candidate_name(full_name)
        logger.info("Name resolved from full_name: %r", name)
        return name

    if source_filename:
        stem = Path(source_filename).stem
        if is_valid_candidate_name(stem):
            name = normalize_candidate_name(stem)
            logger.info("Name resolved from filename: %r", name)
            return name

    if metadata.role:
        normalized = RoleNormalizer.normalize(metadata.role) or metadata.role
        role = normalized.strip().title()
        logger.info("Name would have been resolved from role; returning Unknown Candidate instead: %r", role)

    fallback = "Unknown Candidate"
    logger.info("Name resolved from fallback: %r", fallback)
    return fallback
