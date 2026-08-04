"""UI display-name resolution for ResumeMetadata.

This module provides a presentation-layer helper that falls back through
multiple sources without overwriting the canonical ``candidate_name`` field
in ``ResumeMetadata``.  The original extracted name is preserved for future
parser improvements.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.resume_parser.name_validator import (
    INVALID_CANDIDATE_NAMES,
    fallback_display_name,
    is_valid_candidate_name,
    normalize_candidate_name,
)

from .resume_metadata import ResumeMetadata

logger = logging.getLogger(__name__)


_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b")
_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/([a-zA-Z0-9._-]+)", re.IGNORECASE)


def _extract_name_from_header(text: str) -> str | None:
    """Look for a valid full name in the first 20 lines of the resume text."""
    if not text:
        return None
    for line in text.splitlines()[:10]:
        line = line.strip()
        if line and is_valid_candidate_name(line):
            return normalize_candidate_name(line)
    return None


def _extract_name_from_email_context(text: str, email: str | None) -> str | None:
    """Return a valid-looking line next to the candidate's email address."""
    if not text or not email or "@" not in email:
        return None
    local = email.split("@")[0].lower()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if local in low or email.lower() in low:
            for j in (i - 1, i + 1):
                if 0 <= j < len(lines):
                    candidate = lines[j].strip()
                    if candidate and is_valid_candidate_name(candidate):
                        return normalize_candidate_name(candidate)
    return None


def _extract_name_from_email_local(email: str | None) -> str | None:
    """Derive a name from the local part of an email (e.g., john.doe@x -> John Doe)."""
    if not email or "@" not in email:
        return None
    local = email.split("@")[0]
    parts = [p for p in re.split(r"[._\d]", local) if p and len(p) > 1]
    if len(parts) < 2:
        return None
    name = " ".join(p.capitalize() for p in parts[:4])
    if is_valid_candidate_name(name):
        return name
    return None


def _extract_name_from_linkedin(linkedin: str | None) -> str | None:
    """Extract a name from a LinkedIn URL slug (e.g., linkedin.com/in/john-doe)."""
    if not linkedin:
        return None
    m = _LINKEDIN_RE.search(linkedin)
    if not m:
        return None
    slug = m.group(1)
    parts = [p for p in re.split(r"[-_.]", slug) if p and len(p) > 1 and not p.isdigit()]
    if len(parts) < 2:
        return None
    name = " ".join(p.capitalize() for p in parts[:4])
    if is_valid_candidate_name(name):
        return name
    return None


def _extract_name_from_largest_heading(text: str) -> str | None:
    """Use the largest, valid proper-name near the top of the resume."""
    if not text:
        return None
    candidate: str | None = None
    candidate_len = 0
    for line in text.splitlines()[:50]:
        line = line.strip()
        if not line or len(line) > 60 or len(line) < 4:
            continue
        if line.lower() in INVALID_CANDIDATE_NAMES:
            continue
        if not is_valid_candidate_name(line):
            continue
        if len(line) > candidate_len:
            candidate = line
            candidate_len = len(line)
    return normalize_candidate_name(candidate) if candidate else None


def get_display_name(
    metadata: ResumeMetadata,
    source_filename: str | None = None,
    resume_text: str | None = None,
) -> str:
    """Return the best display name for a candidate.

    Priority:
      1. A validated ``candidate_name`` from metadata.
      2. A validated ``full_name`` from metadata.
      3. A full name from the resume header.
      4. A name associated with the email address.
      5. The source resume filename (without extension).
      6. A name from the LinkedIn profile URL.
      7. The largest proper-name near the top of the resume.
      8. A stable ``Resume #<resume_id>`` label.

    The canonical ``metadata.candidate_name`` is left unchanged, but the
    resolved confidence and source are stored on ``metadata`` for the UI.
    """

    def _return(name: str, confidence: float, source: str) -> str:
        name = normalize_candidate_name(name)
        metadata.name_confidence = round(confidence, 3)
        metadata.name_source = source
        logger.info("Name resolved from %s (conf=%.2f): %r", source, confidence, name)
        return name

    if isinstance(metadata.candidate_name, str) and is_valid_candidate_name(metadata.candidate_name):
        return _return(metadata.candidate_name, 1.0, "candidate_name")

    full_name = getattr(metadata, "full_name", None)
    if isinstance(full_name, str) and is_valid_candidate_name(full_name):
        return _return(full_name, 0.95, "full_name")

    if resume_text:
        header = _extract_name_from_header(resume_text)
        if header:
            return _return(header, 0.85, "resume_header")

    if metadata.email:
        email_name = _extract_name_from_email_context(resume_text or "", metadata.email) or _extract_name_from_email_local(metadata.email)
        if email_name:
            return _return(email_name, 0.6, "email")

    if source_filename:
        filename_name = display_name_from_filename(source_filename)
        if filename_name:
            return _return(filename_name, 0.45, "filename")

    if metadata.linkedin:
        linkedin_name = _extract_name_from_linkedin(metadata.linkedin)
        if linkedin_name:
            return _return(linkedin_name, 0.5, "linkedin")

    if resume_text:
        largest = _extract_name_from_largest_heading(resume_text)
        if largest:
            return _return(largest, 0.3, "largest_heading")

    fallback = fallback_display_name(metadata.resume_id)
    metadata.name_confidence = 0.0
    metadata.name_source = "resume_id"
    logger.info("Name resolved from resume_id: %r", fallback)
    return fallback
