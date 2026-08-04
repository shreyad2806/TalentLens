"""Candidate-name validation and normalization.

This module provides the central `is_valid_candidate_name` helper used by the
metadata extractor, the UI display-name resolver, and the quality-audit script.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Known invalid / section-heading values that should never be treated as names.
INVALID_CANDIDATE_NAMES = {
    "",
    "unknown",
    "n/a",
    "resume",
    "cv",
    "curriculum vitae",
    "profile",
    "summary",
    "professional summary",
    "objective",
    "education",
    "skills",
    "technical skills",
    "core competencies",
    "projects",
    "experience",
    "relevant experience",
    "employment history",
    "career summary",
    "work experience",
    "work history",
    "computer skills",
    "critical thinking",
    "executive",
    "executive summary",
    "highlights",
    "accomplishments",
    "achievements",
    "awards",
    "publications",
    "languages",
    "english",
    "hindi",
    "marathi",
    "gujarati",
    "references",
    "achievements",
    "certifications",
    "contact",
    "personal details",
    "personal information",
    "key competencies",
    "budgeting extensive",
    "dependability staff",
    "dependability",
    "company name",
    "core qualifications",
    "other information",
    "accenture outstanding performer award",
    "outstanding performer award",
    "performer award",
    "professional summary",
    "core qualification",
    "core qualifications",
    "objective",
    "full name",
    "your name",
    "candidate name",
    "applicant name",
    "candidate",
    "applicant",
    "placeholder",
    "not provided",
    "not specified",
    "no name",
}

# Common resume section headings that should not appear as names.
HEADING_KEYWORDS = {
    "summary", "objective", "profile", "education", "experience", "skills",
    "technical", "projects", "certifications", "languages", "references",
    "achievements", "awards", "publications", "interests", "hobbies",
    "contact", "personal", "career", "employment", "work", "relevant",
    "professional", "computer", "critical", "budgeting", "key", "competencies",
    "company", "core", "qualifications", "other", "information", "name",
    "outstanding", "performer", "award", "accomplishments", "activities",
    "affiliations", "associations", "declaration", "details", "overview",
}


def _contains_phone_number(text: str) -> bool:
    """Return True if the text contains a phone-like number."""
    # E.164/local formats: groups of 3-4 digits separated by optional delimiters
    if re.search(r"\(?\d{2,4}\)?[\s\-./]?\d{2,4}[\s\-./]?\d{2,4}[\s\-./]?\d{2,4}", text):
        return True
    digits = re.sub(r"\D", "", text)
    return 10 <= len(digits) <= 15


def _contains_url(text: str) -> bool:
    """Return True if the text contains a URL."""
    return bool(re.search(r"https?://|www\.|\.[a-z]{2,}/", text, re.IGNORECASE))


def is_valid_candidate_name(name: Any, reason: bool = False) -> bool | tuple[bool, str]:
    """Return True only if *name* looks like a probable human name.

    If ``reason=True``, return a (valid, reason) tuple instead.
    """
    if not isinstance(name, str):
        return (False, "not_a_string") if reason else False

    name = name.strip()
    if not name:
        return (False, "empty") if reason else False

    lower = name.lower()
    if lower in INVALID_CANDIDATE_NAMES:
        return (False, f"invalid_value:{lower}") if reason else False

    # Hard rejects
    if "@" in name:
        return (False, "contains_email_at") if reason else False
    if "," in name:
        return (False, "contains_comma") if reason else False
    if "/" in name or "\\" in name or "|" in name:
        return (False, "contains_separator") if reason else False
    if _contains_phone_number(name):
        return (False, "contains_phone_number") if reason else False
    if _contains_url(name):
        return (False, "contains_url") if reason else False
    if re.search(r"\d", name):
        return (False, "contains_digit") if reason else False
    if len(name) > 60:
        return (False, "too_long") if reason else False

    # Reject common multi-word section headings exactly.
    lower = name.lower()
    if lower in INVALID_CANDIDATE_NAMES:
        return (False, f"invalid_heading:{lower}") if reason else False

    tokens = name.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return (False, f"token_count:{len(tokens)}") if reason else False

    # Reject single-token all-caps (likely a heading or acronym).
    if name.isupper() and len(tokens) == 1:
        return (False, "all_caps_heading") if reason else False

    # Reject if any token is a common heading/artifact keyword.
    token_set = {t.lower().strip(".,;:()") for t in tokens}
    overlap = token_set & INVALID_CANDIDATE_NAMES
    if overlap:
        return (False, f"invalid_token:{next(iter(overlap))}") if reason else False

    heading_hits = token_set & HEADING_KEYWORDS
    if heading_hits:
        return (False, f"heading_keyword:{next(iter(heading_hits))}") if reason else False

    # At least one alphabetic token.
    if not any(re.match(r"[A-Za-z]+", t) for t in tokens):
        return (False, "no_alpha_token") if reason else False

    return (True, "") if reason else True


def normalize_candidate_name(name: str) -> str:
    """Normalize a candidate name for display and storage.

    - Collapses whitespace
    - Strips surrounding punctuation
    - Title-cases the name
    """
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip(".,;:()[]{}|\\/")
    if not name:
        return ""

    # Title-case while keeping existing capitals for McX, O'X etc.
    def _title_token(t: str) -> str:
        # Preserve all-caps initials like IBM, PhD if short
        if len(t) <= 2 and t.isupper():
            return t
        # Handle hyphenated names
        if "-" in t:
            return "-".join(_title_token(p) for p in t.split("-"))
        # Handle apostrophe (O'Connor, D'Souza)
        if "'" in t:
            parts = t.split("'")
            return "'".join(_title_token(p) if i == 0 else p.capitalize() for i, p in enumerate(parts))
        return t.capitalize()

    tokens = name.split()
    return " ".join(_title_token(t) for t in tokens)


def display_name_from_filename(filename: str | None) -> str | None:
    """Return a display-friendly name from a resume filename if it looks valid."""
    if not filename:
        return None
    name = re.sub(r"\.[^.]+$", "", str(filename))
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if is_valid_candidate_name(name):
        return normalize_candidate_name(name)
    return None


def fallback_display_name(resume_id: str) -> str:
    """Stable fallback when no real name can be found."""
    return f"Resume #{resume_id}"
