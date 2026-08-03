"""Canonical role title normalizer."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class RoleNormalizer:
    """
    Normalize free-form job titles into a canonical set.

    Examples:
        - "Backend Developer" / "Backend Engineer" -> "Backend Software Engineer"
        - "ML Engineer" -> "Machine Learning Engineer"
        - "AI Developer" -> "AI Engineer"
    """

    # Ordered patterns: first match wins.
    _PATTERNS = [
        (re.compile(r"\b(finance\s+(manager|lead|executive))\b", re.IGNORECASE), "Finance Manager"),
        (re.compile(r"\b(software\s+(engineer|developer)|software|backend\s+(engineer|developer)|backend|frontend\s+(engineer|developer)|frontend|full\s*stack\s+(engineer|developer)|fullstack\s+(engineer|developer)|fullstack)\b", re.IGNORECASE), "Software Engineer"),
        (re.compile(r"\b(ml|machine learning)\b", re.IGNORECASE), "Machine Learning Engineer"),
        (re.compile(r"\b(ai)\b", re.IGNORECASE), "AI Engineer"),
        (re.compile(r"\b(data scientist)\b", re.IGNORECASE), "Data Scientist"),
        (re.compile(r"\b(data analyst)\b", re.IGNORECASE), "Data Analyst"),
    ]

    _EXAMPLES = [
        ("Software Engineer", "Software Engineer"),
        ("Backend Developer", "Backend Software Engineer"),
        ("Backend Engineer", "Backend Software Engineer"),
        ("ML Engineer", "Machine Learning Engineer"),
        ("Machine Learning Engineer", "Machine Learning Engineer"),
        ("AI Developer", "AI Engineer"),
        ("AI Engineer", "AI Engineer"),
    ]

    @classmethod
    def normalize(cls, role: str | None) -> str | None:
        if not role:
            return None
        cleaned = role.strip()
        for pattern, canonical in cls._PATTERNS:
            if pattern.search(cleaned):
                logger.info("Role normalized: %r -> %r", cleaned, canonical)
                return canonical
        logger.info("Role unchanged: %r", cleaned)
        return cleaned

    @classmethod
    def examples(cls) -> list[tuple[str, str]]:
        return cls._EXAMPLES[:]
