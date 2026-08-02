"""Canonical role title normalizer."""

from __future__ import annotations

import re


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
        (re.compile(r"\b(backend)\b", re.IGNORECASE), "Backend Software Engineer"),
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
                return canonical
        return cleaned

    @classmethod
    def examples(cls) -> list[tuple[str, str]]:
        return cls._EXAMPLES[:]
