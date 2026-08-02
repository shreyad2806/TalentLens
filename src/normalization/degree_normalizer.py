"""Canonical degree string normalizer."""

from __future__ import annotations

import re


class DegreeNormalizer:
    """
    Normalize degree abbreviations and variants into canonical full names.

    Examples:
        - "BS" / "B.Sc" / "Bachelor of Science" -> "Bachelor of Science"
        - "BE" / "B.E." / "Bachelor Engineering" -> "Bachelor of Engineering"
        - "M.Tech" -> "Master of Technology"
    """

    _PATTERNS = [
        (re.compile(r"^\s*B\.?\s*E\.?\s*$", re.IGNORECASE), "Bachelor of Engineering"),
        (re.compile(r"^\s*B\.?\s*Eng\.?\s*$", re.IGNORECASE), "Bachelor of Engineering"),
        (re.compile(r"^\s*B\.?\s*Tech\.?\s*$", re.IGNORECASE), "Bachelor of Technology"),
        (re.compile(r"^\s*M\.?\s*Tech\.?\s*$", re.IGNORECASE), "Master of Technology"),
        (re.compile(r"^\s*B\.?\s*S\.?\s*c?\.?\s*$", re.IGNORECASE), "Bachelor of Science"),
        (re.compile(r"^\s*Bachelor\s+of\s+Science\b", re.IGNORECASE), "Bachelor of Science"),
        (re.compile(r"^\s*Bachelor\s+Engineering\b", re.IGNORECASE), "Bachelor of Engineering"),
        (re.compile(r"^\s*Bachelor\s+of\s+Engineering\b", re.IGNORECASE), "Bachelor of Engineering"),
        (re.compile(r"^\s*M\.?\s*S\.?\s*c?\.?\s*$", re.IGNORECASE), "Master of Science"),
        (re.compile(r"^\s*M\.?\s*A\.?\s*$", re.IGNORECASE), "Master of Arts"),
        (re.compile(r"^\s*M\.?\s*B\.?\s*A\.?\s*$", re.IGNORECASE), "Master of Business Administration"),
        (re.compile(r"^\s*Ph\.?\s*D\.?\s*$", re.IGNORECASE), "Doctor of Philosophy"),
    ]

    _EXAMPLES = [
        ("BS", "Bachelor of Science"),
        ("B.Sc", "Bachelor of Science"),
        ("Bachelor of Science", "Bachelor of Science"),
        ("BE", "Bachelor of Engineering"),
        ("B.E.", "Bachelor of Engineering"),
        ("Bachelor Engineering", "Bachelor of Engineering"),
        ("M.Tech", "Master of Technology"),
    ]

    @classmethod
    def normalize(cls, degree: str | None) -> str | None:
        if not degree:
            return None
        value = degree.strip()
        for pattern, canonical in cls._PATTERNS:
            if pattern.match(value):
                return canonical
        return value

    @classmethod
    def examples(cls) -> list[tuple[str, str]]:
        return cls._EXAMPLES[:]
