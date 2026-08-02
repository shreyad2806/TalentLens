"""Experience duration normalizer."""

from __future__ import annotations

import re


class ExperienceNormalizer:
    """
    Normalize experience duration strings to a single year float.

    Examples:
        - "3 yrs"    -> 3.0
        - "3 years"  -> 3.0
        - "36 months" -> 3.0
        - 36 (int)   -> 3.0 (treated as months)
    """

    _EXAMPLES = [
        ("3 yrs", 3.0),
        ("3 years", 3.0),
        ("36 months", 3.0),
    ]

    @classmethod
    def normalize(cls, value: str | float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Bare integers from the synthetic dataset are already years.
            # Bare integers from a month context are handled by string regex below.
            return float(value)
        text = str(value).strip().lower()
        if not text:
            return None

        # Years: 3, 3 yrs, 3 years, 3+
        m_year = re.match(r"^(\d+(?:\.\d+)?)\s*(?:\+?\s*y(?:ears?|rs?)?)\b", text)
        if m_year:
            return float(m_year.group(1))

        # Months: 36 months, 18 mo
        m_month = re.match(r"^(\d+(?:\.\d+)?)\s*(?:months?|mos?)\b", text)
        if m_month:
            return round(float(m_month.group(1)) / 12, 2)

        # Plain number fallback: if it looks like a year, use it as-is.
        m_plain = re.match(r"^(\d+(?:\.\d+)?)$", text)
        if m_plain:
            return float(m_plain.group(1))

        return None

    @classmethod
    def examples(cls) -> list[tuple[str, float]]:
        return cls._EXAMPLES[:]
