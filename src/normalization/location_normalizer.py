"""Canonical location string normalizer."""

from __future__ import annotations

import re


class LocationNormalizer:
    """
    Normalize location variants into canonical city / state names.

    Examples:
        - "NYC" -> "New York"
        - "SF" -> "San Francisco"
        - "CA" -> "California"
        - "Bengaluru" -> "Bangalore"

    Notes:
        "CA" is ambiguous. It is normalized to "California" here; the caller
        is responsible for disambiguating whether a city or state is intended.
    """

    _MAPPING = {
        "nyc": "New York",
        "new york city": "New York",
        "sf": "San Francisco",
        "san fran": "San Francisco",
        "ca": "California",
        "calif": "California",
        "california": "California",
        "bengaluru": "Bangalore",
        "bangalore": "Bangalore",
        "gurgaon": "Gurugram",
    }

    _EXAMPLES = [
        ("NYC", "New York"),
        ("SF", "San Francisco"),
        ("CA", "California"),
        ("Bengaluru", "Bangalore"),
    ]

    @classmethod
    def normalize(cls, location: str | None) -> str | None:
        if not location:
            return None
        key = re.sub(r"[-_\s]+", " ", location.strip()).lower().strip()
        return cls._MAPPING.get(key, location.strip())

    @classmethod
    def examples(cls) -> list[tuple[str, str]]:
        return cls._EXAMPLES[:]
