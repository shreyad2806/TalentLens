"""Canonical skill name normalizer."""

from __future__ import annotations

import re
from typing import Optional


class SkillNormalizer:
    """
    Normalize skill name variants into a canonical form.

    Examples:
        - "NodeJS" -> "Node.js"
        - "ReactJS" -> "React"
        - "PyTorch" -> "pytorch"
        - "Tensor Flow" -> "TensorFlow"
    """

    _MAPPING = {
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "reactjs": "React",
        "pytorch": "pytorch",
        "tensor flow": "TensorFlow",
        "tensorflow": "TensorFlow",
        "machine learning": "Machine Learning",
        "ml": "Machine Learning",
    }

    _EXAMPLES = [
        ("NodeJS", "Node.js"),
        ("ReactJS", "React"),
        ("PyTorch", "pytorch"),
        ("Tensor Flow", "TensorFlow"),
    ]

    @classmethod
    def normalize(cls, skill: Optional[str]) -> Optional[str]:
        if not skill:
            return None
        key = skill.strip()
        normalized = re.sub(r"[-_\s]+", " ", key).strip().lower()
        return cls._MAPPING.get(normalized, key)

    @classmethod
    def normalize_list(cls, skills: list[str]) -> list[str]:
        return [s for s in (cls.normalize(s) for s in skills) if s is not None]

    @classmethod
    def examples(cls) -> list[tuple[str, str]]:
        return cls._EXAMPLES[:]
