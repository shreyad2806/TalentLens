"""Occupation-aware role similarity scoring.

Computes a 0-1 similarity between a query role and a candidate's primary role
using a lightweight, deterministic taxonomy. It does not depend on semantic
embeddings and can be used to rerank resumes by occupation fit.
"""

from __future__ import annotations

import re
from typing import Any


class RoleSimilarityScorer:
    """
    Score how similar two occupation titles are.

    Examples (query = "Machine Learning Engineer"):
        Machine Learning Engineer  -> 1.00
        AI Engineer                -> 0.95
        Data Scientist             -> 0.90
        NLP Engineer               -> 0.88
        Software Engineer          -> 0.70
        Mechanical Engineer        -> 0.20
        Agriculture Professional   -> 0.00
    """

    # Domain keyword groups. Each group is a coarse skill/field family.
    DOMAIN_GROUPS: dict[str, list[str]] = {
        "machine_learning": [
            "machine learning", "ml", "deep learning", "neural network",
            "neural networks", "reinforcement learning", "statistical learning",
        ],
        "artificial_intelligence": [
            "artificial intelligence", "ai",
        ],
        "nlp": [
            "nlp", "natural language processing", "natural language",
            "text processing", "computational linguistics",
        ],
        "computer_vision": [
            "computer vision", "cv", "vision", "image processing",
        ],
        "data_science": [
            "data science", "data scientist", "data sciences",
        ],
        "data_analytics": [
            "data analytics", "data analysis", "data analyst", "analytics",
            "business intelligence", "bi",
        ],
        "data_engineering": [
            "data engineer", "data engineering", "etl", "pipeline", "pipelines",
        ],
        "software": [
            "software", "swe", "web", "mobile", "backend", "frontend",
            "full stack", "fullstack", "devops", "cloud", "site reliability",
            "api", "platform", "react", "angular", "vue", "ui",
        ],
        "mechanical": [
            "mechanical", "mechanics",
        ],
        "electrical": [
            "electrical", "electronics",
        ],
        "civil": [
            "civil", "construction", "structural",
        ],
        "aerospace": [
            "aerospace", "aeronautical", "aviation",
        ],
        "hardware": [
            "hardware", "embedded", "fpga", "vlsi",
        ],
        "agriculture": [
            "agriculture", "agricultural", "agronomist", "agronomy",
            "farming", "farm", "crop", "soil", "horticulture",
        ],
        "healthcare": [
            "healthcare", "medical", "clinical", "nurse", "doctor", "physician",
        ],
        "finance": [
            "finance", "financial", "accounting", "investment", "banking",
        ],
        "consulting": [
            "consulting", "consultant", "advisory", "advisor",
        ],
        "research": [
            "research", "researcher", "research scientist",
        ],
        "marketing": [
            "marketing", "growth", "seo", "brand",
        ],
        "sales": [
            "sales", "business development", "account executive",
        ],
        "human_resources": [
            "human resources", "hr", "recruiter", "talent",
        ],
        "operations": [
            "operations", "supply chain", "logistics", "procurement",
        ],
    }

    # Cross-group domain similarity (1.0 within the same group).
    # Symmetric; if not present, 0.0 unless handled by defaults.
    CROSS_DOMAIN_SIM: dict[frozenset[str, str], float] = {
        frozenset({"machine_learning", "artificial_intelligence"}): 0.95,
        frozenset({"machine_learning", "nlp"}): 0.92,
        frozenset({"machine_learning", "computer_vision"}): 0.92,
        frozenset({"machine_learning", "data_science"}): 0.90,
        frozenset({"machine_learning", "mechanical"}): 0.20,
        frozenset({"machine_learning", "data_analytics"}): 0.85,
        frozenset({"machine_learning", "data_engineering"}): 0.82,
        frozenset({"machine_learning", "software"}): 0.70,
        frozenset({"machine_learning", "research"}): 0.80,
        frozenset({"artificial_intelligence", "nlp"}): 0.90,
        frozenset({"artificial_intelligence", "computer_vision"}): 0.92,
        frozenset({"artificial_intelligence", "data_science"}): 0.88,
        frozenset({"artificial_intelligence", "data_engineering"}): 0.80,
        frozenset({"artificial_intelligence", "software"}): 0.70,
        frozenset({"data_science", "data_analytics"}): 0.90,
        frozenset({"data_science", "data_engineering"}): 0.85,
        frozenset({"data_science", "software"}): 0.65,
        frozenset({"data_analytics", "data_engineering"}): 0.80,
        frozenset({"data_analytics", "software"}): 0.60,
        frozenset({"data_engineering", "software"}): 0.75,
        frozenset({"software", "mechanical"}): 0.20,
        frozenset({"software", "electrical"}): 0.35,
        frozenset({"software", "civil"}): 0.10,
        frozenset({"software", "aerospace"}): 0.10,
        frozenset({"software", "hardware"}): 0.50,
        frozenset({"mechanical", "electrical"}): 0.50,
        frozenset({"mechanical", "civil"}): 0.40,
        frozenset({"mechanical", "aerospace"}): 0.60,
        frozenset({"electrical", "aerospace"}): 0.55,
        frozenset({"electrical", "hardware"}): 0.70,
        frozenset({"hardware", "software"}): 0.50,
        frozenset({"agriculture", "software"}): 0.00,
        frozenset({"agriculture", "mechanical"}): 0.15,
        frozenset({"agriculture", "electrical"}): 0.05,
        frozenset({"healthcare", "software"}): 0.20,
        frozenset({"finance", "software"}): 0.35,
        frozenset({"finance", "data_science"}): 0.60,
        frozenset({"consulting", "software"}): 0.30,
        frozenset({"consulting", "finance"}): 0.45,
        frozenset({"marketing", "sales"}): 0.60,
        frozenset({"human_resources", "operations"}): 0.40,
    }

    # Seniority / professional-level keywords and a coarse level similarity table.
    LEVEL_KEYWORDS: dict[str, list[str]] = {
        "engineer": ["engineer", "engineering", "developer", "programmer"],
        "scientist": ["scientist", "researcher"],
        "analyst": ["analyst"],
        "manager": ["manager", "management"],
        "director": ["director"],
        "consultant": ["consultant"],
        "professional": ["professional", "specialist"],
    }

    LEVEL_SIM: dict[frozenset[str, str], float] = {
        frozenset({"engineer", "scientist"}): 1.0,
        frozenset({"engineer", "analyst"}): 0.85,
        frozenset({"engineer", "developer"}): 1.0,
        frozenset({"scientist", "analyst"}): 0.90,
        frozenset({"engineer", "manager"}): 0.70,
        frozenset({"scientist", "manager"}): 0.70,
        frozenset({"analyst", "manager"}): 0.75,
        frozenset({"engineer", "consultant"}): 0.65,
        frozenset({"scientist", "consultant"}): 0.65,
        frozenset({"professional", "engineer"}): 0.60,
    }

    # Pre-compiled regexes to avoid recompiling on every similarity call.
    _PUNCT_RE = re.compile(r"[^a-z0-9\s]")
    _WHITESPACE_RE = re.compile(r"\s+")
    _SUFFIX_RE = re.compile(
        r"(?:^|\s)(?:"
        + "|".join(re.escape(s) for s in [
            "senior", "sr", "junior", "jr", "lead", "principal", "staff",
            "manager", "director", "head", "chief", "vice president", "vp",
            "executive", "associate", "assistant", "intern", "professional",
        ])
        + r")(?:\s|$)"
    )
    _DOMAIN_PATTERNS: dict[str, list[tuple[re.Pattern, int]]] = {
        group: [(re.compile(rf"\b{re.escape(kw)}\b"), len(kw)) for kw in keywords]
        for group, keywords in DOMAIN_GROUPS.items()
    }
    _LEVEL_PATTERNS: dict[str, list[tuple[re.Pattern, int]]] = {
        level: [(re.compile(rf"\b{re.escape(kw)}\b"), len(kw)) for kw in keywords]
        for level, keywords in LEVEL_KEYWORDS.items()
    }

    @classmethod
    def normalize(cls, role: str | None) -> str:
        """Lowercase and strip noisy punctuation from a title."""
        if not role:
            return ""
        r = role.lower()
        r = cls._PUNCT_RE.sub(" ", r)
        r = cls._WHITESPACE_RE.sub(" ", r).strip()
        # Remove trailing seniority/modifiers that are not part of the occupation core.
        r = cls._SUFFIX_RE.sub(" ", r)
        r = cls._WHITESPACE_RE.sub(" ", r).strip()
        return r

    @classmethod
    def _detect_domains(cls, title: str) -> set[str]:
        """Return the domain groups present in a normalized title."""
        found: set[str] = set()
        for group, patterns in cls._DOMAIN_PATTERNS.items():
            for pat, _ in patterns:
                if pat.search(title):
                    found.add(group)
                    break
        return found

    @classmethod
    def _best_domain(cls, title: str) -> str | None:
        """Return the most specific domain group for the title (prefer longest keyword)."""
        best_group: str | None = None
        best_len = 0
        for group, patterns in cls._DOMAIN_PATTERNS.items():
            for pat, kw_len in patterns:
                if pat.search(title) and kw_len > best_len:
                    best_group = group
                    best_len = kw_len
        return best_group

    @classmethod
    def _detect_levels(cls, title: str) -> set[str]:
        """Return the professional level groups present in a title."""
        found: set[str] = set()
        for level, patterns in cls._LEVEL_PATTERNS.items():
            for pat, _ in patterns:
                if pat.search(title):
                    found.add(level)
                    break
        return found

    @classmethod
    def _domain_similarity(cls, q: str, c: str) -> float:
        """Compute domain/field similarity between two normalized titles."""
        q_domains = cls._detect_domains(q)
        c_domains = cls._detect_domains(c)

        # Exact or near-exact title match.
        if q == c:
            return 1.0
        if q in c or c in q:
            return 0.98

        if q_domains and c_domains:
            # If both share any domain group, they are very similar but not exact
            # unless the titles are the same (handled before this method).
            best = 0.0
            for qd in q_domains:
                for cd in c_domains:
                    if qd == cd:
                        best = max(best, 0.95)
                    else:
                        best = max(best, cls.CROSS_DOMAIN_SIM.get(frozenset({qd, cd}), 0.0))
            return best

        # One side has a domain, the other does not (e.g., generic "Engineer").
        if q_domains or c_domains:
            return 0.20

        # Neither has a recognizable domain.
        return 0.05

    @classmethod
    def _level_similarity(cls, q: str, c: str) -> float:
        """Compute professional-level similarity (engineer/scientist/analyst/etc.)."""
        q_levels = cls._detect_levels(q)
        c_levels = cls._detect_levels(c)

        if not q_levels or not c_levels:
            # If at least one has no explicit level, assume neutral (0.95) to avoid
            # over-penalizing unconventional titles.
            return 0.95

        if q_levels & c_levels:
            return 1.0

        best = 0.0
        for ql in q_levels:
            for cl in c_levels:
                sim = cls.LEVEL_SIM.get(frozenset({ql, cl}), 0.85)
                best = max(best, sim)
        return best

    @classmethod
    def score(cls, query_role: str | None, candidate_role: str | None) -> float:
        """Return 0-1 similarity between the query and candidate occupations."""
        if not query_role or not candidate_role:
            return 0.0

        q = cls.normalize(query_role)
        c = cls.normalize(candidate_role)

        if not q or not c:
            return 0.0
        if q == c:
            return 1.0

        domain_sim = cls._domain_similarity(q, c)
        level_sim = cls._level_similarity(q, c)

        # Combine: domain drives the score, level smooths it slightly.
        return round(min(1.0, domain_sim * level_sim), 4)
