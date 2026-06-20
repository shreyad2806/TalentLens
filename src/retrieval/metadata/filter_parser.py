"""
Filter Parser for Metadata Filtering Engine.

Extracts structured MetadataFilter objects from natural-language recruiter queries
using regex and rule-based parsing.

Architecture Notes:
- Strategy pattern enables future LLM-based parser upgrade
- RuleBasedFilterParser is the default production backend
- FilterParser facade delegates to injectable strategy

Example:
    "Senior Python Developer in Bangalore with 5+ years experience under 25 LPA"
    → MetadataFilter(
        skills=["Python"],
        location="Bangalore",
        minimum_experience=5.0,
        salary_max=25.0,
      )

SOLID Principles Applied:
- Single Responsibility: Parsing only
- Open/Closed: New parsers via FilterParserStrategy
- Dependency Inversion: Depends on parser strategy abstraction
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from .schema import MetadataFilter, ParseResult
from .validator import MetadataFilterValidator, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns — Indian recruitment context (LPA, notice period, cities)
# ---------------------------------------------------------------------------

EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+\s*years?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*years?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*years?\s*(?:of\s*)?experience", re.IGNORECASE),
    re.compile(r"experience\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*\+?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*\+\s*yrs?", re.IGNORECASE),
]

SALARY_MAX_PATTERNS = [
    re.compile(r"under\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"below\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"max(?:imum)?\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"upto\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"up\s*to\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"<\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"ctc\s*(?:under|below|max)?\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
]

SALARY_MIN_PATTERNS = [
    re.compile(r"above\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"min(?:imum)?\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r"at\s*least\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
    re.compile(r">\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE),
]

SALARY_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*lpa", re.IGNORECASE
)

LOCATION_IN_PATTERN = re.compile(
    r"\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s+with|\s+under|\s+having|\s*$|,)",
    re.IGNORECASE,
)

NOTICE_PERIOD_PATTERNS = [
    re.compile(r"(\d+)\s*days?\s*notice", re.IGNORECASE),
    re.compile(r"notice\s*period\s*(?:of\s*)?(\d+)\s*days?", re.IGNORECASE),
    re.compile(r"immediate(?:ly)?\s*joiner", re.IGNORECASE),
]

WORK_MODE_KEYWORDS: Dict[str, str] = {
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
    "hybrid": "hybrid",
    "on-site": "onsite",
    "onsite": "onsite",
    "office": "onsite",
}

EMPLOYMENT_TYPE_KEYWORDS: Dict[str, str] = {
    "full-time": "full-time",
    "full time": "full-time",
    "part-time": "part-time",
    "part time": "part-time",
    "contract": "contract",
    "intern": "intern",
    "internship": "intern",
    "freelance": "freelance",
}

DEGREE_PATTERNS = [
    re.compile(r"\b(b\.?\s*tech|btech|b\.?\s*e\.?|bachelor(?:'s)?(?:\s+of)?(?:\s+technology)?)\b", re.IGNORECASE),
    re.compile(r"\b(m\.?\s*tech|mtech|m\.?\s*e\.?|master(?:'s)?)\b", re.IGNORECASE),
    re.compile(r"\b(mba|m\.?\s*b\.?\s*a\.?)\b", re.IGNORECASE),
    re.compile(r"\b(ph\.?\s*d|doctorate)\b", re.IGNORECASE),
    re.compile(r"\b(bca|mca)\b", re.IGNORECASE),
]

KNOWN_SKILLS: Set[str] = {
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node", "nodejs", "django", "flask", "fastapi", "spring", "kubernetes",
    "docker", "aws", "azure", "gcp", "sql", "postgresql", "mongodb", "redis",
    "kafka", "spark", "hadoop", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "data science", "devops", "ci/cd", "go", "golang",
    "rust", "c++", "c#", ".net", "scala", "kotlin", "swift", "android", "ios",
    "selenium", "pytest", "jenkins", "terraform", "ansible", "linux", "git",
    "html", "css", "graphql", "rest", "api", "microservices", "agile", "scrum",
}

INDIAN_CITIES: Set[str] = {
    "bangalore", "bengaluru", "mumbai", "delhi", "ncr", "gurgaon", "gurugram",
    "noida", "hyderabad", "chennai", "pune", "kolkata", "ahmedabad", "jaipur",
    "chandigarh", "kochi", "indore", "bhopal", "lucknow", "nagpur", "coimbatore",
}

SENIORITY_PREFIXES = {"senior", "sr", "lead", "principal", "staff", "junior", "jr", "mid"}


class FilterParserStrategy(ABC):
    """
    Abstract parser strategy — implement for rule-based or LLM backends.

    Future LLM upgrade: create LLMFilterParserStrategy implementing this
    interface and inject it into FilterParser at initialization.
    """

    @abstractmethod
    def parse(self, query: str) -> MetadataFilter:
        """Parse a recruiter query into a MetadataFilter."""
        ...


class RuleBasedFilterParser(FilterParserStrategy):
    """
    Regex and rule-based filter parser for recruiter queries.

    Extracts experience, salary, location, skills, work mode, and other
    structured filters from unstructured text.
    """

    def __init__(self) -> None:
        self._validator = MetadataFilterValidator()

    def parse(self, query: str) -> MetadataFilter:
        """
        Parse recruiter query into MetadataFilter using rules and regex.

        Args:
            query: Natural-language recruiter query

        Returns:
            MetadataFilter with extracted criteria
        """
        self._validator.validate_query(query)
        normalized = query.strip()
        lower = normalized.lower()

        filters = MetadataFilter()

        self._extract_experience(lower, filters)
        self._extract_salary(lower, filters)
        self._extract_location(normalized, lower, filters)
        self._extract_skills(lower, filters)
        self._extract_work_mode(lower, filters)
        self._extract_employment_type(lower, filters)
        self._extract_degree(lower, filters)
        self._extract_notice_period(lower, filters)
        self._extract_company(lower, filters)
        self._extract_languages(lower, filters)
        self._extract_availability(lower, filters)

        if not filters.is_empty():
            self._validator.validate_filter(filters, allow_empty=False)

        return filters

    def _extract_experience(self, lower: str, filters: MetadataFilter) -> None:
        """Extract minimum/maximum experience from query text."""
        for pattern in EXPERIENCE_PATTERNS:
            match = pattern.search(lower)
            if not match:
                continue
            groups = match.groups()
            if len(groups) == 2 and groups[1] is not None:
                filters.minimum_experience = float(groups[0])
                filters.maximum_experience = float(groups[1])
            else:
                value = float(groups[0])
                if "+" in match.group(0) or "minimum" not in lower[: match.start()]:
                    filters.minimum_experience = value
                else:
                    filters.maximum_experience = value
            return

    def _extract_salary(self, lower: str, filters: MetadataFilter) -> None:
        """Extract salary range filters (LPA)."""
        range_match = SALARY_RANGE_PATTERN.search(lower)
        if range_match:
            filters.salary_min = float(range_match.group(1))
            filters.salary_max = float(range_match.group(2))
            return

        for pattern in SALARY_MAX_PATTERNS:
            match = pattern.search(lower)
            if match:
                filters.salary_max = float(match.group(1))
                break

        for pattern in SALARY_MIN_PATTERNS:
            match = pattern.search(lower)
            if match:
                filters.salary_min = float(match.group(1))
                break

    def _extract_location(
        self, normalized: str, lower: str, filters: MetadataFilter
    ) -> None:
        """Extract primary location from 'in <city>' pattern or known cities."""
        match = LOCATION_IN_PATTERN.search(normalized)
        if match:
            location = match.group(1).strip().rstrip(",.")
            filters.location = location.title()
            return

        for city in INDIAN_CITIES:
            if re.search(rf"\b{re.escape(city)}\b", lower):
                filters.location = city.title() if city != "ncr" else "NCR"
                return

    def _extract_skills(self, lower: str, filters: MetadataFilter) -> None:
        """Extract known technical skills from query text."""
        found: List[str] = []
        for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, lower):
                found.append(skill.title() if skill.islower() else skill)

        if found:
            filters.skills = list(dict.fromkeys(found))

    def _extract_work_mode(self, lower: str, filters: MetadataFilter) -> None:
        """Extract work mode keyword (remote, hybrid, onsite)."""
        for keyword, mode in WORK_MODE_KEYWORDS.items():
            if keyword in lower:
                filters.work_mode = mode
                return

    def _extract_employment_type(self, lower: str, filters: MetadataFilter) -> None:
        """Extract employment type keyword."""
        for keyword, emp_type in EMPLOYMENT_TYPE_KEYWORDS.items():
            if keyword in lower:
                filters.employment_type = emp_type
                return

    def _extract_degree(self, lower: str, filters: MetadataFilter) -> None:
        """Extract degree requirement."""
        for pattern in DEGREE_PATTERNS:
            match = pattern.search(lower)
            if match:
                filters.degree = match.group(1).upper().replace(" ", "")
                return

    def _extract_notice_period(self, lower: str, filters: MetadataFilter) -> None:
        """Extract notice period in days."""
        if re.search(r"immediate(?:ly)?\s*joiner", lower):
            filters.notice_period = 0
            filters.availability = "immediate"
            return

        for pattern in NOTICE_PERIOD_PATTERNS:
            match = pattern.search(lower)
            if match and match.lastindex:
                filters.notice_period = int(match.group(1))
                return

    def _extract_company(self, lower: str, filters: MetadataFilter) -> None:
        """Extract current/previous company references."""
        current_match = re.search(
            r"(?:at|from|working\s+at|currently\s+at)\s+([A-Z][a-zA-Z0-9\s&]+?)(?:\s+with|\s+in|\s*$|,)",
            lower,
            re.IGNORECASE,
        )
        if current_match:
            filters.current_company = current_match.group(1).strip().title()

        prev_match = re.search(
            r"previously\s+(?:at|with)\s+([A-Z][a-zA-Z0-9\s&]+?)(?:\s+with|\s+in|\s*$|,)",
            lower,
            re.IGNORECASE,
        )
        if prev_match:
            filters.previous_company = prev_match.group(1).strip().title()

    def _extract_languages(self, lower: str, filters: MetadataFilter) -> None:
        """Extract language requirements."""
        lang_match = re.search(
            r"(?:speaks?|knows?|fluent\s+in)\s+([a-z,\s]+?)(?:\s+with|\s+in|\s*$|,)",
            lower,
        )
        if lang_match:
            langs = [l.strip().title() for l in lang_match.group(1).split(",") if l.strip()]
            if langs:
                filters.languages = langs

    def _extract_availability(self, lower: str, filters: MetadataFilter) -> None:
        """Extract availability status when not already set."""
        if filters.availability:
            return
        if "immediate" in lower:
            filters.availability = "immediate"
        elif "30 days" in lower or "1 month" in lower:
            filters.availability = "30_days"
        elif "15 days" in lower:
            filters.availability = "15_days"


class FilterParser:
    """
    Facade for recruiter query parsing.

    Delegates to an injectable FilterParserStrategy (default: rule-based).
    Swap strategy to LLMFilterParserStrategy when ready without changing callers.
    """

    def __init__(self, strategy: Optional[FilterParserStrategy] = None) -> None:
        self._strategy = strategy or RuleBasedFilterParser()
        self._backend_name = (
            "rule_based" if isinstance(self._strategy, RuleBasedFilterParser) else "custom"
        )
        logger.info(f"FilterParser initialized with backend={self._backend_name}")

    def parse(self, query: str) -> ParseResult:
        """
        Parse a recruiter query and return structured filters with timing.

        Args:
            query: Natural-language recruiter query

        Returns:
            ParseResult containing MetadataFilter and latency metrics
        """
        start = time.perf_counter()
        try:
            filters = self._strategy.parse(query)
        except ValidationError:
            raise
        except Exception as exc:
            logger.error(f"Filter parsing failed: {exc}")
            raise ValidationError(f"Failed to parse query: {exc}", field="query") from exc

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Parsed recruiter query in {latency_ms:.2f}ms — "
            f"active_filters={self._count_active_filters(filters)}"
        )

        return ParseResult(
            filters=filters,
            raw_query=query.strip(),
            parse_latency_ms=latency_ms,
            parser_backend=self._backend_name,
            cache_hit=False,
        )

    def set_strategy(self, strategy: FilterParserStrategy) -> None:
        """
        Replace parser strategy (e.g., upgrade to LLM backend).

        Args:
            strategy: New FilterParserStrategy implementation
        """
        self._strategy = strategy
        self._backend_name = (
            "rule_based" if isinstance(strategy, RuleBasedFilterParser) else "custom"
        )
        logger.info(f"FilterParser strategy updated to backend={self._backend_name}")

    @staticmethod
    def _count_active_filters(filters: MetadataFilter) -> int:
        """Count populated filter fields for logging."""
        count = 0
        for field_name in MetadataFilter.model_fields:
            value = getattr(filters, field_name)
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            count += 1
        return count
