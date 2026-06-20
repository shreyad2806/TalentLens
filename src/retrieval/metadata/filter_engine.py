"""
Filter Engine for Metadata Filtering Engine.

Applies MetadataFilter criteria to a candidate pool and returns filtered
candidate IDs ready for dense/sparse retrieval.

Filter Application Order (sequential AND):
    1. Experience range
    2. Location / preferred locations
    3. Skills (list intersection)
    4. Excluded skills (NOT)
    5. Education / degree
    6. Company filters
    7. Salary range
    8. Notice period
    9. Work mode / employment type
    10. Certifications / languages
    11. Availability
    12. Custom filters
    13. OR groups
    14. NOT conditions

SOLID Principles Applied:
- Single Responsibility: Filtering logic only
- Open/Closed: New filter types via FilterCondition
- Dependency Inversion: Operates on schema abstractions
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Set

from .schema import (
    CandidateMetadata,
    FilterCondition,
    FilterLogic,
    FilterOperator,
    FilterResult,
    MetadataFilter,
    OrFilterGroup,
)
from .validator import MetadataFilterValidator

logger = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    return value.strip().lower()


def _list_intersects(candidate_values: List[str], filter_values: List[str]) -> bool:
    """Return True when candidate and filter lists share at least one value."""
    candidate_set = {_normalize(v) for v in candidate_values}
    filter_set = {_normalize(v) for v in filter_values}
    return bool(candidate_set & filter_set)


def _contains(haystack: Optional[str], needle: str) -> bool:
    """Case-insensitive substring match."""
    if not haystack:
        return False
    return _normalize(needle) in _normalize(haystack)


def _exact_match(candidate_value: Optional[str], filter_value: str) -> bool:
    """Case-insensitive exact match."""
    if not candidate_value:
        return False
    return _normalize(candidate_value) == _normalize(filter_value)


class FilterEngine:
    """
    Applies metadata filters to candidate profiles sequentially.

    Each filter step narrows the candidate pool. Default logic is AND across
    all active filter fields. OR groups and NOT conditions are supported
    for advanced boolean expressions.
    """

    def __init__(self) -> None:
        self._validator = MetadataFilterValidator()
        self._field_evaluators: Dict[str, Callable[[CandidateMetadata, FilterCondition], bool]] = {
            "experience_years": self._eval_numeric_field,
            "location": self._eval_string_field,
            "skills": self._eval_list_field,
            "education": self._eval_list_field,
            "degree": self._eval_string_field,
            "current_company": self._eval_string_field,
            "previous_companies": self._eval_list_field,
            "salary_expectation": self._eval_numeric_field,
            "notice_period_days": self._eval_numeric_field,
            "work_mode": self._eval_string_field,
            "employment_type": self._eval_string_field,
            "certifications": self._eval_list_field,
            "languages": self._eval_list_field,
            "availability": self._eval_string_field,
        }
        logger.info("FilterEngine initialized")

    def apply(
        self,
        filters: MetadataFilter,
        candidates: List[CandidateMetadata],
    ) -> FilterResult:
        """
        Apply filters to candidates and return filtered candidate IDs.

        Args:
            filters: Structured metadata filter
            candidates: Full candidate metadata pool

        Returns:
            FilterResult with candidate IDs and metrics
        """
        start = time.perf_counter()
        total_before = len(candidates)

        self._validator.validate_candidates(candidates)
        self._validator.validate_filter(filters, allow_empty=True)

        if filters.is_empty():
            candidate_ids = [c.candidate_id for c in candidates]
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"No active filters — returning all {total_before} candidates "
                f"in {latency_ms:.2f}ms"
            )
            return FilterResult(
                candidate_ids=candidate_ids,
                total_before=total_before,
                total_after=len(candidate_ids),
                filters_applied=0,
                filter_latency_ms=latency_ms,
                cache_hit=False,
            )

        pool = list(candidates)
        filters_applied = 0

        # --- Sequential AND filters ---

        # minimum_experience: Range GTE — candidate experience >= threshold
        if filters.minimum_experience is not None:
            pool = [
                c for c in pool
                if c.experience_years is not None
                and c.experience_years >= filters.minimum_experience
            ]
            filters_applied += 1
            logger.debug(
                f"After minimum_experience>={filters.minimum_experience}: {len(pool)} candidates"
            )

        # maximum_experience: Range LTE — candidate experience <= ceiling
        if filters.maximum_experience is not None:
            pool = [
                c for c in pool
                if c.experience_years is not None
                and c.experience_years <= filters.maximum_experience
            ]
            filters_applied += 1
            logger.debug(
                f"After maximum_experience<={filters.maximum_experience}: {len(pool)} candidates"
            )

        # location: Exact match — primary location must equal filter value
        if filters.location:
            pool = [c for c in pool if _exact_match(c.location, filters.location)]
            filters_applied += 1
            logger.debug(f"After location={filters.location}: {len(pool)} candidates")

        # preferred_locations: OR + Contains — match any preferred location
        if filters.preferred_locations:
            preferred = [_normalize(l) for l in filters.preferred_locations]
            pool = [
                c for c in pool
                if any(
                    _contains(c.location, loc)
                    or any(_contains(pl, loc) for pl in c.preferred_locations)
                    for loc in preferred
                )
            ]
            filters_applied += 1
            logger.debug(f"After preferred_locations: {len(pool)} candidates")

        # skills: List intersection — candidate must share >=1 required skill
        if filters.skills:
            pool = [c for c in pool if _list_intersects(c.skills, filters.skills)]
            filters_applied += 1
            logger.debug(f"After skills intersection: {len(pool)} candidates")

        # excluded_skills: NOT intersection — exclude candidates with any excluded skill
        if filters.excluded_skills:
            excluded = {_normalize(s) for s in filters.excluded_skills}
            pool = [
                c for c in pool
                if not {_normalize(s) for s in c.skills} & excluded
            ]
            filters_applied += 1
            logger.debug(f"After excluded_skills NOT: {len(pool)} candidates")

        # education: Contains — any education entry contains keyword
        if filters.education:
            pool = [
                c for c in pool
                if any(
                    any(_contains(edu, kw) for edu in c.education)
                    for kw in filters.education
                )
            ]
            filters_applied += 1
            logger.debug(f"After education contains: {len(pool)} candidates")

        # degree: Exact match — degree field must equal filter value
        if filters.degree:
            pool = [c for c in pool if _exact_match(c.degree, filters.degree)]
            filters_applied += 1
            logger.debug(f"After degree={filters.degree}: {len(pool)} candidates")

        # current_company: Contains — current employer name contains value
        if filters.current_company:
            pool = [
                c for c in pool
                if _contains(c.current_company, filters.current_company)
            ]
            filters_applied += 1
            logger.debug(f"After current_company contains: {len(pool)} candidates")

        # previous_company: Contains — any previous employer contains value
        if filters.previous_company:
            pool = [
                c for c in pool
                if any(_contains(comp, filters.previous_company) for comp in c.previous_companies)
            ]
            filters_applied += 1
            logger.debug(f"After previous_company contains: {len(pool)} candidates")

        # salary_min: Range GTE — expected salary >= floor
        if filters.salary_min is not None:
            pool = [
                c for c in pool
                if c.salary_expectation is not None
                and c.salary_expectation >= filters.salary_min
            ]
            filters_applied += 1
            logger.debug(f"After salary_min>={filters.salary_min}: {len(pool)} candidates")

        # salary_max: Range LTE — expected salary <= ceiling
        if filters.salary_max is not None:
            pool = [
                c for c in pool
                if c.salary_expectation is not None
                and c.salary_expectation <= filters.salary_max
            ]
            filters_applied += 1
            logger.debug(f"After salary_max<={filters.salary_max}: {len(pool)} candidates")

        # notice_period: Range LTE — notice period days <= threshold
        if filters.notice_period is not None:
            pool = [
                c for c in pool
                if c.notice_period_days is not None
                and c.notice_period_days <= filters.notice_period
            ]
            filters_applied += 1
            logger.debug(f"After notice_period<={filters.notice_period}: {len(pool)} candidates")

        # work_mode: Exact match — remote | hybrid | onsite
        if filters.work_mode:
            pool = [c for c in pool if _exact_match(c.work_mode, filters.work_mode)]
            filters_applied += 1
            logger.debug(f"After work_mode={filters.work_mode}: {len(pool)} candidates")

        # employment_type: Exact match — full-time | contract | etc.
        if filters.employment_type:
            pool = [
                c for c in pool
                if _exact_match(c.employment_type, filters.employment_type)
            ]
            filters_applied += 1
            logger.debug(f"After employment_type: {len(pool)} candidates")

        # certifications: List intersection — candidate holds >=1 certification
        if filters.certifications:
            pool = [
                c for c in pool
                if _list_intersects(c.certifications, filters.certifications)
            ]
            filters_applied += 1
            logger.debug(f"After certifications intersection: {len(pool)} candidates")

        # languages: List intersection — candidate speaks >=1 language
        if filters.languages:
            pool = [
                c for c in pool
                if _list_intersects(c.languages, filters.languages)
            ]
            filters_applied += 1
            logger.debug(f"After languages intersection: {len(pool)} candidates")

        # availability: Exact match — immediate | 30_days | etc.
        if filters.availability:
            pool = [
                c for c in pool
                if _exact_match(c.availability, filters.availability)
            ]
            filters_applied += 1
            logger.debug(f"After availability: {len(pool)} candidates")

        # custom_filters: Contains — match arbitrary key-value pairs
        if filters.custom_filters:
            pool = [
                c for c in pool
                if all(
                    _contains(str(c.custom_fields.get(key, "")), str(val))
                    for key, val in filters.custom_filters.items()
                )
            ]
            filters_applied += 1
            logger.debug(f"After custom_filters: {len(pool)} candidates")

        # or_groups: OR — candidate must match at least one group
        if filters.or_groups:
            pool = [
                c for c in pool
                if any(self._evaluate_or_group(c, group) for group in filters.or_groups)
            ]
            filters_applied += 1
            logger.debug(f"After or_groups: {len(pool)} candidates")

        # not_conditions: NOT — exclude candidates matching any NOT condition
        if filters.not_conditions:
            pool = [
                c for c in pool
                if not any(
                    self._evaluate_condition(c, cond) for cond in filters.not_conditions
                )
            ]
            filters_applied += 1
            logger.debug(f"After not_conditions: {len(pool)} candidates")

        candidate_ids = [c.candidate_id for c in pool]
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Metadata filtering complete — "
            f"before={total_before}, after={len(candidate_ids)}, "
            f"filters_applied={filters_applied}, latency={latency_ms:.2f}ms"
        )

        return FilterResult(
            candidate_ids=candidate_ids,
            total_before=total_before,
            total_after=len(candidate_ids),
            filters_applied=filters_applied,
            filter_latency_ms=latency_ms,
            cache_hit=False,
        )

    def _evaluate_or_group(self, candidate: CandidateMetadata, group: OrFilterGroup) -> bool:
        """Return True if ANY condition in the OR group matches."""
        return any(self._evaluate_condition(candidate, cond) for cond in group.conditions)

    def _evaluate_condition(self, candidate: CandidateMetadata, condition: FilterCondition) -> bool:
        """Evaluate a single FilterCondition against a candidate."""
        field = condition.field
        evaluator = self._field_evaluators.get(field)
        if evaluator is None:
            custom_val = candidate.custom_fields.get(field)
            return self._apply_operator(custom_val, condition.operator, condition.value)

        return evaluator(candidate, condition)

    def _eval_numeric_field(
        self, candidate: CandidateMetadata, condition: FilterCondition
    ) -> bool:
        value = getattr(candidate, condition.field, None)
        return self._apply_operator(value, condition.operator, condition.value)

    def _eval_string_field(
        self, candidate: CandidateMetadata, condition: FilterCondition
    ) -> bool:
        value = getattr(candidate, condition.field, None)
        return self._apply_operator(value, condition.operator, condition.value)

    def _eval_list_field(
        self, candidate: CandidateMetadata, condition: FilterCondition
    ) -> bool:
        value = getattr(candidate, condition.field, [])
        if condition.operator == FilterOperator.INTERSECTS:
            if isinstance(condition.value, list):
                return _list_intersects(value, condition.value)
            return False
        if condition.operator == FilterOperator.CONTAINS:
            if isinstance(condition.value, str):
                return any(_contains(item, condition.value) for item in value)
            return False
        return self._apply_operator(value, condition.operator, condition.value)

    @staticmethod
    def _apply_operator(candidate_value, operator: FilterOperator, filter_value) -> bool:
        """Apply a comparison operator to a candidate field value."""
        if candidate_value is None:
            return False

        if operator == FilterOperator.EQ:
            if isinstance(candidate_value, str):
                return _exact_match(candidate_value, str(filter_value))
            return candidate_value == filter_value

        if operator == FilterOperator.CONTAINS:
            return _contains(str(candidate_value), str(filter_value))

        if operator == FilterOperator.GTE:
            return float(candidate_value) >= float(filter_value)

        if operator == FilterOperator.LTE:
            return float(candidate_value) <= float(filter_value)

        if operator == FilterOperator.IN:
            if isinstance(filter_value, list):
                normalized = {_normalize(str(v)) for v in filter_value}
                return _normalize(str(candidate_value)) in normalized
            return False

        if operator == FilterOperator.NOT_IN:
            if isinstance(filter_value, list):
                normalized = {_normalize(str(v)) for v in filter_value}
                return _normalize(str(candidate_value)) not in normalized
            return True

        if operator == FilterOperator.INTERSECTS:
            if isinstance(candidate_value, list) and isinstance(filter_value, list):
                return _list_intersects(candidate_value, filter_value)
            return False

        return False
