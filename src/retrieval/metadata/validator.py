"""
Validator for Metadata Filtering Engine.

This module validates MetadataFilter objects, candidate metadata, and
recruiter queries before parsing or filtering.

Validation Rules:
- Invalid experience ranges (min > max)
- Negative experience values
- Salary inconsistencies (min > max)
- Duplicate filter values within list fields
- Empty filter objects with no criteria
- Invalid location strings
- Invalid work_mode and employment_type enums

SOLID Principles Applied:
- Single Responsibility: Handles only validation
- Open/Closed: Open for new validation rules
"""

import logging
import re
from typing import List, Optional, Set

from .schema import CandidateMetadata, MetadataFilter

logger = logging.getLogger(__name__)

VALID_WORK_MODES = {"remote", "hybrid", "onsite", "on-site", "wfh"}
VALID_EMPLOYMENT_TYPES = {
    "full-time",
    "fulltime",
    "part-time",
    "parttime",
    "contract",
    "intern",
    "internship",
    "freelance",
}
VALID_AVAILABILITY = {
    "immediate",
    "15_days",
    "30_days",
    "60_days",
    "90_days",
    "negotiable",
}
LOCATION_PATTERN = re.compile(r"^[a-zA-Z\s,\-'.]{2,100}$")


class ValidationError(Exception):
    """Custom exception for metadata validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class MetadataFilterValidator:
    """
    Validator for metadata filters and candidate profiles.

    Ensures filter criteria are logically consistent before the FilterEngine
    applies them to the candidate pool.
    """

    def __init__(self) -> None:
        logger.info("MetadataFilterValidator initialized")

    def validate_query(self, query: str) -> None:
        """
        Validate a recruiter query string before parsing.

        Args:
            query: Raw recruiter query

        Raises:
            ValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise ValidationError("Recruiter query cannot be empty", field="query")

        if len(query.strip()) < 3:
            raise ValidationError(
                "Recruiter query must be at least 3 characters", field="query"
            )

        if len(query) > 2000:
            raise ValidationError(
                "Recruiter query cannot exceed 2000 characters", field="query"
            )

    def validate_filter(self, filters: MetadataFilter, allow_empty: bool = False) -> None:
        """
        Validate a MetadataFilter for logical consistency.

        Args:
            filters: Filter object to validate
            allow_empty: Whether an empty filter is acceptable

        Raises:
            ValidationError: If filter is invalid
        """
        if filters is None:
            raise ValidationError("MetadataFilter cannot be None", field="filters")

        if not allow_empty and filters.is_empty():
            raise ValidationError(
                "MetadataFilter has no active criteria", field="filters"
            )

        self._validate_experience_range(filters)
        self._validate_salary_range(filters)
        self._validate_negative_experience(filters)
        self._validate_duplicate_list_values(filters)
        self._validate_locations(filters)
        self._validate_enums(filters)
        self._validate_conflicting_skills(filters)

        logger.debug("MetadataFilter validation passed")

    def validate_candidates(self, candidates: List[CandidateMetadata]) -> None:
        """
        Validate candidate metadata list before filtering.

        Args:
            candidates: List of candidate metadata objects

        Raises:
            ValidationError: If candidates list is invalid
        """
        if candidates is None:
            raise ValidationError("Candidates list cannot be None", field="candidates")

        seen_ids: Set[str] = set()
        for candidate in candidates:
            if not candidate.candidate_id:
                raise ValidationError(
                    "Candidate must have a candidate_id", field="candidate_id"
                )
            if candidate.candidate_id in seen_ids:
                raise ValidationError(
                    f"Duplicate candidate_id: {candidate.candidate_id}",
                    field="candidate_id",
                )
            seen_ids.add(candidate.candidate_id)

            if candidate.experience_years is not None and candidate.experience_years < 0:
                raise ValidationError(
                    f"Negative experience for candidate {candidate.candidate_id}",
                    field="experience_years",
                )

            if candidate.salary_expectation is not None and candidate.salary_expectation < 0:
                raise ValidationError(
                    f"Negative salary for candidate {candidate.candidate_id}",
                    field="salary_expectation",
                )

    def _validate_experience_range(self, filters: MetadataFilter) -> None:
        """Reject invalid experience ranges where min exceeds max."""
        if (
            filters.minimum_experience is not None
            and filters.maximum_experience is not None
            and filters.minimum_experience > filters.maximum_experience
        ):
            raise ValidationError(
                f"minimum_experience ({filters.minimum_experience}) "
                f"cannot exceed maximum_experience ({filters.maximum_experience})",
                field="minimum_experience",
            )

    def _validate_salary_range(self, filters: MetadataFilter) -> None:
        """Reject salary ranges where min exceeds max."""
        if (
            filters.salary_min is not None
            and filters.salary_max is not None
            and filters.salary_min > filters.salary_max
        ):
            raise ValidationError(
                f"salary_min ({filters.salary_min}) "
                f"cannot exceed salary_max ({filters.salary_max})",
                field="salary_min",
            )

    def _validate_negative_experience(self, filters: MetadataFilter) -> None:
        """Reject negative experience filter values."""
        if filters.minimum_experience is not None and filters.minimum_experience < 0:
            raise ValidationError(
                "minimum_experience cannot be negative", field="minimum_experience"
            )
        if filters.maximum_experience is not None and filters.maximum_experience < 0:
            raise ValidationError(
                "maximum_experience cannot be negative", field="maximum_experience"
            )

    def _validate_duplicate_list_values(self, filters: MetadataFilter) -> None:
        """Reject duplicate entries within list-based filter fields."""
        list_field_map = {
            "preferred_locations": filters.preferred_locations,
            "skills": filters.skills,
            "excluded_skills": filters.excluded_skills,
            "education": filters.education,
            "certifications": filters.certifications,
            "languages": filters.languages,
        }
        for field_name, values in list_field_map.items():
            if not values:
                continue
            normalized = [v.strip().lower() for v in values if v and v.strip()]
            if len(normalized) != len(set(normalized)):
                raise ValidationError(
                    f"Duplicate values in {field_name}", field=field_name
                )

    def _validate_locations(self, filters: MetadataFilter) -> None:
        """Reject malformed location strings."""
        locations: List[str] = []
        if filters.location:
            locations.append(filters.location)
        if filters.preferred_locations:
            locations.extend(filters.preferred_locations)

        for loc in locations:
            if not loc or not loc.strip():
                raise ValidationError("Location cannot be empty", field="location")
            if not LOCATION_PATTERN.match(loc.strip()):
                raise ValidationError(
                    f"Invalid location format: {loc}", field="location"
                )

    def _validate_enums(self, filters: MetadataFilter) -> None:
        """Validate enumerated filter fields."""
        if filters.work_mode:
            normalized = filters.work_mode.strip().lower().replace(" ", "-")
            if normalized not in VALID_WORK_MODES:
                raise ValidationError(
                    f"Invalid work_mode: {filters.work_mode}. "
                    f"Expected one of {sorted(VALID_WORK_MODES)}",
                    field="work_mode",
                )

        if filters.employment_type:
            normalized = filters.employment_type.strip().lower().replace(" ", "-")
            if normalized not in VALID_EMPLOYMENT_TYPES:
                raise ValidationError(
                    f"Invalid employment_type: {filters.employment_type}",
                    field="employment_type",
                )

        if filters.availability:
            normalized = filters.availability.strip().lower()
            if normalized not in VALID_AVAILABILITY:
                raise ValidationError(
                    f"Invalid availability: {filters.availability}",
                    field="availability",
                )

    def _validate_conflicting_skills(self, filters: MetadataFilter) -> None:
        """Reject filters where required and excluded skills overlap."""
        if not filters.skills or not filters.excluded_skills:
            return

        required = {s.strip().lower() for s in filters.skills}
        excluded = {s.strip().lower() for s in filters.excluded_skills}
        overlap = required & excluded
        if overlap:
            raise ValidationError(
                f"Skills appear in both skills and excluded_skills: {overlap}",
                field="skills",
            )
