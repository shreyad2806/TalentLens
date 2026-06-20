"""
Schema definitions for Metadata Filtering Engine.

This module defines Pydantic schemas for recruiter filters, candidate metadata,
filter conditions, and filtering metrics.

Architecture Notes:
- Pydantic models for data validation
- Frozen models for immutability where appropriate
- Explicit filter semantics documented on each field
- Type safety throughout

SOLID Principles Applied:
- Single Responsibility: Schema definitions only
- Open/Closed: Open for extension with new filter fields
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FilterOperator(str, Enum):
    """Supported filter comparison operators."""

    EQ = "eq"  # Exact match
    CONTAINS = "contains"  # Substring / partial match
    GTE = "gte"  # Greater than or equal (range lower bound)
    LTE = "lte"  # Less than or equal (range upper bound)
    IN = "in"  # Value in list
    NOT_IN = "not_in"  # Value not in list
    INTERSECTS = "intersects"  # Non-empty list intersection


class FilterLogic(str, Enum):
    """Logical combinator for filter groups."""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterCondition(BaseModel):
    """
    Atomic filter condition for explicit OR/NOT groups.

    Used by FilterEngine for advanced boolean logic beyond default
    field-level AND semantics.
    """

    field: str = Field(..., description="CandidateMetadata field name")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    logic: FilterLogic = Field(
        default=FilterLogic.AND,
        description="How this condition combines with siblings in a group",
    )

    class Config:
        frozen = True


class OrFilterGroup(BaseModel):
    """
    OR filter group — candidate passes if ANY condition in the group matches.

    Example: match candidates in Bangalore OR Mumbai.
    """

    conditions: List[FilterCondition] = Field(
        ...,
        min_length=1,
        description="Conditions combined with OR logic",
    )

    class Config:
        frozen = True


class MetadataFilter(BaseModel):
    """
    Structured metadata filter extracted from a recruiter query or API request.

    Default semantics: all populated fields are combined with AND logic.
    Field-level behavior is documented inline below.

    Filter Semantics:
        minimum_experience  — Range (GTE): candidate experience >= value
        maximum_experience  — Range (LTE): candidate experience <= value
        location            — Exact match (case-insensitive) on primary location
        preferred_locations — OR + Contains: match any listed location
        skills              — List intersection: candidate must share >=1 skill
        excluded_skills     — NOT + Intersection: exclude if any skill matches
        education           — Contains: any education entry contains value
        degree              — Exact match (case-insensitive) on degree field
        current_company     — Contains: current employer name contains value
        previous_company    — Contains: any previous employer contains value
        salary_min          — Range (GTE): expected salary >= value (LPA)
        salary_max          — Range (LTE): expected salary <= value (LPA)
        notice_period       — Range (LTE): notice period days <= value
        work_mode           — Exact match: remote | hybrid | onsite
        employment_type     — Exact match: full-time | part-time | contract | intern
        certifications      — List intersection: candidate holds >=1 certification
        languages           — List intersection: candidate speaks >=1 language
        availability        — Exact match: immediate | 15_days | 30_days | etc.
        custom_filters      — Key-value pairs applied with Contains semantics
        or_groups           — Explicit OR groups (any group may match)
        not_conditions      — Explicit NOT conditions (exclude on match)
    """

    minimum_experience: Optional[float] = Field(
        None, ge=0, description="Minimum years of experience (inclusive)"
    )
    maximum_experience: Optional[float] = Field(
        None, ge=0, description="Maximum years of experience (inclusive)"
    )
    location: Optional[str] = Field(
        None, description="Primary location — exact match, case-insensitive"
    )
    preferred_locations: Optional[List[str]] = Field(
        None, description="Preferred locations — OR logic, contains match"
    )
    skills: Optional[List[str]] = Field(
        None, description="Required skills — list intersection (>=1 match)"
    )
    excluded_skills: Optional[List[str]] = Field(
        None, description="Excluded skills — NOT intersection"
    )
    education: Optional[List[str]] = Field(
        None, description="Education keywords — contains match on any entry"
    )
    degree: Optional[str] = Field(
        None, description="Degree requirement — exact match, case-insensitive"
    )
    current_company: Optional[str] = Field(
        None, description="Current employer — contains match"
    )
    previous_company: Optional[str] = Field(
        None, description="Previous employer — contains match on any entry"
    )
    salary_min: Optional[float] = Field(
        None, ge=0, description="Minimum salary in LPA (inclusive)"
    )
    salary_max: Optional[float] = Field(
        None, ge=0, description="Maximum salary in LPA (inclusive)"
    )
    notice_period: Optional[int] = Field(
        None, ge=0, description="Maximum notice period in days (inclusive)"
    )
    work_mode: Optional[str] = Field(
        None, description="Work mode: remote | hybrid | onsite"
    )
    employment_type: Optional[str] = Field(
        None, description="Employment type: full-time | part-time | contract | intern"
    )
    certifications: Optional[List[str]] = Field(
        None, description="Required certifications — list intersection"
    )
    languages: Optional[List[str]] = Field(
        None, description="Required languages — list intersection"
    )
    availability: Optional[str] = Field(
        None, description="Availability status — exact match"
    )
    custom_filters: Optional[Dict[str, Any]] = Field(
        None, description="Arbitrary key-value filters — contains match"
    )
    or_groups: Optional[List[OrFilterGroup]] = Field(
        None, description="Explicit OR filter groups"
    )
    not_conditions: Optional[List[FilterCondition]] = Field(
        None, description="Explicit NOT filter conditions"
    )

    def is_empty(self) -> bool:
        """Return True when no filter criteria are set."""
        scalar_fields = (
            self.minimum_experience,
            self.maximum_experience,
            self.location,
            self.degree,
            self.current_company,
            self.previous_company,
            self.salary_min,
            self.salary_max,
            self.notice_period,
            self.work_mode,
            self.employment_type,
            self.availability,
        )
        list_fields = (
            self.preferred_locations,
            self.skills,
            self.excluded_skills,
            self.education,
            self.certifications,
            self.languages,
        )
        if any(v is not None for v in scalar_fields):
            return False
        if any(v for v in list_fields):
            return False
        if self.custom_filters:
            return False
        if self.or_groups:
            return False
        if self.not_conditions:
            return False
        return True

    class Config:
        frozen = False


class CandidateMetadata(BaseModel):
    """
    Metadata profile for a candidate used during pre-retrieval filtering.

    Attributes map to MetadataFilter fields for symmetric comparison.
    """

    candidate_id: str = Field(..., description="Unique candidate identifier")
    resume_id: str = Field(..., description="Resume identifier")
    candidate_name: Optional[str] = Field(None, description="Candidate display name")
    experience_years: Optional[float] = Field(None, ge=0, description="Total experience in years")
    location: Optional[str] = Field(None, description="Primary location")
    preferred_locations: List[str] = Field(default_factory=list, description="Preferred work locations")
    skills: List[str] = Field(default_factory=list, description="Candidate skills")
    education: List[str] = Field(default_factory=list, description="Education entries")
    degree: Optional[str] = Field(None, description="Highest or relevant degree")
    current_company: Optional[str] = Field(None, description="Current employer")
    previous_companies: List[str] = Field(default_factory=list, description="Previous employers")
    salary_expectation: Optional[float] = Field(None, ge=0, description="Expected salary in LPA")
    notice_period_days: Optional[int] = Field(None, ge=0, description="Notice period in days")
    work_mode: Optional[str] = Field(None, description="Preferred work mode")
    employment_type: Optional[str] = Field(None, description="Preferred employment type")
    certifications: List[str] = Field(default_factory=list, description="Certifications held")
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    availability: Optional[str] = Field(None, description="Availability status")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    class Config:
        frozen = True


class FilterResult(BaseModel):
    """Output of a metadata filtering operation."""

    candidate_ids: List[str] = Field(..., description="Filtered candidate IDs for retrieval")
    total_before: int = Field(..., ge=0, description="Candidate count before filtering")
    total_after: int = Field(..., ge=0, description="Candidate count after filtering")
    filters_applied: int = Field(..., ge=0, description="Number of active filter criteria")
    parse_latency_ms: Optional[float] = Field(None, ge=0, description="Filter parsing latency")
    filter_latency_ms: float = Field(..., ge=0, description="Filter application latency")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")

    class Config:
        frozen = True


class ParseResult(BaseModel):
    """Output of filter parsing from a recruiter query."""

    filters: MetadataFilter = Field(..., description="Extracted metadata filter")
    raw_query: str = Field(..., description="Original recruiter query")
    parse_latency_ms: float = Field(..., ge=0, description="Parsing latency in milliseconds")
    parser_backend: str = Field(default="rule_based", description="Parser backend used")
    cache_hit: bool = Field(default=False, description="Whether parse was served from cache")

    class Config:
        frozen = True
