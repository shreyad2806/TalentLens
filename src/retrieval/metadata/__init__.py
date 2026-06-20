"""
Metadata Filtering Package.

This package provides production-grade metadata filtering that runs BEFORE
dense and sparse retrieval. It parses recruiter queries into structured filters,
validates them, and returns filtered candidate IDs for downstream retrieval.

Components:
- schema: Pydantic schemas for filters, candidates, and results
- filter_parser: Regex/rule-based query parser (LLM-upgradeable)
- filter_engine: Sequential filter application with AND/OR/NOT support
- validator: Filter and candidate validation
- cache: LRU cache with TTL for parsed filters and filter results
- metadata_service: Main facade — application code calls this only

Usage:
    from src.retrieval.metadata import MetadataService, MetadataFilter

    service = MetadataService(cache_enabled=True)

    # Parse + filter from recruiter query
    result = service.filter_candidates(
        candidates=candidate_pool,
        query="Senior Python Developer in Bangalore with 5+ years under 25 LPA",
    )
    candidate_ids = result.candidate_ids  # Pass to dense/BM25 retrieval

    # Or apply pre-built filters
    filters = MetadataFilter(minimum_experience=5, skills=["Python"])
    result = service.apply_filters(filters, candidate_pool)

Pipeline Position:
    Recruiter Query → MetadataService → Filtered Candidate IDs → Dense/BM25/Hybrid

Architecture:
- Facade pattern for clean API
- Strategy pattern for parser upgrade path
- Caching for parse and filter results
- Comprehensive logging and metrics
- Independent from retrieval and reranking layers

SOLID Principles:
- Single Responsibility: Each component has one clear purpose
- Open/Closed: Parser swappable; new filters extend schema
- Liskov Substitution: Parser strategies interchangeable
- Interface Segregation: Focused public API on MetadataService
- Dependency Inversion: Service depends on abstractions
"""

from .schema import (
    CandidateMetadata,
    FilterCondition,
    FilterLogic,
    FilterOperator,
    FilterResult,
    MetadataFilter,
    OrFilterGroup,
    ParseResult,
)
from .filter_parser import (
    FilterParser,
    FilterParserStrategy,
    RuleBasedFilterParser,
)
from .filter_engine import FilterEngine
from .validator import MetadataFilterValidator, ValidationError
from .cache import MetadataFilterCache
from .metadata_service import MetadataService

__all__ = [
    # Schema
    "CandidateMetadata",
    "FilterCondition",
    "FilterLogic",
    "FilterOperator",
    "FilterResult",
    "MetadataFilter",
    "OrFilterGroup",
    "ParseResult",
    # Parser
    "FilterParser",
    "FilterParserStrategy",
    "RuleBasedFilterParser",
    # Engine
    "FilterEngine",
    # Validation
    "MetadataFilterValidator",
    "ValidationError",
    # Cache
    "MetadataFilterCache",
    # Service
    "MetadataService",
]
