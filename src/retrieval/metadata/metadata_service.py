"""
Metadata Service — facade for the Metadata Filtering Engine.

This module provides the main MetadataService that orchestrates filter parsing,
validation, caching, and application. Application code should only call this service.

Pipeline (runs BEFORE dense/BM25 retrieval):
    1. Parse recruiter query → MetadataFilter (optional)
    2. Validate filters and candidates
    3. Check result cache
    4. Apply filters via FilterEngine
    5. Return filtered candidate IDs for retrieval

Architecture Notes:
- Facade pattern — single entry point for metadata filtering
- Retrieval layers remain independent; pass candidate IDs downstream
- Comprehensive logging for parsing time, counts, latency, cache hits

SOLID Principles Applied:
- Single Responsibility: Orchestrates metadata filtering pipeline
- Open/Closed: Parser strategy swappable via FilterParser
- Dependency Inversion: Depends on component abstractions
"""

import logging
import time
from typing import List, Optional, Union

from .cache import MetadataFilterCache
from .filter_engine import FilterEngine
from .filter_parser import FilterParser, FilterParserStrategy
from .schema import (
    CandidateMetadata,
    FilterResult,
    MetadataFilter,
    ParseResult,
)
from .validator import MetadataFilterValidator, ValidationError

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Production metadata filtering service.

    Exposes three public methods:
        - parse_filters(): Extract MetadataFilter from recruiter query
        - apply_filters(): Apply MetadataFilter to candidate pool
        - filter_candidates(): End-to-end parse + apply (convenience)

    Filtered candidate IDs are returned for downstream dense/sparse retrieval.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: int = 3600,
        parser_strategy: Optional[FilterParserStrategy] = None,
    ) -> None:
        """
        Initialize the metadata filtering service.

        Args:
            cache_enabled: Enable parse and result caching
            cache_max_size: Maximum cache entries per cache
            cache_ttl: Cache TTL in seconds
            parser_strategy: Optional custom parser strategy (e.g., future LLM parser)
        """
        self.parser = FilterParser(strategy=parser_strategy)
        self.engine = FilterEngine()
        self.validator = MetadataFilterValidator()
        self.cache_enabled = cache_enabled

        if cache_enabled:
            self.cache = MetadataFilterCache(max_size=cache_max_size, ttl=cache_ttl)
        else:
            self.cache = None

        logger.info(
            f"MetadataService initialized — cache_enabled={cache_enabled}, "
            f"cache_ttl={cache_ttl}s"
        )

    def parse_filters(self, query: str) -> ParseResult:
        """
        Parse a recruiter query into structured metadata filters.

        Args:
            query: Natural-language recruiter query

        Returns:
            ParseResult with MetadataFilter and parsing metrics

        Raises:
            ValidationError: If query or parsed filters are invalid
        """
        start = time.perf_counter()

        if self.cache_enabled and self.cache:
            cached = self.cache.get_parsed_filter(query)
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"parse_filters cache hit — query='{query[:60]}...', "
                    f"latency={latency_ms:.2f}ms"
                )
                return ParseResult(
                    filters=cached.filters,
                    raw_query=cached.raw_query,
                    parse_latency_ms=latency_ms,
                    parser_backend=cached.parser_backend,
                    cache_hit=True,
                )

        result = self.parser.parse(query)

        if self.cache_enabled and self.cache:
            self.cache.put_parsed_filter(query, result)

        logger.info(
            f"parse_filters complete — query='{query[:60]}...', "
            f"parse_latency={result.parse_latency_ms:.2f}ms, "
            f"cache_hit=False"
        )
        return result

    def apply_filters(
        self,
        filters: MetadataFilter,
        candidates: List[CandidateMetadata],
    ) -> FilterResult:
        """
        Apply metadata filters to a candidate pool.

        Args:
            filters: Structured metadata filter
            candidates: Full candidate metadata pool

        Returns:
            FilterResult with filtered candidate IDs and metrics

        Raises:
            ValidationError: If filters or candidates are invalid
        """
        start = time.perf_counter()
        candidate_ids = [c.candidate_id for c in candidates]

        if self.cache_enabled and self.cache:
            cached = self.cache.get_filter_result(filters, candidate_ids)
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    f"apply_filters cache hit — "
                    f"before={cached.total_before}, after={cached.total_after}, "
                    f"latency={latency_ms:.2f}ms"
                )
                return FilterResult(
                    candidate_ids=cached.candidate_ids,
                    total_before=cached.total_before,
                    total_after=cached.total_after,
                    filters_applied=cached.filters_applied,
                    filter_latency_ms=latency_ms,
                    cache_hit=True,
                )

        result = self.engine.apply(filters, candidates)

        if self.cache_enabled and self.cache:
            self.cache.put_filter_result(filters, candidate_ids, result)

        logger.info(
            f"apply_filters complete — "
            f"before={result.total_before}, after={result.total_after}, "
            f"filter_latency={result.filter_latency_ms:.2f}ms, "
            f"cache_hit=False"
        )
        return result

    def filter_candidates(
        self,
        candidates: List[CandidateMetadata],
        query: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> FilterResult:
        """
        End-to-end metadata filtering: parse query (optional) and apply filters.

        Provide either a recruiter query or a pre-built MetadataFilter.
        When both are provided, the explicit filters object takes precedence
        and the query is used only for logging.

        Args:
            candidates: Full candidate metadata pool
            query: Optional recruiter query to parse into filters
            filters: Optional pre-built MetadataFilter

        Returns:
            FilterResult with filtered candidate IDs ready for retrieval

        Raises:
            ValidationError: If neither query nor filters provided, or validation fails
        """
        pipeline_start = time.perf_counter()
        total_before = len(candidates)
        parse_latency_ms: Optional[float] = None

        if filters is None and query is None:
            raise ValidationError(
                "Either query or filters must be provided", field="filters"
            )

        resolved_filters: MetadataFilter
        if filters is not None:
            resolved_filters = filters
            self.validator.validate_filter(resolved_filters, allow_empty=True)
        else:
            parse_result = self.parse_filters(query)  # type: ignore[arg-type]
            resolved_filters = parse_result.filters
            parse_latency_ms = parse_result.parse_latency_ms

            if resolved_filters.is_empty():
                logger.info(
                    f"No filters extracted from query — returning all {total_before} candidates"
                )
                latency_ms = (time.perf_counter() - pipeline_start) * 1000
                return FilterResult(
                    candidate_ids=[c.candidate_id for c in candidates],
                    total_before=total_before,
                    total_after=total_before,
                    filters_applied=0,
                    parse_latency_ms=parse_latency_ms,
                    filter_latency_ms=latency_ms,
                    cache_hit=parse_result.cache_hit,
                )

        result = self.apply_filters(resolved_filters, candidates)

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"filter_candidates pipeline complete — "
            f"before={result.total_before}, after={result.total_after}, "
            f"parse_latency={parse_latency_ms or 0:.2f}ms, "
            f"filter_latency={result.filter_latency_ms:.2f}ms, "
            f"total_latency={total_latency_ms:.2f}ms, "
            f"cache_hit={result.cache_hit}"
        )

        return FilterResult(
            candidate_ids=result.candidate_ids,
            total_before=result.total_before,
            total_after=result.total_after,
            filters_applied=result.filters_applied,
            parse_latency_ms=parse_latency_ms,
            filter_latency_ms=result.filter_latency_ms,
            cache_hit=result.cache_hit,
        )

    def get_cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        if self.cache_enabled and self.cache:
            return self.cache.get_stats()
        return {
            "parse_cache": {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0},
            "result_cache": {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0},
        }

    def clear_cache(self) -> None:
        """Clear all metadata filter caches."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("MetadataService cache cleared")

    def set_parser_strategy(self, strategy: FilterParserStrategy) -> None:
        """
        Upgrade parser backend (e.g., switch to LLM parser).

        Args:
            strategy: New FilterParserStrategy implementation
        """
        self.parser.set_strategy(strategy)
        self.clear_cache()
        logger.info("MetadataService parser strategy updated — cache cleared")
