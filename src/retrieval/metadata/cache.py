"""
Cache for Metadata Filtering Engine.

Provides LRU caching with TTL for:
- Parsed recruiter filters (query → MetadataFilter)
- Filter application results (filter + candidate pool → candidate IDs)

Architecture Notes:
- Separate caches for parse and filter results
- Configurable max size and TTL
- Cache hit/miss tracking for observability

SOLID Principles Applied:
- Single Responsibility: Caching only
- Open/Closed: Open for new cache strategies
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .schema import FilterResult, MetadataFilter, ParseResult

logger = logging.getLogger(__name__)


class _LRUCache:
    """Internal LRU cache with TTL expiration."""

    def __init__(self, max_size: int, ttl: int, name: str) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")

        self.max_size = max_size
        self.ttl = ttl
        self.name = name
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            self.misses += 1
            return None

        entry = self._store[key]
        if time.time() - entry["timestamp"] > self.ttl:
            del self._store[key]
            self.misses += 1
            logger.debug(f"{self.name} cache entry expired for key={key[:40]}...")
            return None

        self._store.move_to_end(key)
        self.hits += 1
        return entry["value"]

    def put(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]

        self._store[key] = {"value": value, "timestamp": time.time()}

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class MetadataFilterCache:
    """
    Dual-cache for metadata filtering operations.

    Caches:
        1. Parsed filters from recruiter queries
        2. Filter results from filter + candidate pool combinations
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        parse_max_size: Optional[int] = None,
        parse_ttl: Optional[int] = None,
        result_max_size: Optional[int] = None,
        result_ttl: Optional[int] = None,
    ) -> None:
        self.parse_cache = _LRUCache(
            max_size=parse_max_size or max_size,
            ttl=parse_ttl or ttl,
            name="ParseFilter",
        )
        self.result_cache = _LRUCache(
            max_size=result_max_size or max_size,
            ttl=result_ttl or ttl,
            name="FilterResult",
        )
        logger.info(
            f"MetadataFilterCache initialized — "
            f"parse(max={self.parse_cache.max_size}, ttl={self.parse_cache.ttl}s), "
            f"result(max={self.result_cache.max_size}, ttl={self.result_cache.ttl}s)"
        )

    @staticmethod
    def _normalize_query_key(query: str) -> str:
        return query.strip().lower()

    @staticmethod
    def _filter_hash(filters: MetadataFilter) -> str:
        payload = filters.model_dump(exclude_none=True)
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @staticmethod
    def _candidate_pool_hash(candidate_ids: List[str]) -> str:
        sorted_ids = sorted(candidate_ids)
        serialized = json.dumps(sorted_ids)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def get_parsed_filter(self, query: str) -> Optional[ParseResult]:
        """
        Retrieve cached parse result for a recruiter query.

        Args:
            query: Recruiter query string

        Returns:
            Cached ParseResult or None
        """
        key = self._normalize_query_key(query)
        cached = self.parse_cache.get(key)
        if cached is not None:
            logger.info(f"Parse cache hit for query: {query[:60]}...")
        return cached

    def put_parsed_filter(self, query: str, result: ParseResult) -> None:
        """
        Cache a parse result for a recruiter query.

        Args:
            query: Recruiter query string
            result: ParseResult to cache
        """
        key = self._normalize_query_key(query)
        self.parse_cache.put(key, result)
        logger.debug(f"Cached parsed filter for query: {query[:60]}...")

    def get_filter_result(
        self,
        filters: MetadataFilter,
        candidate_ids: List[str],
    ) -> Optional[FilterResult]:
        """
        Retrieve cached filter result.

        Args:
            filters: Applied metadata filter
            candidate_ids: Input candidate pool IDs

        Returns:
            Cached FilterResult or None
        """
        key = f"{self._filter_hash(filters)}:{self._candidate_pool_hash(candidate_ids)}"
        cached = self.result_cache.get(key)
        if cached is not None:
            logger.info(
                f"Filter result cache hit — "
                f"filter_hash={self._filter_hash(filters)}, "
                f"pool_size={len(candidate_ids)}"
            )
        return cached

    def put_filter_result(
        self,
        filters: MetadataFilter,
        candidate_ids: List[str],
        result: FilterResult,
    ) -> None:
        """
        Cache a filter application result.

        Args:
            filters: Applied metadata filter
            candidate_ids: Input candidate pool IDs
            result: FilterResult to cache
        """
        key = f"{self._filter_hash(filters)}:{self._candidate_pool_hash(candidate_ids)}"
        self.result_cache.put(key, result)
        logger.debug(f"Cached filter result for key={key}")

    def clear(self) -> None:
        """Clear both parse and result caches."""
        self.parse_cache.clear()
        self.result_cache.clear()
        logger.info("MetadataFilterCache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Return combined cache statistics."""
        return {
            "parse_cache": self.parse_cache.stats(),
            "result_cache": self.result_cache.stats(),
        }

    def update_ttl(self, ttl: int, cache_type: str = "both") -> None:
        """
        Update TTL for one or both caches.

        Args:
            ttl: New TTL in seconds
            cache_type: 'parse', 'result', or 'both'
        """
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")

        if cache_type in ("parse", "both"):
            self.parse_cache.ttl = ttl
        if cache_type in ("result", "both"):
            self.result_cache.ttl = ttl

        logger.info(f"MetadataFilterCache TTL updated to {ttl}s for {cache_type}")
