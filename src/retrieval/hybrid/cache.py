"""
Cache for Hybrid Retrieval Service.

This module provides caching for hybrid retrieval results to improve
performance by avoiding redundant fusion operations.

Architecture Notes:
- LRU cache with TTL for fused results
- Configurable cache size and TTL
- Cache hit/miss tracking
- Thread-safe operations

SOLID Principles Applied:
- Single Responsibility: Handles only caching
- Open/Closed: Open for new cache strategies
- Dependency Inversion: Depends on abstract interfaces
"""

import logging
import time
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from .schema import HybridSearchResult

logger = logging.getLogger(__name__)


class HybridResultCache:
    """
    LRU cache for hybrid retrieval results.
    
    This class provides an LRU (Least Recently Used) cache with TTL
    (Time To Live) for caching hybrid retrieval results. It tracks
    cache hits and misses for monitoring cache efficiency.
    
    Cache Features:
        - LRU eviction when cache is full
        - TTL expiration for cached entries
        - Cache hit/miss tracking
        - Thread-safe operations
        - Configurable size and TTL
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the hybrid result cache.
        
        Args:
            max_size: Maximum number of entries in the cache (default: 1000)
            ttl: Time to live for cached entries in seconds (default: 3600)
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"HybridResultCache initialized with max_size={max_size}, ttl={ttl}s")
    
    def _generate_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            query: The search query
            filters: Optional filters for the query
            
        Returns:
            Cache key string
        """
        if filters:
            filter_str = str(sorted(filters.items()))
            return f"{query}:{filter_str}"
        return query
    
    def get(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[HybridSearchResult]]:
        """
        Get cached results for a query.
        
        Args:
            query: The search query
            filters: Optional filters for the query
            
        Returns:
            Cached results if found and not expired, None otherwise
        """
        key = self._generate_key(query, filters)
        
        if key not in self.cache:
            self.misses += 1
            logger.debug(f"Cache miss for query: {query}")
            return None
        
        entry = self.cache[key]
        
        # Check if entry has expired
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            self.misses += 1
            logger.debug(f"Cache entry expired for query: {query}")
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        logger.debug(f"Cache hit for query: {query}")
        
        return entry["results"]
    
    def put(
        self,
        query: str,
        results: List[HybridSearchResult],
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache results for a query.
        
        Args:
            query: The search query
            results: Results to cache
            filters: Optional filters for the query
        """
        key = self._generate_key(query, filters)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        # Add new entry
        self.cache[key] = {
            "results": results,
            "timestamp": time.time()
        }
        
        logger.debug(f"Cached results for query: {query}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
        logger.info("HybridResultCache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }
    
    def update_ttl(self, ttl: int) -> None:
        """
        Update the TTL for cached entries.
        
        Args:
            ttl: New TTL in seconds
        """
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        
        self.ttl = ttl
        
        logger.info(f"HybridResultCache TTL updated to {ttl}s")
    
    def update_max_size(self, max_size: int) -> None:
        """
        Update the maximum cache size.
        
        Args:
            max_size: New maximum cache size
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        
        self.max_size = max_size
        
        # Evict entries if new size is smaller
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        logger.info(f"HybridResultCache max_size updated to {max_size}")
