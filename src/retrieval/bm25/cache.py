"""
Cache module - In-memory cache for BM25 search results.

This module provides the BM25Cache class for caching search query results
to avoid redundant computations and improve performance.
"""

import time
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import logging

from .schema import BM25Document

logger = logging.getLogger(__name__)


class SearchResult:
    """
    Represents a search result with document and score.
    """
    
    def __init__(self, document: BM25Document, score: float, rank: int):
        """
        Initialize a search result.
        
        Args:
            document: BM25Document object
            score: BM25 relevance score
            rank: Rank in the results (1-based)
        """
        self.document = document
        self.score = score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search result to dictionary.
        
        Returns:
            Dictionary representation of the search result
        """
        return {
            'rank': self.rank,
            'score': self.score,
            'document': self.document.to_dict()
        }


class BM25Cache:
    """
    In-memory cache for BM25 search results.
    
    This class implements an LRU (Least Recently Used) cache for storing
    search query results to improve performance by avoiding redundant computations.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries in the cache (default: 1000)
            ttl: Time-to-live for cache entries in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        
        # OrderedDict for LRU cache
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"BM25Cache initialized with max_size={max_size}, ttl={ttl}s")
    
    def get(self, key: str) -> Optional[List[SearchResult]]:
        """
        Get cached results for a query.
        
        Args:
            key: Cache key (typically generated from query parameters)
            
        Returns:
            List of SearchResult objects if found and not expired, None otherwise
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if entry has expired
        if time.time() - entry['timestamp'] > self.ttl:
            del self.cache[key]
            self.misses += 1
            logger.debug(f"Cache entry expired for key: {key}")
            return None
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        self.hits += 1
        
        logger.debug(f"Cache hit for key: {key}")
        return entry['results']
    
    def set(self, key: str, results: List[SearchResult]) -> None:
        """
        Cache search results for a query.
        
        Args:
            key: Cache key
            results: List of SearchResult objects to cache
        """
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.evictions += 1
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        # Add new entry
        self.cache[key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        
        logger.debug(f"Cached results for key: {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
        
        return {
            'enabled': True,
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'ttl': self.ttl
        }
    
    def remove(self, key: str) -> bool:
        """
        Remove a specific entry from the cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed, False if key not found
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Removed cache entry for key: {key}")
            return True
        return False
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self.cache
    
    def __len__(self) -> int:
        """Get the current cache size."""
        return len(self.cache)
