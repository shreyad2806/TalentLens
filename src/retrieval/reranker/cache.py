"""
Cache module for Reranker.

This module provides a caching mechanism for rerank scores to avoid repeated
inference for the same query-chunk pairs. It supports TTL (Time-To-Live)
for cache entries and thread-safe operations.

Architecture Notes:
- Thread-Safe: Uses threading.Lock for concurrent access
- TTL Support: Cache entries expire after specified time
- Memory-Efficient: Uses LRU eviction when cache is full
- Key-Value Store: Simple (query, chunk_id) -> score mapping

Cache Key Design:
The cache key is a tuple of (query, chunk_id) to uniquely identify a
reranking operation. This ensures that:
- Different queries with the same chunk are cached separately
- Same query with different chunks are cached separately
- The same query-chunk pair returns cached score

Cross-Encoder Caching Benefits:
Cross-encoder inference is computationally expensive compared to bi-encoder
retrieval. Caching rerank scores provides significant performance benefits:
1. Avoids repeated inference for common queries
2. Reduces latency for repeated searches
3. Lowers computational costs
4. Enables faster experimentation with different ranking strategies

SOLID Principles Applied:
- Single Responsibility: Only handles caching
- Open/Closed: Can be extended with different eviction policies
- Dependency Inversion: Depends on cache abstraction
- Interface Segregation: Focused cache interface
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Data class for a cache entry.
    
    This class represents a single entry in the cache, storing the score
    along with metadata for TTL and LRU eviction.
    
    Attributes:
        score: The rerank score
        timestamp: When the entry was created
        access_count: Number of times this entry has been accessed
        last_accessed: Last time this entry was accessed
    """
    
    score: float
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """
        Check if the cache entry has expired.
        
        Args:
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            True if entry has expired, False otherwise
        """
        return (time.time() - self.timestamp) > ttl_seconds
    
    def touch(self):
        """Update access timestamp and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class RerankCache:
    """
    Thread-safe cache for rerank scores with TTL and LRU eviction.
    
    This class provides a caching mechanism for rerank scores to avoid
    repeated inference for the same query-chunk pairs. It uses an OrderedDict
    to implement LRU (Least Recently Used) eviction when the cache is full.
    
    Architecture Pattern: Cache Pattern
    - Key-Value store with TTL
    - LRU eviction for memory management
    - Thread-safe operations
    - Statistics tracking
    
    Thread Safety:
    All public methods use a threading.Lock to ensure thread-safe access
    to the cache. This is important because the reranker service may be
    used in multi-threaded environments.
    
    TTL (Time-To-Live):
    Cache entries expire after a specified time to ensure freshness.
    Expired entries are removed on access and during cleanup operations.
    
    LRU Eviction:
    When the cache reaches its maximum size, the least recently used
    entries are evicted to make room for new entries. This ensures that
    frequently accessed entries remain in the cache.
    
    Attributes:
        max_size: Maximum number of entries in the cache
        ttl_seconds: Time-to-live for cache entries in seconds
        _cache: OrderedDict storing cache entries
        _lock: Thread lock for thread-safe operations
        _hits: Number of cache hits
        _misses: Number of cache misses
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the rerank cache.
        
        Args:
            max_size: Maximum number of entries in the cache (default: 1000)
            ttl_seconds: Time-to-live for cache entries in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[Tuple[str, str], CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        logger.info(
            f"RerankCache initialized with max_size={max_size}, "
            f"ttl={ttl_seconds}s"
        )
    
    def _make_key(self, query: str, chunk_id: str) -> Tuple[str, str]:
        """
        Create a cache key from query and chunk_id.
        
        The key is a tuple of (query, chunk_id) to uniquely identify
        a reranking operation.
        
        Args:
            query: The search query
            chunk_id: The chunk identifier
            
        Returns:
            Tuple (query, chunk_id) as cache key
        """
        return (query, chunk_id)
    
    def get(self, query: str, chunk_id: str) -> Optional[float]:
        """
        Get a rerank score from the cache.
        
        This method retrieves a score from the cache if it exists and has
        not expired. If the entry exists but has expired, it is removed
        from the cache and None is returned.
        
        Args:
            query: The search query
            chunk_id: The chunk identifier
            
        Returns:
            The cached score if found and not expired, None otherwise
        """
        key = self._make_key(query, chunk_id)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            # Check if entry has expired
            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache entry expired for key: {key}")
                return None
            
            # Update access time and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Cache hit for key: {key}")
            
            return entry.score
    
    def set(self, query: str, chunk_id: str, score: float) -> None:
        """
        Set a rerank score in the cache.
        
        This method stores a score in the cache. If the cache is full,
        the least recently used entry is evicted before adding the new entry.
        
        Args:
            query: The search query
            chunk_id: The chunk identifier
            score: The rerank score to cache
        """
        key = self._make_key(query, chunk_id)
        
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict the least recently used entry
                self._cache.popitem(last=False)
                logger.debug("Evicted LRU entry from cache")
            
            # Add or update the entry
            entry = CacheEntry(score=score, timestamp=time.time())
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            logger.debug(f"Cached score for key: {key}")
    
    def delete(self, query: str, chunk_id: str) -> bool:
        """
        Delete a cache entry.
        
        Args:
            query: The search query
            chunk_id: The chunk identifier
            
        Returns:
            True if entry was deleted, False if it didn't exist
        """
        key = self._make_key(query, chunk_id)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("RerankCache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        This method iterates through the cache and removes entries that
        have exceeded their TTL. It's useful for periodic maintenance.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.ttl_seconds)
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics including size, hit rate,
            miss rate, and other metrics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            miss_rate = self._misses / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "ttl_seconds": self.ttl_seconds
            }
    
    def get_size(self) -> int:
        """
        Get the current number of entries in the cache.
        
        Returns:
            Number of entries in the cache
        """
        with self._lock:
            return len(self._cache)
    
    def is_empty(self) -> bool:
        """
        Check if the cache is empty.
        
        Returns:
            True if cache is empty, False otherwise
        """
        with self._lock:
            return len(self._cache) == 0
