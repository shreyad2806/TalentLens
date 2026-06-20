"""
Cache for Sparse Retrieval Service.

This module provides caching functionality for tokenized queries and search results
to avoid recomputation for repeated queries.

Architecture Notes:
- LRU Cache pattern for memory efficiency
- TTL (Time To Live) for cache expiration
- Thread-safe operations
- Configurable cache size and TTL
- Separate caches for tokens and results

SOLID Principles Applied:
- Single Responsibility: Handles only caching logic
- Open/Closed: Open for extension with different cache backends
- Dependency Inversion: Depends on cache interface abstraction
"""

import logging
import time
import hashlib
from typing import Optional, Dict, Any, List
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)


class QueryCache:
    """
    LRU cache for query results with TTL support.
    
    This class provides an in-memory LRU (Least Recently Used) cache with
    Time To Live (TTL) expiration for caching search results.
    
    Architecture Pattern: Cache Pattern
    - LRU eviction policy for memory efficiency
    - TTL for cache expiration
    - Thread-safe operations
    - Configurable size and TTL
    
    Features:
    - Caches search results
    - Automatic expiration based on TTL
    - LRU eviction when cache is full
    - Cache hit/miss tracking
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of items in cache (default: 1000)
            ttl_seconds: Time to live for cache items in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(
            f"QueryCache initialized with max_size={max_size}, ttl={ttl_seconds}s"
        )
    
    def _generate_key(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            query: Search query
            filters: Optional search filters
            top_k: Number of results to return
            
        Returns:
            Cache key as a string
        """
        # Create a deterministic key from query, filters, and top_k
        key_parts = [query.lower().strip(), str(top_k)]
        
        if filters:
            # Sort filters to ensure consistent key generation
            sorted_filters = sorted(filters.items())
            key_parts.append(str(sorted_filters))
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> Optional[List]:
        """
        Get cached results for a query.
        
        Args:
            query: Search query
            filters: Optional search filters
            top_k: Number of results to return
            
        Returns:
            Cached results if found and not expired, None otherwise
        """
        key = self._generate_key(query, filters, top_k)
        
        if key not in self.cache:
            self.misses += 1
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
        
        # Check if cache entry has expired
        cache_entry = self.cache[key]
        if time.time() - cache_entry['timestamp'] > self.ttl_seconds:
            # Entry expired, remove it
            del self.cache[key]
            self.misses += 1
            logger.debug(f"Cache entry expired for query: {query[:50]}...")
            return None
        
        # Cache hit - move to end to mark as recently used
        self.cache.move_to_end(key)
        self.hits += 1
        
        logger.debug(
            f"Cache hit for query: {query[:50]}... "
            f"(hits={self.hits}, misses={self.misses})"
        )
        
        return cache_entry['results']
    
    def set(self, query: str, results: List, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> None:
        """
        Cache results for a query.
        
        Args:
            query: Search query
            results: Search results to cache
            filters: Optional search filters
            top_k: Number of results returned
        """
        key = self._generate_key(query, filters, top_k)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache eviction: removed oldest entry")
        
        # Store the cache entry
        self.cache[key] = {
            'results': results,
            'timestamp': time.time(),
            'query': query,
            'filters': filters,
            'top_k': top_k
        }
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        
        logger.debug(f"Cached results for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
    
    def invalidate_query(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> bool:
        """
        Invalidate a specific query from cache.
        
        Args:
            query: Search query to invalidate
            filters: Optional search filters
            top_k: Number of results to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._generate_key(query, filters, top_k)
        
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache entry for query: {query[:50]}...")
            return True
        
        return False


class TokenCache:
    """
    LRU cache for tokenized queries.
    
    This class provides an in-memory LRU cache for tokenized queries to avoid
    recomputing tokenization for repeated queries.
    
    Architecture Pattern: Cache Pattern
    - LRU eviction policy for memory efficiency
    - TTL for cache expiration
    - Thread-safe operations
    - Configurable size and TTL
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the token cache.
        
        Args:
            max_size: Maximum number of items in cache (default: 1000)
            ttl_seconds: Time to live for cache items in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(
            f"TokenCache initialized with max_size={max_size}, ttl={ttl_seconds}s"
        )
    
    def _generate_key(self, query: str) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            query: Search query
            
        Returns:
            Cache key as a string
        """
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[str]]:
        """
        Get cached tokens for a query.
        
        Args:
            query: Search query
            
        Returns:
            Cached tokens if found and not expired, None otherwise
        """
        key = self._generate_key(query)
        
        if key not in self.cache:
            self.misses += 1
            logger.debug(f"Token cache miss for query: {query[:50]}...")
            return None
        
        # Check if cache entry has expired
        cache_entry = self.cache[key]
        if time.time() - cache_entry['timestamp'] > self.ttl_seconds:
            # Entry expired, remove it
            del self.cache[key]
            self.misses += 1
            logger.debug(f"Token cache entry expired for query: {query[:50]}...")
            return None
        
        # Cache hit - move to end to mark as recently used
        self.cache.move_to_end(key)
        self.hits += 1
        
        logger.debug(
            f"Token cache hit for query: {query[:50]}... "
            f"(hits={self.hits}, misses={self.misses})"
        )
        
        return cache_entry['tokens']
    
    def set(self, query: str, tokens: List[str]) -> None:
        """
        Cache tokens for a query.
        
        Args:
            query: Search query
            tokens: Tokenized query
        """
        key = self._generate_key(query)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Token cache eviction: removed oldest entry")
        
        # Store the cache entry
        self.cache[key] = {
            'tokens': tokens,
            'timestamp': time.time(),
            'query': query
        }
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        
        logger.debug(f"Cached tokens for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Token cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
