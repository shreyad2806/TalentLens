"""
Cache module - In-memory cache for embedding results.

This module provides an in-memory cache to avoid duplicate inference for identical
chunk text. When the same text is embedded multiple times, the cached embedding
is returned instead of re-running the model.

The cache uses a simple dictionary-based approach with text as the key and
the embedding vector as the value.
"""

from typing import List, Optional
from threading import Lock
import hashlib


class EmbeddingCache:
    """
    In-memory cache for embedding results.
    
    This class provides a thread-safe in-memory cache to store embedding results
    keyed by the text content. When the same text is requested again, the cached
    embedding is returned, avoiding duplicate model inference.
    
    The cache is useful for:
    - Avoiding duplicate inference for identical chunks
    - Improving performance in batch processing
    - Reducing computational costs
    """
    
    def __init__(self):
        """
        Initialize the embedding cache.
        """
        self._cache: dict = {}
        self._lock: Lock = Lock()
        self._hits: int = 0
        self._misses: int = 0
    
    def _generate_key(self, text: str) -> str:
        """
        Generate a cache key from text.
        
        Uses SHA256 hash to generate a unique key for the text.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for the given text.
        
        Args:
            text: The text to look up in the cache
            
        Returns:
            Cached embedding vector if found, None otherwise
        """
        key = self._generate_key(text)
        
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding for the given text.
        
        Args:
            text: The text to cache the embedding for
            embedding: The embedding vector to cache
        """
        key = self._generate_key(text)
        
        with self._lock:
            self._cache[key] = embedding
    
    def has(self, text: str) -> bool:
        """
        Check if the cache contains an embedding for the given text.
        
        Args:
            text: The text to check
            
        Returns:
            True if cached, False otherwise
        """
        key = self._generate_key(text)
        
        with self._lock:
            return key in self._cache
    
    def clear(self) -> None:
        """
        Clear all cached embeddings.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def size(self) -> int:
        """
        Get the number of cached embeddings.
        
        Returns:
            Number of cached embeddings
        """
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }


# Global singleton cache instance
_embedding_cache: Optional[EmbeddingCache] = None
_embedding_cache_lock: Lock = Lock()


def get_embedding_cache() -> EmbeddingCache:
    """
    Get the singleton EmbeddingCache instance.
    
    This function provides thread-safe access to the singleton EmbeddingCache instance.
    
    Returns:
        The singleton EmbeddingCache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        with _embedding_cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache()
    return _embedding_cache
