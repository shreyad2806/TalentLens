"""
Search Service module - BM25 search service with ranking and filtering.

This module provides the SearchService class that handles BM25 search queries,
ranking, filtering, and caching of results.
"""

import time
from typing import List, Dict, Any, Optional, Callable
import logging

from .bm25_index import BM25Index
from .schema import BM25Document
from .cache import BM25Cache, SearchResult

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for BM25 search with ranking and filtering.
    
    This class provides a high-level interface for searching the BM25 index,
    applying filters, caching results, and returning ranked results.
    """
    
    def __init__(self, index: BM25Index, enable_cache: bool = True):
        """
        Initialize the search service.
        
        Args:
            index: BM25Index to search
            enable_cache: Whether to enable result caching (default: True)
        """
        self.index = index
        self.enable_cache = enable_cache
        self.cache = BM25Cache() if enable_cache else None
        self.index_builder = None  # Will be set if needed for tokenization
        
        logger.info(f"SearchService initialized with cache={'enabled' if enable_cache else 'disabled'}")
    
    def set_index_builder(self, index_builder) -> None:
        """
        Set the index builder for tokenization.
        
        Args:
            index_builder: IndexBuilder instance for tokenizing queries
        """
        self.index_builder = index_builder
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize the query string.
        
        Args:
            query: Query string
            
        Returns:
            List of tokens
        """
        if self.index_builder:
            return self.index_builder.tokenize(query)
        else:
            # Simple fallback tokenization
            import re
            query = query.lower()
            query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
            return query.split()
    
    def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search the BM25 index for documents matching the query.
        
        Args:
            query: Search query string
            k: Number of top results to return (default: 10)
            filters: Optional filters to apply (e.g., {'section': 'skills'})
            
        Returns:
            List of SearchResult objects, sorted by score descending
        """
        start_time = time.time()
        
        # Tokenize query
        query_tokens = self._tokenize_query(query)
        
        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []
        
        # Check cache
        cache_key = self._generate_cache_key(query, k, filters)
        if self.enable_cache and self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: '{query}'")
                return cached_results
        
        # Perform search
        logger.info(f"Searching for: '{query}' (k={k})")
        scored_results = self.index.search(query_tokens, k=k * 2)  # Get more results for filtering
        
        # Convert to SearchResult objects
        results = []
        rank = 1
        for document_id, score in scored_results:
            document = self.index.get_document(document_id)
            if document:
                # Apply filters
                if filters and not self._apply_filters(document, filters):
                    continue
                
                result = SearchResult(document, score, rank)
                results.append(result)
                rank += 1
            
            # Stop if we have enough results
            if len(results) >= k:
                break
        
        # Cache results
        if self.enable_cache and self.cache:
            self.cache.set(cache_key, results)
        
        # Log performance
        query_latency = time.time() - start_time
        logger.info(f"Search completed in {query_latency:.3f}s, found {len(results)} results")
        
        return results
    
    def search_with_stats(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the BM25 index and return results with statistics.
        
        Args:
            query: Search query string
            k: Number of top results to return (default: 10)
            filters: Optional filters to apply
            
        Returns:
            Dictionary with 'results' and 'stats' keys
        """
        start_time = time.time()
        
        # Perform search
        results = self.search(query, k=k, filters=filters)
        
        # Calculate statistics
        query_latency = time.time() - start_time
        stats = {
            'query': query,
            'query_latency': query_latency,
            'num_results': len(results),
            'k': k,
            'filters_applied': filters is not None,
            'cache_enabled': self.enable_cache,
            'index_stats': self.index.get_statistics()
        }
        
        if results:
            stats['top_score'] = results[0].score
            stats['avg_score'] = sum(r.score for r in results) / len(results)
            stats['min_score'] = results[-1].score
        else:
            stats['top_score'] = 0.0
            stats['avg_score'] = 0.0
            stats['min_score'] = 0.0
        
        return {
            'results': results,
            'stats': stats
        }
    
    def _apply_filters(self, document: BM25Document, filters: Dict[str, Any]) -> bool:
        """
        Apply filters to a document.
        
        Args:
            document: BM25Document to filter
            filters: Dictionary of filter criteria
            
        Returns:
            True if document passes all filters, False otherwise
        """
        for key, value in filters.items():
            if key == 'section':
                if document.section != value:
                    return False
            elif key == 'candidate_name':
                if document.candidate_name.lower() != value.lower():
                    return False
            elif key == 'resume_id':
                if document.resume_id != value:
                    return False
            elif key in document.metadata:
                if document.metadata[key] != value:
                    return False
            else:
                # Unknown filter key, skip
                continue
        
        return True
    
    def _generate_cache_key(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> str:
        """
        Generate a cache key for a search query.
        
        Args:
            query: Search query string
            k: Number of results
            filters: Optional filters
            
        Returns:
            Cache key string
        """
        key_parts = [query.lower(), str(k)]
        if filters:
            # Sort filters for consistent key generation
            sorted_filters = sorted(filters.items())
            key_parts.extend([f"{k}:{v}" for k, v in sorted_filters])
        return "|".join(key_parts)
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache:
            return self.cache.get_stats()
        return {'enabled': False}
