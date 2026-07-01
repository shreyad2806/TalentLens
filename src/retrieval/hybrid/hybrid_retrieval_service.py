"""
Hybrid Retrieval Service.

This module provides the main HybridRetrievalService that combines dense
and sparse retrieval using Reciprocal Rank Fusion (RRF).

Architecture Notes:
- Facade pattern for hybrid retrieval
- Internally calls DenseRetrievalService and SparseRetrievalService
- Uses RRF for result fusion
- Caches fused results
- Validates all results

SOLID Principles Applied:
- Single Responsibility: Handles only hybrid retrieval orchestration
- Open/Closed: Open for new retrieval systems
- Dependency Inversion: Depends on retrieval service interfaces
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .schema import HybridSearchResult, FusionMetrics
from .fusion_service import FusionService
from .validator import HybridRetrievalValidator
from .cache import HybridResultCache

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """
    Hybrid retrieval service combining dense and sparse retrieval.
    
    This class provides the main interface for hybrid retrieval, combining
    results from dense and sparse retrieval systems using Reciprocal Rank
    Fusion (RRF). It serves as a facade for the hybrid retrieval pipeline.
    
    Retrieval Pipeline:
        1. Query validation
        2. Dense retrieval (via DenseRetrievalService)
        3. Sparse retrieval (via SparseRetrievalService)
        4. RRF fusion of results
        5. Result validation
        6. Result caching
        7. Return fused results
    
    Application code should only call this service for hybrid retrieval.
    """
    
    def __init__(
        self,
        dense_retrieval_service: Any,
        sparse_retrieval_service: Any,
        strategy: Optional[Any] = None,
        strategy_name: str = "rrf",
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """
        Initialize the hybrid retrieval service.
        
        Args:
            dense_retrieval_service: DenseRetrievalService instance
            sparse_retrieval_service: SparseRetrievalService instance
            strategy: Custom fusion strategy (optional)
            strategy_name: Name of fusion strategy to use (default: "rrf")
            cache_enabled: Whether to enable caching (default: True)
            cache_max_size: Maximum cache size (default: 1000)
            cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.dense_service = dense_retrieval_service
        self.sparse_service = sparse_retrieval_service
        self.fusion_service = FusionService(strategy=strategy, strategy_name=strategy_name)
        self.validator = HybridRetrievalValidator()
        
        if cache_enabled:
            self.cache = HybridResultCache(
                max_size=cache_max_size,
                ttl=cache_ttl
            )
        else:
            self.cache = None
        
        self.cache_enabled = cache_enabled
        
        # Temporary identity logging (dense/sparse services reference shared deps)
        if logger.isEnabledFor(logging.INFO):
            try:
                dense_bm25 = getattr(getattr(self.sparse_service, 'index', None), '__class__', None)
            except Exception:
                dense_bm25 = None
            logger.info(
                f"[IDENTITY] HybridRetrievalService dense_id={id(self.dense_service)} sparse_id={id(self.sparse_service)}"
            )

        logger.info(
            f"HybridRetrievalService initialized with strategy={strategy_name}, "
            f"cache_enabled={cache_enabled}"
        )

    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """
        Search using hybrid retrieval (dense + sparse + RRF).
        
        This method performs hybrid retrieval by:
        1. Checking cache for existing results
        2. Retrieving results from dense service
        3. Retrieving results from sparse service
        4. Fusing results using RRF
        5. Validating fused results
        6. Caching fused results
        7. Returning top-k results
        
        Args:
            query: The search query
            top_k: Number of results to return (default: 10)
            filters: Optional filters for the query
            
        Returns:
            List of HybridSearchResult objects
        """
        start_time = time.time()
        
        # Check cache
        if self.cache_enabled and self.cache:
            cached_results = self.cache.get(query, filters)
            if cached_results:
                logger.info(f"Cache hit for query: {query}")
                return cached_results[:top_k]
        
        logger.info(f"Hybrid search for query: {query}")
        
        # Dense retrieval
        dense_start = time.time()
        dense_results = self._retrieve_dense(query, top_k, filters)
        dense_latency = time.time() - dense_start
        
        # Sparse retrieval
        sparse_start = time.time()
        sparse_results = self._retrieve_sparse(query, top_k, filters)
        sparse_latency = time.time() - sparse_start
        
        # Fusion
        fusion_start = time.time()
        fused_results, metrics = self.fusion_service.fuse_results(
            dense_results,
            sparse_results,
            query
        )
        fusion_latency = time.time() - fusion_start
        
        # Update metrics
        metrics.dense_latency = dense_latency
        metrics.sparse_latency = sparse_latency
        metrics.total_latency = time.time() - start_time
        
        # Validate results
        self.validator.validate_results(fused_results, strict=False)
        
        # Cache results
        if self.cache_enabled and self.cache:
            self.cache.put(query, fused_results, filters)
        
        # Log metrics
        logger.info(
            f"Hybrid search completed for query: {query}... "
            f"returned {len(fused_results)} results in {metrics.total_latency:.3f}s "
            f"(dense: {metrics.dense_latency:.3f}s, "
            f"sparse: {metrics.sparse_latency:.3f}s, "
            f"fusion: {metrics.fusion_latency:.3f}s)"
        )
        
        logger.info(
            f"Fusion statistics: "
            f"overlap={metrics.overlap_count}, "
            f"dense_only={metrics.dense_only_count}, "
            f"sparse_only={metrics.sparse_only_count}"
        )
        
        # Return top-k results
        return fused_results[:top_k]
    
    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results from dense service.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            filters: Optional filters for the query
            
        Returns:
            List of dense retrieval results as dictionaries
        """
        try:
            results = self.dense_service.search(query, top_k=top_k, filters=filters)
            
            # Convert to dictionaries
            dense_results = []
            for idx, result in enumerate(results):
                dense_results.append({
                    "chunk_id": result.chunk_id,
                    "candidate_name": result.candidate_name,
                    "resume_id": result.resume_id,
                    "section": result.section,
                    "rank": idx,
                    "score": result.score,
                    "matched_text": result.matched_text,
                    "metadata": result.metadata
                })
            
            logger.info(f"Dense retrieval: {len(dense_results)} results")
            return dense_results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _retrieve_sparse(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results from sparse service.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            filters: Optional filters for the query
            
        Returns:
            List of sparse retrieval results as dictionaries
        """
        try:
            results = self.sparse_service.search(query, top_k=top_k, filters=filters)
            
            # Convert to dictionaries
            sparse_results = []
            for idx, result in enumerate(results):
                sparse_results.append({
                    "chunk_id": result.chunk_id,
                    "candidate_name": result.candidate_name,
                    "resume_id": result.resume_id,
                    "section": result.section,
                    "rank": idx,
                    "bm25_score": result.bm25_score,
                    "matched_text": result.matched_text,
                    "metadata": result.metadata
                })
            
            logger.info(f"Sparse retrieval: {len(sparse_results)} results")
            return sparse_results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache_enabled and self.cache:
            return self.cache.get_stats()
        return {
            "size": 0,
            "max_size": 0,
            "ttl": 0,
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("Hybrid retrieval cache cleared")
    
    def update_k(self, k: int) -> None:
        """
        Update the RRF constant k.
        
        Args:
            k: New constant for RRF calculation
        """
        self.fusion_service.update_k(k)
        self.k = k
        
        # Clear cache since k changed
        self.clear_cache()
        
        logger.info(f"HybridRetrievalService updated k to {k}")
