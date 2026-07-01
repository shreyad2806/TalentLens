"""
Dense Retrieval Service for Production Semantic Search.

This module provides the main DenseRetrievalService that orchestrates
the complete dense retrieval pipeline: query embedding, vector search,
score normalization, candidate aggregation, and result formatting.

Architecture Notes:
- Facade Pattern for retrieval pipeline
- Orchestrates multiple components
- Comprehensive logging
- Performance metrics tracking
- Application code only calls this service

SOLID Principles Applied:
- Single Responsibility: Orchestrates retrieval pipeline
- Open/Closed: Open for extension with new components
- Dependency Inversion: Depends on component abstractions
- Interface Segregation: Focused service interface
"""

import logging
import time
from typing import List, Dict, Any, Optional
from .schema import DenseSearchResult, RetrievalMetrics
from .validator import RetrievalValidator, ValidationError
from .cache import QueryCache
from .score_normalizer import ScoreNormalizer, NormalizationStrategy
from .candidate_aggregator import CandidateAggregator
from .query_embedder import QueryEmbedder
from src.vector_store import VectorStoreService
from src.embeddings.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class DenseRetrievalService:
    """
    Production Dense Retrieval Service.
    
    This service provides a complete dense retrieval pipeline for semantic
    search over resume data. It orchestrates query embedding, vector search,
    score normalization, candidate aggregation, and result formatting.
    
    Architecture Pattern: Facade Pattern
    - Simplifies complex retrieval pipeline
    - Orchestrates multiple components
    - Provides single entry point for applications
    - Handles all retrieval complexity internally
    
    Pipeline:
        1. Query validation
        2. Query embedding (with cache)
        3. Vector store query
        4. Score normalization
        5. Candidate aggregation
        6. Result formatting
        7. Metrics logging
    
    Features:
        - Query embedding with caching
        - Vector similarity search
        - Score normalization
        - Candidate-level aggregation
        - Comprehensive logging
        - Performance metrics
    """
    
    def __init__(
        self,
        vector_store_service: Optional[VectorStoreService] = None,
        embedding_service: Optional[EmbeddingService] = None,

        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.COSINE,
        section_weights: Optional[Dict[str, float]] = None
    ):

        """
        Initialize the dense retrieval service.
        
        Args:
            vector_store_service: Optional vector store service. If None, creates default.
            cache_enabled: Whether to enable query caching (default: True)
            cache_max_size: Maximum cache size (default: 1000)
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
            normalization_strategy: Score normalization strategy (default: COSINE)
            section_weights: Section weights for candidate aggregation
        """
        # Initialize components
        # Dependency injection: never instantiate dependent services here when provided.
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.validator = RetrievalValidator(vector_dimension=self.vector_store_service.config.dimension)

        # Query embedder depends on EmbeddingService
        if embedding_service is not None:
            self.query_embedder = QueryEmbedder(
                expected_dimension=self.vector_store_service.config.dimension,
                embedding_service=embedding_service,
            )

        else:
            self.query_embedder = QueryEmbedder()

        self.score_normalizer = ScoreNormalizer(strategy=normalization_strategy)
        self.candidate_aggregator = CandidateAggregator(section_weights=section_weights)

        
        # Initialize cache
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = QueryCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
        else:
            self.cache = None
        
        logger.info(
            f"DenseRetrievalService initialized with cache_enabled={cache_enabled}, "
            f"normalization_strategy={normalization_strategy.value}"
        )

    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[DenseSearchResult]:
        """
        Perform dense semantic search.
        
        This is the main entry point for the retrieval service. It performs
        the complete retrieval pipeline and returns formatted results.
        
        Args:
            query: Search query
            top_k: Number of results to return (default: 10)
            filters: Optional metadata filters
            
        Returns:
            List of DenseSearchResult objects
            
        Raises:
            ValidationError: If input validation fails
            RuntimeError: If retrieval pipeline fails
        """
        # Validate inputs
        self.validator.validate_query(query)
        self.validator.validate_top_k(top_k)
        self.validator.validate_filters(filters)
        
        # Check cache
        if self.cache_enabled and self.cache:
            cached_results = self.cache.get(query, filters, top_k)
            if cached_results is not None:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results
        
        # Track metrics
        start_time = time.time()
        embedding_latency = 0.0
        vector_latency = 0.0
        aggregation_latency = 0.0
        
        try:
            # Step 1: Generate query embedding
            embedding_start = time.time()
            query_vector = self.query_embedder.embed_query(query)
            embedding_latency = time.time() - embedding_start
            
            logger.debug(f"Query embedding generated in {embedding_latency:.3f}s")
            
            # Step 2: Query vector store
            vector_start = time.time()
            vector_results = self.vector_store_service.query(query_vector, k=top_k, filters=filters)
            vector_latency = time.time() - vector_start
            
            logger.debug(f"Vector store query completed in {vector_latency:.3f}s, returned {len(vector_results)} results")
            
            # Step 3: Normalize scores
            raw_scores = [result['score'] for result in vector_results]
            normalized_scores = self.score_normalizer.normalize(raw_scores)
            
            # Step 4: Convert to DenseSearchResult
            aggregation_start = time.time()
            search_results = self._convert_to_dense_results(query, vector_results, normalized_scores)
            aggregation_latency = time.time() - aggregation_start
            
            # Step 5: Sort by normalized score
            search_results.sort(key=lambda x: x.normalized_score, reverse=True)
            
            # Step 6: Reassign ranks after sorting
            # Since DenseSearchResult is frozen, we need to recreate with new ranks
            search_results = [
                DenseSearchResult(
                    query=result.query,
                    candidate_name=result.candidate_name,
                    resume_id=result.resume_id,
                    chunk_id=result.chunk_id,
                    section=result.section,
                    score=result.score,
                    normalized_score=result.normalized_score,
                    metadata=result.metadata,
                    matched_text=result.matched_text,
                    rank=i
                )
                for i, result in enumerate(search_results)
            ]
            
            # Cache results
            if self.cache_enabled and self.cache:
                self.cache.set(query, search_results, filters, top_k)
            
            # Calculate total latency
            total_latency = time.time() - start_time
            
            # Log metrics
            self._log_metrics(
                query_latency=total_latency,
                embedding_latency=embedding_latency,
                vector_latency=vector_latency,
                aggregation_latency=aggregation_latency,
                total_latency=total_latency,
                retrieved_chunks=len(search_results),
                candidates=len(set(r.resume_id for r in search_results)),
                cache_hit=False
            )
            
            logger.info(
                f"Search completed for query: {query[:50]}... "
                f"returned {len(search_results)} results in {total_latency:.3f}s"
            )
            
            return search_results
            
        except Exception as e:
            total_latency = time.time() - start_time
            logger.error(f"Search failed for query: {query[:50]}... after {total_latency:.3f}s: {e}")
            raise RuntimeError(f"Search failed: {e}") from e
    
    def search_aggregated(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[DenseSearchResult]:
        """
        Perform dense semantic search with candidate aggregation.
        
        This method performs search and then aggregates results by candidate
        to provide candidate-level scores instead of chunk-level scores.
        
        Args:
            query: Search query
            top_k: Number of candidates to return (default: 10)
            filters: Optional metadata filters
            
        Returns:
            List of DenseSearchResult objects with aggregated scores
        """
        # Perform regular search with more chunks to get better aggregation
        # Use 3x top_k to get enough chunks for aggregation
        chunk_results = self.search(query, top_k=top_k * 3, filters=filters)
        
        if not chunk_results:
            return []
        
        # Aggregate by candidate
        aggregated_candidates = self.candidate_aggregator.aggregate(chunk_results)
        
        # Convert aggregated results back to DenseSearchResult
        # Use the top chunk as the representative
        final_results = []
        for i, agg_candidate in enumerate(aggregated_candidates[:top_k]):
            # Get the top evidence chunk
            top_evidence = max(agg_candidate.evidence_chunks, key=lambda x: x['score'])
            
            result = DenseSearchResult(
                query=query,
                candidate_name=agg_candidate.candidate_name,
                resume_id=agg_candidate.resume_id,
                chunk_id=top_evidence['chunk_id'],
                section=top_evidence['section'],
                score=top_evidence['score'],
                normalized_score=agg_candidate.final_score,
                metadata={
                    'section_scores': agg_candidate.section_scores,
                    'num_chunks': agg_candidate.metadata['num_chunks'],
                    'aggregated': True
                },
                matched_text=top_evidence['matched_text'],
                rank=i
            )
            final_results.append(result)
        
        logger.info(
            f"Aggregated search completed: {len(final_results)} candidates from "
            f"{len(chunk_results)} chunks"
        )
        
        return final_results
    
    def _convert_to_dense_results(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        normalized_scores: List[float]
    ) -> List[DenseSearchResult]:
        """
        Convert vector store results to DenseSearchResult objects.
        
        Args:
            query: Original search query
            vector_results: Results from vector store
            normalized_scores: Normalized scores
            
        Returns:
            List of DenseSearchResult objects
        """
        search_results = []
        
        for i, (vector_result, normalized_score) in enumerate(zip(vector_results, normalized_scores)):
            # Handle different vector store result formats
            # Memory adapter returns: {"id": ..., "score": ..., "metadata": ...}
            # Pinecone/Qdrant adapters may return different formats
            
            if 'record' in vector_result:
                # Format with VectorRecord object
                record = vector_result['record']
                score = vector_result['score']
                metadata = record.metadata
                resume_id = record.resume_id
                chunk_id = record.chunk_id
                candidate_name = record.candidate_name
                section = record.section
            else:
                # Format with metadata directly (memory adapter)
                metadata = vector_result.get('metadata', {})
                score = vector_result.get('score', 0.0)
                resume_id = metadata.get('resume_id', '')
                chunk_id = vector_result.get('id', '')
                candidate_name = metadata.get('candidate_name', '')
                section = metadata.get('section', '')
            
            result = DenseSearchResult(
                query=query,
                candidate_name=candidate_name,
                resume_id=resume_id,
                chunk_id=chunk_id,
                section=section,
                score=score,
                normalized_score=normalized_score,
                metadata=metadata,
                matched_text=metadata.get('text_preview', ''),
                rank=i
            )
            
            search_results.append(result)
        
        return search_results
    
    def _log_metrics(
        self,
        query_latency: float,
        embedding_latency: float,
        vector_latency: float,
        aggregation_latency: float,
        total_latency: float,
        retrieved_chunks: int,
        candidates: int,
        cache_hit: bool
    ) -> None:
        """
        Log retrieval performance metrics.
        
        Args:
            query_latency: Total query latency
            embedding_latency: Query embedding latency
            vector_latency: Vector store query latency
            aggregation_latency: Candidate aggregation latency
            total_latency: Total end-to-end latency
            retrieved_chunks: Number of chunks retrieved
            candidates: Number of unique candidates
            cache_hit: Whether query was served from cache
        """
        logger.info(
            f"Retrieval metrics: "
            f"query_latency={query_latency:.3f}s, "
            f"embedding_latency={embedding_latency:.3f}s, "
            f"vector_latency={vector_latency:.3f}s, "
            f"aggregation_latency={aggregation_latency:.3f}s, "
            f"total_latency={total_latency:.3f}s, "
            f"retrieved_chunks={retrieved_chunks}, "
            f"candidates={candidates}, "
            f"cache_hit={cache_hit}"
        )
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary, or None if cache is disabled
        """
        if self.cache_enabled and self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("Query cache cleared")
    
    def close(self) -> None:
        """Close the retrieval service and release resources."""
        self.vector_store_service.close()
        logger.info("DenseRetrievalService closed")
