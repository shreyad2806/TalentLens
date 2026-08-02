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
import re
import time
from typing import Any

from src.debug_logger import log_error, log_stage_end, log_stage_start
from src.embeddings.embedding_service import EmbeddingService
from src.vector_store import VectorStoreService

from ...models import ResumeMetadata
from .cache import QueryCache
from .candidate_aggregator import CandidateAggregator
from .query_embedder import QueryEmbedder
from .schema import DenseSearchResult
from .score_normalizer import NormalizationStrategy, ScoreNormalizer
from .validator import RetrievalValidator

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
        vector_store_service: VectorStoreService | None = None,
        embedding_service: EmbeddingService | None = None,

        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.COSINE,
        section_weights: dict[str, float] | None = None
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
        if vector_store_service is None:
            raise ValueError("DenseRetrievalService requires an injected vector_store_service instance")
        self.vector_store_service = vector_store_service
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

    
    def search(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[DenseSearchResult]:
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
                log_stage_end(5, "EMBEDDING", status="SUCCESS", time_ms=0, output_count=len(cached_results),
                              sample={"source": "CACHE HIT"})
                log_stage_end(6, "DENSE RETRIEVAL", status="SUCCESS", time_ms=0, output_count=len(cached_results),
                              sample={"source": "CACHE HIT"})
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results
        
        # Track metrics
        start_time = time.perf_counter()
        embedding_latency = 0.0
        vector_latency = 0.0
        aggregation_latency = 0.0
        
        try:
            # ── STAGE 5 — EMBEDDING ─────────────────────────────────────────────
            log_stage_start(5, "EMBEDDING", Query=query[:80], Model="BAAI/bge-small-en-v1.5",
                            Embedding_Dimension=self.validator.vector_dimension)
            
            embedding_start = time.perf_counter()
            query_vector = self.query_embedder.embed_query(query)
            embedding_latency = time.perf_counter() - embedding_start
            
            log_stage_end(5, "EMBEDDING", status="SUCCESS",
                          time_ms=embedding_latency * 1000,
                          output_count=1,
                          sample={
                              "Vector_Shape": f"({len(query_vector)},)",
                              "First_5_Values": f"[{', '.join(f'{v:.4f}' for v in query_vector[:5])}]",
                          })
            
            # ── STAGE 6 — DENSE RETRIEVAL ───────────────────────────────────────
            logger.info(f"Incoming Filters: {filters}")
            log_stage_start(6, "DENSE RETRIEVAL", Top_K=top_k, Filters=filters)
            
            try:
                dense_count = self.vector_store_service.count()
                print(f"[DIAGNOSTIC][DenseRetrievalService] vector store count before query: {dense_count}")
            except Exception as e:
                print(f"[DIAGNOSTIC][DenseRetrievalService] vector store count unavailable: {e}")
            
            vector_start = time.perf_counter()
            vector_results = self.vector_store_service.query(query_vector, k=top_k, filters=filters)
            print(f"[DIAGNOSTIC][DenseRetrievalService] dense candidates returned: {len(vector_results)}")
            vector_latency = time.perf_counter() - vector_start
            
            logger.info(f"Applied Filters: {list(filters.keys()) if filters else 'None'}")
            logger.info(f"Remaining Candidates: {len(vector_results)}")
            
            # Normalize scores
            raw_scores = [result['score'] for result in vector_results]
            normalized_scores = self.score_normalizer.normalize(raw_scores)
            
            # Convert to DenseSearchResult
            aggregation_start = time.perf_counter()
            search_results = self._convert_to_dense_results(query, vector_results, normalized_scores)
            aggregation_latency = time.perf_counter() - aggregation_start
            
            # Sort by normalized score
            search_results.sort(key=lambda x: x.normalized_score, reverse=True)
            
            # Reassign ranks after sorting (DenseSearchResult is frozen)
            search_results = [
                DenseSearchResult(
                    query=result.query,
                    chunk_id=result.chunk_id,
                    section=result.section,
                    score=result.score,
                    normalized_score=result.normalized_score,
                    resume_metadata=result.resume_metadata,
                    matched_text=result.matched_text,
                    offset=result.offset,
                    rank=i
                )
                for i, result in enumerate(search_results)
            ]
            
            # Cache results
            if self.cache_enabled and self.cache:
                self.cache.set(query, search_results, filters, top_k)
            
            # Calculate total latency
            total_latency = time.perf_counter() - start_time
            
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
            
            # Stage 6 END banner
            top5_ids = [r.resume_id for r in search_results[:5]]
            top5_scores = [f"{r.normalized_score:.4f}" for r in search_results[:5]]
            sample_result = None
            if search_results:
                sample_result = {
                    "Top_1_ID": search_results[0].resume_id,
                    "Top_1_Name": search_results[0].candidate_name,
                    "Top_1_Score": f"{search_results[0].normalized_score:.4f}",
                }
            
            log_stage_end(6, "DENSE RETRIEVAL", status="SUCCESS",
                          time_ms=total_latency * 1000,
                          output_count=len(search_results),
                          sample=sample_result,
                          extra={
                              "Top_5_IDs": top5_ids,
                              "Top_5_Scores": top5_scores,
                              "Unique_Candidates": len(set(r.resume_id for r in search_results)),
                              "Vector_Search_Time_ms": f"{vector_latency * 1000:.1f}",
                          })
            
            return search_results
            
        except Exception as e:
            total_latency = time.perf_counter() - start_time
            logger.error(f"Search failed for query: {query[:50]}... after {total_latency:.3f}s: {e}")
            log_error(6, "DENSE RETRIEVAL", e, reraise=True)
            raise RuntimeError(f"Search failed: {e}") from e
    
    def search_aggregated(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[DenseSearchResult]:
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
            
            resume_metadata = ResumeMetadata.model_validate(
                {**agg_candidate.metadata, 'resume_id': agg_candidate.resume_id, 'candidate_name': agg_candidate.candidate_name}
            )

            result = DenseSearchResult(
                query=query,
                chunk_id=top_evidence['chunk_id'],
                section=top_evidence['section'],
                score=top_evidence['score'],
                normalized_score=agg_candidate.final_score,
                resume_metadata=resume_metadata,
                matched_text=top_evidence['matched_text'],
                rank=i
            )
            final_results.append(result)
        
        logger.info(
            f"Aggregated search completed: {len(final_results)} candidates from "
            f"{len(chunk_results)} chunks"
        )
        
        return final_results
    
    def _extract_matched_text(self, query: str, text: str | None) -> tuple[str, int]:
        """
        Extract a query-relevant snippet and its character offset from chunk text.

        If no query term is found, falls back to the first 300 characters.
        """
        if not text or not text.strip():
            return "", 0

        text = text.strip()
        text_lower = text.lower()
        terms = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 2]

        for term in terms:
            pos = text_lower.find(term)
            if pos != -1:
                start = max(0, pos - 80)
                end = min(len(text), pos + len(term) + 80)
                return text[start:end], start

        return text[:300], 0

    def _convert_to_dense_results(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
        normalized_scores: list[float]
    ) -> list[DenseSearchResult]:
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
                record = vector_result['record']
                score = vector_result['score']
                resume_metadata = record.resume_metadata
                chunk_id = record.chunk_id
                section = record.section
                text = resume_metadata.model_dump().get('text') or resume_metadata.model_dump().get('text_preview') or ""
            else:
                metadata = vector_result.get('metadata', {})
                score = vector_result.get('score', 0.0)
                chunk_id = vector_result.get('id', '')
                section = metadata.get('section', '')
                resume_metadata = ResumeMetadata.model_validate(metadata)
                text = metadata.get('text') or metadata.get('text_preview') or metadata.get('source_text') or metadata.get('chunk_text') or ""

            matched_text, offset = self._extract_matched_text(query, text)
            if not matched_text and 'metadata' in locals() and metadata.get('text_preview'):
                matched_text = metadata['text_preview'][:300]
                offset = 0

            result = DenseSearchResult(
                query=query,
                chunk_id=chunk_id,
                section=section,
                score=score,
                normalized_score=normalized_score,
                resume_metadata=resume_metadata,
                matched_text=matched_text,
                offset=offset,
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
    
    def get_cache_stats(self) -> dict[str, Any] | None:
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
