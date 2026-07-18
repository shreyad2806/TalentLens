"""
Reranker Service module for Cross-Encoder Reranking.

This module provides the main RerankerService that orchestrates the complete
reranking pipeline: validation, scoring, ranking, and result formatting.

Architecture Notes:
- Facade Pattern: Simplifies complex reranking pipeline
- Orchestrates multiple components
- Comprehensive logging
- Performance metrics tracking
- Application code only calls this service

Reranking Pipeline:
1. Input validation (query, candidates)
2. Check cache for existing scores
3. Score candidates with cross-encoder
4. Sort by rerank score
5. Format results as RerankedResult
6. Return reranked list

Cross-Encoder Reranking:
Cross-encoders take a query-document pair as input and output a relevance score.
Unlike bi-encoders (which encode queries and documents separately), cross-encoders
jointly process the pair, allowing for more accurate relevance assessment.

The reranking process:
1. Take top-k candidates from hybrid retrieval
2. For each candidate, create [query, candidate_text] pair
3. Pass through cross-encoder to get relevance score
4. Sort candidates by cross-encoder score
5. Return reranked list

SOLID Principles Applied:
- Single Responsibility: Orchestrates reranking pipeline
- Open/Closed: Open for extension with new components
- Dependency Inversion: Depends on component abstractions
- Interface Segregation: Focused service interface
"""

import logging
import time
from typing import List, Dict, Any, Optional
from .schema import RerankedResult, RerankMetrics, RerankEvidence
from .model_loader import ModelLoader, RerankerModel
from .cache import RerankCache
from .validator import RerankerValidator, ValidationError
from .batch_processor import BatchProcessor
from .scorer import CrossEncoderScorer
from src.debug_logger import log_stage_start, log_stage_end, log_error

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Production Reranker Service for cross-encoder reranking.
    
    This service provides a complete reranking pipeline for candidates
    retrieved from hybrid search. It orchestrates validation, scoring,
    ranking, and result formatting.
    
    Architecture Pattern: Facade Pattern
    - Simplifies complex reranking pipeline
    - Orchestrates multiple components
    - Provides single entry point for applications
    - Handles all reranking complexity internally
    
    Pipeline:
        1. Query validation
        2. Candidate validation
        3. Cache lookup
        4. Cross-encoder scoring
        5. Score normalization
        6. Ranking by score
        7. Result formatting
        8. Metrics logging
    
    Features:
        - Cross-encoder scoring with configurable models
        - Batch processing for efficient inference
        - Score caching with TTL
        - Comprehensive validation
        - Performance metrics tracking
        - Original score preservation
    """
    
    def __init__(
        self,
        model_name: str = RerankerModel.MINILM_V2.value,
        offline_mode: bool = False,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        batch_size: int = 32,
        normalize_scores: Optional[str] = None
    ):
        """
        Initialize the reranker service.
        
        Args:
            model_name: Name of the cross-encoder model (default: MINILM_V2)
            offline_mode: Whether to operate in offline mode (default: False)
            cache_enabled: Whether to enable score caching (default: True)
            cache_max_size: Maximum cache size (default: 1000)
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
            batch_size: Batch size for inference (default: 32)
            normalize_scores: Score normalization strategy (default: None)
        """
        # Initialize model loader
        self.model_loader = ModelLoader(
            model_name=model_name,
            offline_mode=offline_mode
        )
        
        # Initialize validator
        self.validator = RerankerValidator()
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = RerankCache(
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds
            )
        else:
            self.cache = None
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        # Initialize scorer
        self.scorer = CrossEncoderScorer(
            model_loader=self.model_loader,
            batch_processor=self.batch_processor,
            cache=self.cache,
            normalize=normalize_scores
        )
        
        logger.info(
            f"RerankerService initialized with model={model_name}, "
            f"cache_enabled={cache_enabled}, batch_size={batch_size}"
        )
    
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder.
        
        This is the main entry point for the reranker service. It performs
        the complete reranking pipeline and returns formatted results.
        
        Args:
            query: Search query
            candidates: List of candidates (HybridSearchResult objects)
            top_k: Number of results to return (default: all)
            
        Returns:
            List of RerankedResult objects sorted by rerank score
            
        Raises:
            ValidationError: If input validation fails
            RuntimeError: If reranking pipeline fails
        """
        # ── STAGE 10 — CROSS ENCODER RERANKER ─────────────────────────────────
        log_stage_start(10, "CROSS ENCODER RERANKER",
                        Query=query[:80],
                        Candidates_Before=len(candidates),
                        Top_K=top_k or "all",
                        Model=self.model_loader.model_name)
        
        # Track metrics
        start_time = time.perf_counter()
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Step 1: Validate inputs
            self.validator.validate_query(query)
            self.validator.validate_candidates(candidates)
            
            # Step 2: Score candidates
            scores = self.scorer.score_candidates(query, candidates)
            
            # Get cache statistics
            if self.cache:
                cache_stats = self.cache.get_stats()
                cache_hits = cache_stats.get('hits', 0)
                cache_misses = cache_stats.get('misses', 0)
            
            # Step 3: Sort by score (descending)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Step 4: Apply top_k if specified
            if top_k is not None:
                scored_candidates = scored_candidates[:top_k]
            
            # Step 5: Format results
            results = []
            for rank, (candidate, score) in enumerate(scored_candidates):
                # Extract matched text
                matched_text = self._extract_matched_text(candidate)
                
                # Get original rank and score if available
                original_rank = getattr(candidate, 'rank', None)
                original_score = getattr(candidate, 'rrf_score', None)
                
                result = RerankedResult(
                    query=query,
                    candidate_name=candidate.candidate_name,
                    resume_id=candidate.resume_id,
                    chunk_id=candidate.chunk_id,
                    section=candidate.section,
                    original_rank=original_rank,
                    original_score=original_score,
                    rerank_score=score,
                    final_rank=rank,
                    metadata=getattr(candidate, 'metadata', {}),
                    matched_text=matched_text,
                    evidence=RerankEvidence.CROSS_ENCODER
                )
                results.append(result)
            
            # Step 6: Log metrics
            total_time = time.perf_counter() - start_time
            self._log_metrics(
                len(candidates),
                len(results),
                cache_hits,
                cache_misses,
                total_time,
                scores
            )
            
            # Stage 10 END banner
            top10_scores = [f"{r.rerank_score:.4f}" for r in results[:10]]
            sample_result = None
            if results:
                sample_result = {
                    "Top_1_ID": results[0].resume_id,
                    "Top_1_Name": results[0].candidate_name,
                    "Top_1_Rerank": f"{results[0].rerank_score:.4f}",
                }
            
            log_stage_end(10, "CROSS ENCODER RERANKER", status="SUCCESS",
                          time_ms=total_time * 1000,
                          output_count=len(results),
                          sample=sample_result,
                          extra={
                              "Candidates_Before": len(candidates),
                              "Candidates_After": len(results),
                              "Top_10_Scores": top10_scores,
                          })
            
            return results
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            log_error(10, "CROSS ENCODER RERANKER", e, reraise=True)
            raise
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            log_error(10, "CROSS ENCODER RERANKER", e, reraise=True)
            raise RuntimeError(f"Reranking failed: {e}")
    
    def _extract_matched_text(self, candidate: Any) -> str:
        """
        Extract matched text from a candidate.
        
        Args:
            candidate: Candidate object
            
        Returns:
            Matched text string
        """
        # Try to get text from matched_chunks
        if hasattr(candidate, 'matched_chunks') and candidate.matched_chunks:
            texts = []
            for chunk in candidate.matched_chunks:
                if hasattr(chunk, 'matched_text'):
                    texts.append(chunk.matched_text)
            return " ".join(texts) if texts else ""
        
        # Try to get text from matched_text
        if hasattr(candidate, 'matched_text'):
            return candidate.matched_text
        
        # Try to get text from metadata
        if hasattr(candidate, 'metadata') and 'text' in candidate.metadata:
            return candidate.metadata['text']
        
        # Fallback to empty string
        return ""
    
    def _log_metrics(
        self,
        total_candidates: int,
        returned_candidates: int,
        cache_hits: int,
        cache_misses: int,
        total_time: float,
        scores: List[float]
    ) -> None:
        """
        Log reranking metrics.
        
        Args:
            total_candidates: Total number of candidates
            returned_candidates: Number of candidates returned
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            total_time: Total reranking time
            scores: List of rerank scores
        """
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        logger.info(
            f"Reranking completed: "
            f"candidates={total_candidates}, "
            f"returned={returned_candidates}, "
            f"cache_hits={cache_hits}, "
            f"cache_misses={cache_misses}, "
            f"cache_hit_rate={cache_hit_rate:.2%}, "
            f"avg_score={avg_score:.4f}, "
            f"score_range=[{min_score:.4f}, {max_score:.4f}], "
            f"total_time={total_time:.4f}s"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get reranker service metrics.
        
        Returns:
            Dictionary with service metrics including cache statistics,
            model information, and batch processor statistics
        """
        metrics = {
            "model_info": self.model_loader.get_model_info(),
            "cache_enabled": self.cache_enabled,
            "batch_processor_stats": self.batch_processor.get_stats()
        }
        
        if self.cache:
            metrics["cache_stats"] = self.cache.get_stats()
        
        return metrics
    
    def clear_cache(self) -> None:
        """Clear the rerank cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Reranker cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return self.model_loader.get_model_info()
