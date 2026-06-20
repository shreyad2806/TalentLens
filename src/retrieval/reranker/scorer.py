"""
Scorer module for Reranker.

This module provides cross-encoder scoring functionality for reranking
candidates, integrating with the model loader and batch processor.

Architecture Notes:
- Cross-Encoder Scoring: Uses cross-encoder models for relevance scoring
- Batch Processing: Leverages batch processor for efficient inference
- Cache Integration: Checks cache before inference
- Score Normalization: Normalizes scores for consistency

Cross-Encoder Scoring:
Cross-encoders take a query-document pair as input and output a relevance score.
The scoring process:
1. Create query-document pairs from candidates
2. Pass pairs through cross-encoder model
3. Get relevance scores from model output
4. Normalize scores if needed
5. Return scores for ranking

Score Interpretation:
Different cross-encoder models output scores in different ranges:
- MS MARCO models: Typically [0, 1] or [-10, 10]
- BGE rerankers: Typically [-inf, +inf] or [0, 1]
- The scorer handles normalization for consistent ranking

SOLID Principles Applied:
- Single Responsibility: Only handles scoring
- Open/Closed: Can be extended with different scoring strategies
- Dependency Inversion: Depends on model and batch processor abstractions
- Interface Segregation: Focused scoring interface
"""

import logging
import time
from typing import List, Any, Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderScorer:
    """
    Cross-encoder scorer for reranking candidates.
    
    This class provides scoring functionality using cross-encoder models.
    It integrates with the model loader for lazy model loading and the
    batch processor for efficient batched inference.
    
    Architecture Pattern: Strategy Pattern
    - Encapsulates scoring algorithm
    - Can be extended with different scoring strategies
    - Depends on model and batch processor abstractions
    - Provides consistent scoring interface
    
    Scoring Workflow:
    1. Check cache for existing scores
    2. For uncached candidates, create query-document pairs
    3. Process pairs through cross-encoder in batches
    4. Cache new scores
    5. Return all scores (cached + new)
    
    Score Normalization:
    Cross-encoder models output scores in different ranges. The scorer
    provides normalization options:
    - None: Use raw scores from model
    - MinMax: Normalize to [0, 1] range
    - ZScore: Normalize to zero mean, unit variance
    - Sigmoid: Apply sigmoid transformation
    
    Attributes:
        model_loader: Model loader instance
        batch_processor: Batch processor instance
        cache: Optional cache instance
        normalize: Score normalization strategy (default: None)
        _model: Cached model instance
    """
    
    def __init__(
        self,
        model_loader: Any,
        batch_processor: Any,
        cache: Optional[Any] = None,
        normalize: Optional[str] = None
    ):
        """
        Initialize the cross-encoder scorer.
        
        Args:
            model_loader: Model loader instance
            batch_processor: Batch processor instance
            cache: Optional cache instance for score caching
            normalize: Score normalization strategy (None, 'minmax', 'zscore', 'sigmoid')
        """
        self.model_loader = model_loader
        self.batch_processor = batch_processor
        self.cache = cache
        self.normalize = normalize
        self._model = None
        
        logger.info(
            f"CrossEncoderScorer initialized with normalize={normalize}, "
            f"cache={'enabled' if cache else 'disabled'}"
        )
    
    def _get_model(self) -> Any:
        """
        Get the cross-encoder model.
        
        This method lazily loads the model using the model loader and
        caches it for subsequent use.
        
        Returns:
            Loaded cross-encoder model
        """
        if self._model is None:
            self._model = self.model_loader.get_model()
        return self._model
    
    def score_candidates(
        self,
        query: str,
        candidates: List[Any]
    ) -> List[float]:
        """
        Score candidates using cross-encoder.
        
        This method scores a list of candidates using the cross-encoder model.
        It checks the cache for existing scores and only performs inference
        for uncached candidates.
        
        Args:
            query: The search query
            candidates: List of candidates to score
            
        Returns:
            List of rerank scores for each candidate
        """
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Check cache for existing scores
        cached_scores = {}
        uncached_candidates = []
        uncached_indices = []
        
        if self.cache:
            for i, candidate in enumerate(candidates):
                chunk_id = candidate.chunk_id
                cached_score = self.cache.get(query, chunk_id)
                
                if cached_score is not None:
                    cached_scores[i] = cached_score
                else:
                    uncached_candidates.append(candidate)
                    uncached_indices.append(i)
            
            logger.info(
                f"Cache hit: {len(cached_scores)}/{len(candidates)} candidates"
            )
        else:
            uncached_candidates = candidates
            uncached_indices = list(range(len(candidates)))
        
        # Score uncached candidates
        new_scores = []
        if uncached_candidates:
            new_scores = self._score_uncached(query, uncached_candidates)
            
            # Cache new scores
            if self.cache:
                for candidate, score in zip(uncached_candidates, new_scores):
                    self.cache.set(query, candidate.chunk_id, score)
        
        # Combine cached and new scores
        all_scores = [0.0] * len(candidates)
        for i, score in cached_scores.items():
            all_scores[i] = score
        for i, score in zip(uncached_indices, new_scores):
            all_scores[i] = score
        
        # Normalize scores if requested
        if self.normalize:
            all_scores = self._normalize_scores(all_scores)
        
        scoring_time = time.time() - start_time
        logger.info(
            f"Scored {len(candidates)} candidates in {scoring_time:.4f}s "
            f"(cached: {len(cached_scores)}, new: {len(new_scores)})"
        )
        
        return all_scores
    
    def _score_uncached(
        self,
        query: str,
        candidates: List[Any]
    ) -> List[float]:
        """
        Score uncached candidates using cross-encoder.
        
        This method performs cross-encoder inference on candidates that
        were not found in the cache.
        
        Args:
            query: The search query
            candidates: List of uncached candidates to score
            
        Returns:
            List of rerank scores for each candidate
        """
        model = self._get_model()
        scores = self.batch_processor.process_all_batches(
            model,
            query,
            candidates
        )
        
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using the specified strategy.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return scores
        
        if self.normalize == 'minmax':
            return self._normalize_minmax(scores)
        elif self.normalize == 'zscore':
            return self._normalize_zscore(scores)
        elif self.normalize == 'sigmoid':
            return self._normalize_sigmoid(scores)
        else:
            logger.warning(f"Unknown normalization strategy: {self.normalize}")
            return scores
    
    def _normalize_minmax(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Formula: (score - min) / (max - min)
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores in [0, 1] range
        """
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _normalize_zscore(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to zero mean and unit variance.
        
        Formula: (score - mean) / std
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores with zero mean and unit variance
        """
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        
        if std == 0:
            return [0.0] * len(scores)
        
        normalized = (scores_array - mean) / std
        return normalized.tolist()
    
    def _normalize_sigmoid(self, scores: List[float]) -> List[float]:
        """
        Apply sigmoid transformation to scores.
        
        Formula: 1 / (1 + exp(-score))
        
        This maps any real-valued score to the (0, 1) range.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Sigmoid-transformed scores in (0, 1) range
        """
        scores_array = np.array(scores)
        normalized = 1 / (1 + np.exp(-scores_array))
        return normalized.tolist()
    
    def get_score_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Get statistics about the scores.
        
        Args:
            scores: List of scores to analyze
            
        Returns:
            Dictionary with score statistics
        """
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75))
        }
