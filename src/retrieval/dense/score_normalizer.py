"""
Score Normalizer for Dense Retrieval Service.

This module provides score normalization functionality to convert raw similarity
scores to a normalized range (0.0 - 1.0) for consistent scoring across
different vector stores and distance metrics.

Architecture Notes:
- Strategy Pattern for different normalization methods
- Configurable normalization strategies
- Handles cosine similarity normalization
- Supports min-max scaling and other methods

SOLID Principles Applied:
- Single Responsibility: Handles only score normalization
- Open/Closed: Open for new normalization strategies
- Dependency Inversion: Depends on normalization interface
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class NormalizationStrategy(Enum):
    """
    Enumeration of available normalization strategies.
    
    Strategies:
        MIN_MAX: Min-max scaling to [0, 1]
        COSINE: Cosine similarity normalization
        Z_SCORE: Z-score normalization
        NONE: No normalization (use raw scores)
    """
    MIN_MAX = "min_max"
    COSINE = "cosine"
    Z_SCORE = "z_score"
    NONE = "none"


class ScoreNormalizer:
    """
    Normalizes similarity scores to a standard range.
    
    This class provides various normalization strategies to convert raw
    similarity scores to a normalized range (0.0 - 1.0) for consistent
    scoring across different vector stores and distance metrics.
    
    Architecture Pattern: Strategy Pattern
    - Multiple normalization strategies
    - Configurable strategy selection
    - Consistent output range
    - Handles edge cases
    
    Normalization Strategies:
        - MIN_MAX: Scales scores to [0, 1] using min-max scaling
        - COSINE: Normalizes cosine similarity to [0, 1]
        - Z_SCORE: Uses z-score normalization
        - NONE: Returns raw scores unchanged
    """
    
    def __init__(self, strategy: NormalizationStrategy = NormalizationStrategy.COSINE):
        """
        Initialize the score normalizer.
        
        Args:
            strategy: Normalization strategy to use (default: COSINE)
        """
        self.strategy = strategy
        logger.info(f"ScoreNormalizer initialized with strategy: {strategy.value}")
    
    def normalize(self, scores: List[float]) -> List[float]:
        """
        Normalize a list of scores.
        
        Args:
            scores: List of raw scores to normalize
            
        Returns:
            List of normalized scores in [0.0, 1.0]
        """
        if not scores:
            return []
        
        if self.strategy == NormalizationStrategy.NONE:
            logger.debug("No normalization applied")
            return scores.copy()
        
        if self.strategy == NormalizationStrategy.MIN_MAX:
            return self._normalize_min_max(scores)
        
        if self.strategy == NormalizationStrategy.COSINE:
            return self._normalize_cosine(scores)
        
        if self.strategy == NormalizationStrategy.Z_SCORE:
            return self._normalize_z_score(scores)
        
        raise ValueError(f"Unknown normalization strategy: {self.strategy}")
    
    def _normalize_min_max(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using min-max scaling.
        
        Scales scores to [0, 1] using the formula:
        normalized = (score - min) / (max - min)
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            List of normalized scores
        """
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same, return 0.5 for all
            return [0.5] * len(scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        
        logger.debug(f"Min-max normalization applied: range [{min_score:.4f}, {max_score:.4f}]")
        return normalized
    
    def _normalize_cosine(self, scores: List[float]) -> List[float]:
        """
        Normalize cosine similarity scores.
        
        Cosine similarity ranges from [-1, 1]. This method normalizes
        to [0, 1] using the formula: normalized = (score + 1) / 2
        
        Args:
            scores: List of cosine similarity scores
            
        Returns:
            List of normalized scores in [0, 1]
        """
        # Cosine similarity ranges from -1 to 1
        # Normalize to [0, 1] using (score + 1) / 2
        normalized = [(score + 1) / 2 for score in scores]
        
        logger.debug("Cosine normalization applied")
        return normalized
    
    def _normalize_z_score(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using z-score normalization.
        
        Converts scores to z-scores and then scales to [0, 1].
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            List of normalized scores in [0, 1]
        """
        import statistics
        
        if len(scores) < 2:
            # Not enough data for z-score, return 0.5 for all
            return [0.5] * len(scores)
        
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        
        if stdev == 0:
            # All scores are the same, return 0.5 for all
            return [0.5] * len(scores)
        
        # Calculate z-scores
        z_scores = [(score - mean) / stdev for score in scores]
        
        # Scale z-scores to [0, 1]
        # Assuming z-scores are roughly in [-3, 3]
        min_z = min(z_scores)
        max_z = max(z_scores)
        
        if max_z == min_z:
            return [0.5] * len(scores)
        
        normalized = [(z - min_z) / (max_z - min_z) for z in z_scores]
        
        logger.debug("Z-score normalization applied")
        return normalized
    
    def normalize_single(self, score: float, min_score: float, max_score: float) -> float:
        """
        Normalize a single score using min-max scaling.
        
        Args:
            score: Score to normalize
            min_score: Minimum score in the range
            max_score: Maximum score in the range
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_score == min_score:
            return 0.5
        
        normalized = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))
    
    def normalize_dict(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize a dictionary of scores.
        
        Args:
            score_dict: Dictionary of scores to normalize
            
        Returns:
            Dictionary with normalized scores
        """
        scores = list(score_dict.values())
        normalized_scores = self.normalize(scores)
        
        return {key: normalized_scores[i] for i, key in enumerate(score_dict.keys())}
