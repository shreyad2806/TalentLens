"""
RRF Scorer for Hybrid Retrieval.

This module provides the RRF scorer that calculates fusion scores for
documents based on their ranks in dense and sparse retrieval systems.

Architecture Notes:
- Wraps ReciprocalRankFusion for scoring
- Provides scoring interface for fusion service
- Supports configurable k parameter
- Handles missing ranks gracefully

SOLID Principles Applied:
- Single Responsibility: Handles only RRF scoring
- Open/Closed: Open for new scoring methods
- Dependency Inversion: Depends on RRF interface
"""

import logging
from typing import Optional

from .rrf import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class RRFScorer:
    """
    RRF scorer for hybrid retrieval.
    
    This class provides a scoring interface for the Reciprocal Rank Fusion
    algorithm. It wraps the RRF implementation and provides a clean API for
    calculating fusion scores based on dense and sparse ranks.
    
    Scoring Process:
        1. Initialize with RRF constant k
        2. Calculate RRF score based on dense and sparse ranks
        3. Handle missing ranks (documents in only one system)
        4. Return non-negative fusion score
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize the RRF scorer.
        
        Args:
            k: Constant for RRF calculation (default: 60)
        """
        self.rrf = ReciprocalRankFusion(k=k)
        self.k = k
        
        logger.info(f"RRFScorer initialized with k={k}")
    
    def calculate_score(
        self,
        dense_rank: Optional[int],
        sparse_rank: Optional[int]
    ) -> float:
        """
        Calculate RRF score for a document.
        
        This method calculates the RRF fusion score for a document based
        on its ranks in the dense and sparse retrieval systems.
        
        Args:
            dense_rank: Rank from dense retrieval (None if not in results)
            sparse_rank: Rank from sparse retrieval (None if not in results)
            
        Returns:
            RRF score for the document
        """
        score = self.rrf.calculate_rrf_score(dense_rank, sparse_rank)
        
        # Ensure score is non-negative
        if score < 0:
            logger.warning(f"Negative RRF score detected: {score}, setting to 0")
            score = 0.0
        
        return score
    
    def update_k(self, k: int) -> None:
        """
        Update the RRF constant k.
        
        Args:
            k: New constant for RRF calculation
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        self.k = k
        self.rrf = ReciprocalRankFusion(k=k)
        
        logger.info(f"RRFScorer updated k to {k}")
