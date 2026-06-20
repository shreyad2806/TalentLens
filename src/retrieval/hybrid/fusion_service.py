"""
Fusion Service for Hybrid Retrieval.

This module provides the FusionService that combines results from dense
and sparse retrieval systems using configurable fusion strategies.

Architecture Notes:
- Strategy pattern for extensible fusion algorithms
- Merges duplicate candidates from both systems
- Preserves matched chunks, metadata, and evidence
- Tracks retrieval source for each result
- Supports RRF, weighted fusion, and score averaging

SOLID Principles Applied:
- Single Responsibility: Handles only result fusion
- Open/Closed: Open for new fusion algorithms via strategy pattern
- Dependency Inversion: Depends on abstract fusion strategy interface
"""

import logging
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .schema import HybridSearchResult, MatchedChunk, RetrievalSource, FusionMetrics, FusionStrategy
from .rrf import ReciprocalRankFusion
from .scorer import RRFScorer

logger = logging.getLogger(__name__)


class FusionStrategyBase(ABC):
    """
    Abstract base class for fusion strategies.
    
    This class defines the interface for fusion strategies. Each fusion
    strategy must implement the calculate_score method to compute a fusion
    score based on dense and sparse retrieval information.
    
    Strategy Pattern:
        This allows different fusion algorithms to be plugged in without
        modifying the FusionService. New strategies can be added by extending
        this base class and implementing the calculate_score method.
    """
    
    @abstractmethod
    def calculate_score(
        self,
        dense_rank: Optional[int],
        dense_score: Optional[float],
        sparse_rank: Optional[int],
        sparse_score: Optional[float]
    ) -> float:
        """
        Calculate fusion score for a candidate.
        
        Args:
            dense_rank: Rank from dense retrieval (None if not in results)
            dense_score: Score from dense retrieval (None if not in results)
            sparse_rank: Rank from sparse retrieval (None if not in results)
            sparse_score: Score from sparse retrieval (None if not in results)
            
        Returns:
            Fusion score for the candidate
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of the fusion strategy.
        
        Returns:
            Name of the fusion strategy
        """
        pass


class RRFFusionStrategy(FusionStrategyBase):
    """
    Reciprocal Rank Fusion (RRF) strategy.
    
    This strategy uses the RRF algorithm to combine ranks from dense and
    sparse retrieval systems. RRF is a rank-based fusion method that does
    not require score normalization.
    
    RRF Formula:
        rrf_score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
    
    Where:
        - k is a constant (default: 60)
        - dense_rank is the rank from dense retrieval
        - sparse_rank is the rank from sparse retrieval
    
    Handling Missing Documents:
        Documents that appear in only one retrieval system are handled by
        assigning a rank of infinity for the missing system, effectively
        giving them a score of 0 from that system.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize the RRF fusion strategy.
        
        Args:
            k: Constant for RRF calculation (default: 60)
        """
        self.rrf_scorer = RRFScorer(k=k)
        self.k = k
        
        logger.info(f"RRFFusionStrategy initialized with k={k}")
    
    def calculate_score(
        self,
        dense_rank: Optional[int],
        dense_score: Optional[float],
        sparse_rank: Optional[int],
        sparse_score: Optional[float]
    ) -> float:
        """
        Calculate RRF score for a candidate.
        
        Args:
            dense_rank: Rank from dense retrieval (None if not in results)
            dense_score: Score from dense retrieval (not used in RRF)
            sparse_rank: Rank from sparse retrieval (None if not in results)
            sparse_score: Score from sparse retrieval (not used in RRF)
            
        Returns:
            RRF score for the candidate
        """
        return self.rrf_scorer.calculate_score(dense_rank, sparse_rank)
    
    def get_strategy_name(self) -> str:
        """Get the name of the fusion strategy."""
        return FusionStrategy.RRF.value
    
    def update_k(self, k: int) -> None:
        """
        Update the RRF constant k.
        
        Args:
            k: New constant for RRF calculation
        """
        self.rrf_scorer.update_k(k)
        self.k = k


class WeightedFusionStrategy(FusionStrategyBase):
    """
    Weighted fusion strategy (future implementation).
    
    This strategy uses weighted combination of dense and sparse scores.
    The weights can be configured to give more importance to one
    retrieval system over the other.
    
    Weighted Fusion Formula:
        weighted_score = w_dense * dense_score + w_sparse * sparse_score
    
    Where:
        - w_dense is the weight for dense retrieval (default: 0.5)
        - w_sparse is the weight for sparse retrieval (default: 0.5)
        - dense_score is the score from dense retrieval
        - sparse_score is the score from sparse retrieval
    
    Note: This strategy requires score normalization to handle different
    score scales between retrieval systems.
    """
    
    def __init__(self, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        """
        Initialize the weighted fusion strategy.
        
        Args:
            dense_weight: Weight for dense retrieval (default: 0.5)
            sparse_weight: Weight for sparse retrieval (default: 0.5)
        """
        if dense_weight < 0 or dense_weight > 1:
            raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")
        
        if sparse_weight < 0 or sparse_weight > 1:
            raise ValueError(f"sparse_weight must be between 0 and 1, got {sparse_weight}")
        
        if dense_weight + sparse_weight != 1.0:
            raise ValueError(f"Weights must sum to 1.0, got {dense_weight + sparse_weight}")
        
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(
            f"WeightedFusionStrategy initialized with "
            f"dense_weight={dense_weight}, sparse_weight={sparse_weight}"
        )
    
    def calculate_score(
        self,
        dense_rank: Optional[int],
        dense_score: Optional[float],
        sparse_rank: Optional[int],
        sparse_score: Optional[float]
    ) -> float:
        """
        Calculate weighted fusion score for a candidate.
        
        Args:
            dense_rank: Rank from dense retrieval (not used in weighted fusion)
            dense_score: Score from dense retrieval
            sparse_rank: Rank from sparse retrieval (not used in weighted fusion)
            sparse_score: Score from sparse retrieval
            
        Returns:
            Weighted fusion score for the candidate
        """
        score = 0.0
        
        if dense_score is not None:
            score += self.dense_weight * dense_score
        
        if sparse_score is not None:
            score += self.sparse_weight * sparse_score
        
        return score
    
    def get_strategy_name(self) -> str:
        """Get the name of the fusion strategy."""
        return FusionStrategy.WEIGHTED.value


class ScoreAveragingFusionStrategy(FusionStrategyBase):
    """
    Score averaging fusion strategy (future implementation).
    
    This strategy averages the scores from dense and sparse retrieval
    systems. This is a simple fusion method that gives equal weight
    to both systems.
    
    Score Averaging Formula:
        avg_score = (dense_score + sparse_score) / 2
    
    Where:
        - dense_score is the score from dense retrieval
        - sparse_score is the score from sparse retrieval
    
    Note: This strategy requires score normalization to handle different
    score scales between retrieval systems.
    """
    
    def __init__(self):
        """Initialize the score averaging fusion strategy."""
        logger.info("ScoreAveragingFusionStrategy initialized")
    
    def calculate_score(
        self,
        dense_rank: Optional[int],
        dense_score: Optional[float],
        sparse_rank: Optional[int],
        sparse_score: Optional[float]
    ) -> float:
        """
        Calculate score averaging fusion score for a candidate.
        
        Args:
            dense_rank: Rank from dense retrieval (not used in score averaging)
            dense_score: Score from dense retrieval
            sparse_rank: Rank from sparse retrieval (not used in score averaging)
            sparse_score: Score from sparse retrieval
            
        Returns:
        Score averaging fusion score for the candidate
        """
        scores = []
        
        if dense_score is not None:
            scores.append(dense_score)
        
        if sparse_score is not None:
            scores.append(sparse_score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def get_strategy_name(self) -> str:
        """Get the name of the fusion strategy."""
        return FusionStrategy.SCORE_AVERAGING.value


class FusionService:
    """
    Fusion service for combining dense and sparse retrieval results.
    
    This class provides the functionality to merge results from dense and
    sparse retrieval systems using configurable fusion strategies. It handles
    duplicate candidates, preserves matched chunks and metadata, and tracks
    retrieval sources.
    
    Fusion Process:
        1. Receive dense and sparse results
        2. Merge duplicate candidates by chunk_id
        3. Calculate fusion scores using selected strategy
        4. Sort by fusion score (descending)
        5. Assign final ranks
        6. Return fused HybridSearchResult objects
    
    Handling Different Candidate Scenarios:
        - Candidate exists in both: Merge information from both systems
        - Candidate exists only in dense: Use dense information only
        - Candidate exists only in sparse: Use sparse information only
    
    Preservation:
        - Dense evidence (matched chunks, scores, metadata)
        - Sparse evidence (matched chunks, scores, metadata)
        - Matched chunks from both systems
        - Metadata from both systems
        - Retrieval source tracking
    
    Strategy Pattern:
        The FusionService uses the Strategy pattern to support different
        fusion algorithms. This allows new fusion strategies to be added
        without modifying the FusionService class.
    
    Supported Strategies:
        - RRF (Reciprocal Rank Fusion): Rank-based fusion (default)
        - Weighted Fusion: Weighted combination of scores
        - Score Averaging: Simple average of scores
    """
    
    def __init__(
        self,
        strategy: Optional[FusionStrategyBase] = None,
        strategy_name: str = FusionStrategy.RRF.value
    ):
        """
        Initialize the fusion service.
        
        Args:
            strategy: Custom fusion strategy (optional)
            strategy_name: Name of fusion strategy to use (default: "rrf")
        """
        if strategy:
            self.strategy = strategy
        else:
            # Initialize default strategy based on name
            if strategy_name == FusionStrategy.RRF.value:
                self.strategy = RRFFusionStrategy(k=60)
            elif strategy_name == FusionStrategy.WEIGHTED.value:
                self.strategy = WeightedFusionStrategy(dense_weight=0.5, sparse_weight=0.5)
            elif strategy_name == FusionStrategy.SCORE_AVERAGING.value:
                self.strategy = ScoreAveragingFusionStrategy()
            else:
                raise ValueError(f"Unknown fusion strategy: {strategy_name}")
        
        self.strategy_name = self.strategy.get_strategy_name()
        
        logger.info(
            f"FusionService initialized with strategy={self.strategy_name}"
        )
    
    def fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        query: str
    ) -> tuple[List[HybridSearchResult], FusionMetrics]:
        """
        Fuse dense and sparse retrieval results using configured strategy.
        
        This method combines results from dense and sparse retrieval systems
        using the configured fusion strategy. It handles three scenarios:
        1. Candidate exists in both systems: Merge information
        2. Candidate exists only in dense: Use dense information
        3. Candidate exists only in sparse: Use sparse information
        
        The method preserves matched chunks, metadata, and evidence from
        both systems, and tracks retrieval sources for each result.
        
        Fusion Process:
            1. Create candidate mapping by chunk_id
            2. Process dense results and add to mapping
            3. Process sparse results and add to mapping
            4. Calculate fusion scores using selected strategy
            5. Sort by fusion score (descending)
            6. Assign final ranks
            7. Return fused HybridSearchResult objects
        
        Args:
            dense_results: List of dense retrieval results
            sparse_results: List of sparse retrieval results
            query: The search query
            
        Returns:
            Tuple of (fused results, fusion metrics)
        """
        start_time = time.time()
        
        # Create candidate mapping by chunk_id
        candidates = {}
        
        # Process dense results
        for idx, result in enumerate(dense_results):
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in candidates:
                # Candidate not seen before, create new entry
                candidates[chunk_id] = {
                    "query": query,
                    "candidate_name": result.get("candidate_name", "Unknown"),
                    "resume_id": result.get("resume_id", ""),
                    "chunk_id": chunk_id,
                    "section": result.get("section", ""),
                    "dense_rank": idx,
                    "dense_score": result.get("score", 0.0),
                    "sparse_rank": None,
                    "sparse_score": None,
                    "metadata": result.get("metadata", {}),
                    "matched_chunks": [],
                    "retrieval_sources": set()
                }
            else:
                # Candidate already exists (from sparse), update dense info
                candidates[chunk_id]["dense_rank"] = idx
                candidates[chunk_id]["dense_score"] = result.get("score", 0.0)
            
            # Add matched chunk from dense retrieval (preserve dense evidence)
            matched_chunk = MatchedChunk(
                chunk_id=chunk_id,
                section=result.get("section", ""),
                matched_text=result.get("matched_text", ""),
                score=result.get("score", 0.0),
                retrieval_source=RetrievalSource.DENSE
            )
            candidates[chunk_id]["matched_chunks"].append(matched_chunk)
            candidates[chunk_id]["retrieval_sources"].add(RetrievalSource.DENSE)
            
            # Merge metadata (preserve dense metadata)
            if "metadata" in result:
                candidates[chunk_id]["metadata"].update(result["metadata"])
        
        # Process sparse results
        for idx, result in enumerate(sparse_results):
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in candidates:
                # Candidate not seen before, create new entry
                candidates[chunk_id] = {
                    "query": query,
                    "candidate_name": result.get("candidate_name", "Unknown"),
                    "resume_id": result.get("resume_id", ""),
                    "chunk_id": chunk_id,
                    "section": result.get("section", ""),
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": idx,
                    "sparse_score": result.get("bm25_score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "matched_chunks": [],
                    "retrieval_sources": set()
                }
            else:
                # Candidate already exists (from dense), update sparse info
                candidates[chunk_id]["sparse_rank"] = idx
                candidates[chunk_id]["sparse_score"] = result.get("bm25_score", 0.0)
            
            # Add matched chunk from sparse retrieval (preserve sparse evidence)
            matched_chunk = MatchedChunk(
                chunk_id=chunk_id,
                section=result.get("section", ""),
                matched_text=result.get("matched_text", ""),
                score=result.get("bm25_score", 0.0),
                retrieval_source=RetrievalSource.SPARSE
            )
            candidates[chunk_id]["matched_chunks"].append(matched_chunk)
            candidates[chunk_id]["retrieval_sources"].add(RetrievalSource.SPARSE)
            
            # Merge metadata (preserve sparse metadata)
            if "metadata" in result:
                candidates[chunk_id]["metadata"].update(result["metadata"])
        
        # Calculate fusion scores using selected strategy
        fused_results = []
        for chunk_id, candidate in candidates.items():
            fusion_score = self.strategy.calculate_score(
                dense_rank=candidate["dense_rank"],
                dense_score=candidate["dense_score"],
                sparse_rank=candidate["sparse_rank"],
                sparse_score=candidate["sparse_score"]
            )
            
            # Create hybrid result with fusion score
            hybrid_result = HybridSearchResult(
                query=query,
                candidate_name=candidate["candidate_name"],
                resume_id=candidate["resume_id"],
                chunk_id=chunk_id,
                section=candidate["section"],
                dense_rank=candidate["dense_rank"],
                sparse_rank=candidate["sparse_rank"],
                rrf_score=fusion_score,  # Using rrf_score field for fusion score
                metadata=candidate["metadata"],
                matched_chunks=candidate["matched_chunks"],
                rank=0  # Will be assigned after sorting
            )
            
            fused_results.append(hybrid_result)
        
        # Sort by fusion score (descending)
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Assign final ranks
        for idx, result in enumerate(fused_results):
            result.rank = idx
        
        fusion_latency = time.time() - start_time
        
        # Calculate fusion metrics
        dense_only_count = sum(
            1 for r in fused_results 
            if r.dense_rank is not None and r.sparse_rank is None
        )
        sparse_only_count = sum(
            1 for r in fused_results 
            if r.sparse_rank is not None and r.dense_rank is None
        )
        overlap_count = sum(
            1 for r in fused_results 
            if r.dense_rank is not None and r.sparse_rank is not None
        )
        
        metrics = FusionMetrics(
            dense_latency=0.0,  # Will be set by caller
            sparse_latency=0.0,  # Will be set by caller
            fusion_latency=fusion_latency,
            total_latency=0.0,  # Will be set by caller
            dense_candidate_count=len(dense_results),
            sparse_candidate_count=len(sparse_results),
            fused_candidate_count=len(fused_results),
            dense_only_count=dense_only_count,
            sparse_only_count=sparse_only_count,
            overlap_count=overlap_count
        )
        
        logger.info(
            f"Fused {len(dense_results)} dense and {len(sparse_results)} sparse results "
            f"into {len(fused_results)} hybrid results using {self.strategy_name} "
            f"in {fusion_latency:.3f}s"
        )
        
        logger.info(
            f"Fusion statistics: "
            f"overlap={overlap_count}, "
            f"dense_only={dense_only_count}, "
            f"sparse_only={sparse_only_count}"
        )
        
        return fused_results, metrics
    
    def update_strategy(self, strategy: FusionStrategyBase) -> None:
        """
        Update the fusion strategy.
        
        Args:
            strategy: New fusion strategy to use
        """
        self.strategy = strategy
        self.strategy_name = strategy.get_strategy_name()
        
        logger.info(f"FusionService updated strategy to {self.strategy_name}")
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the current fusion strategy.
        
        Returns:
            Name of the current fusion strategy
        """
        return self.strategy_name
