"""
Reciprocal Rank Fusion (RRF) Implementation.

This module implements the Reciprocal Rank Fusion algorithm for combining
results from multiple retrieval systems. RRF is a simple yet effective
method for rank aggregation that does not require score normalization.

RRF Algorithm:
    RRF combines ranked lists from multiple retrieval systems by computing
    a fusion score for each document based on its rank in each list.
    
    Formula:
        rrf_score = 1 / (k + rank_1) + 1 / (k + rank_2) + ...
    
    Where:
        - k is a constant (default: 60)
        - rank_i is the rank of the document in retrieval system i
    
    Documents that appear in only one retrieval system are handled by
    assigning a rank of infinity (or a large value) for missing systems,
    effectively giving them a score of 0 from that system.

Architecture Notes:
- Configurable constant k for tuning fusion behavior
- Handles documents appearing in only one retrieval system
- No score normalization required (rank-based)
- Robust to score scale differences between systems

SOLID Principles Applied:
- Single Responsibility: Handles only RRF calculation
- Open/Closed: Open for new fusion algorithms
- Dependency Inversion: Depends on abstract interfaces
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) implementation.
    
    This class implements the RRF algorithm for combining ranked results
    from multiple retrieval systems. RRF is a rank-based fusion method
    that does not require score normalization, making it robust to
    differences in score scales between retrieval systems.
    
    RRF Formula:
        rrf_score = 1 / (k + rank_1) + 1 / (k + rank_2) + ...
    
    Where:
        - k is a constant (default: 60)
        - rank_i is the rank of the document in retrieval system i
    
    The constant k controls the influence of rank on the fusion score:
        - Higher k: More balanced fusion, less sensitive to rank differences
        - Lower k: More sensitive to rank differences, top ranks dominate
    
    Default k = 60 is a well-established value in information retrieval research.
    
    Handling Missing Documents:
        Documents that appear in only one retrieval system are handled by
        assigning a rank of infinity (or a large value) for missing systems,
        effectively giving them a score of 0 from that system.
    
    Example:
        Document A: rank 1 in dense, rank 5 in sparse
        Document B: rank 3 in dense, not in sparse
        Document C: not in dense, rank 2 in sparse
        
        With k = 60:
        RRF(A) = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318
        RRF(B) = 1/(60+3) + 0 = 0.0159
        RRF(C) = 0 + 1/(60+2) = 0.0161
        
        Final ranking: A > C > B
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize the RRF fusion algorithm.
        
        Args:
            k: Constant for RRF calculation (default: 60)
                Higher values make fusion more balanced
                Lower values make fusion more sensitive to rank differences
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        self.k = k
        logger.info(f"ReciprocalRankFusion initialized with k={k}")
    
    def calculate_rrf_score(
        self,
        dense_rank: Optional[int],
        sparse_rank: Optional[int]
    ) -> float:
        """
        Calculate RRF score for a document based on its ranks.
        
        This method calculates the RRF score for a document based on its
        ranks in the dense and sparse retrieval systems. If a document
        does not appear in a retrieval system, its rank is treated as
        infinity (effectively giving it a score of 0 from that system).
        
        RRF Formula:
            rrf_score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
        
        Args:
            dense_rank: Rank from dense retrieval (None if not in results)
            sparse_rank: Rank from sparse retrieval (None if not in results)
            
        Returns:
            RRF score for the document
        """
        score = 0.0
        
        # Add contribution from dense retrieval
        if dense_rank is not None and dense_rank >= 0:
            score += 1.0 / (self.k + dense_rank)
        
        # Add contribution from sparse retrieval
        if sparse_rank is not None and sparse_rank >= 0:
            score += 1.0 / (self.k + sparse_rank)
        
        return score
    
    def fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_rank_key: str = "rank",
        sparse_rank_key: str = "rank",
        id_key: str = "chunk_id"
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from dense and sparse retrieval using RRF.
        
        This method combines results from dense and sparse retrieval systems
        using RRF. It handles documents that appear in only one system and
        calculates fusion scores for all documents.
        
        Fusion Process:
        1. Create mapping from document ID to rank in each system
        2. Calculate RRF score for each document
        3. Sort documents by RRF score (descending)
        4. Return fused results with fusion scores
        
        Args:
            dense_results: List of dense retrieval results
            sparse_results: List of sparse retrieval results
            dense_rank_key: Key for rank in dense results (default: "rank")
            sparse_rank_key: Key for rank in sparse results (default: "rank")
            id_key: Key for document ID (default: "chunk_id")
            
        Returns:
            List of fused results with RRF scores
        """
        # Create rank mappings
        dense_ranks = {result[id_key]: result[dense_rank_key] for result in dense_results}
        sparse_ranks = {result[id_key]: result[sparse_rank_key] for result in sparse_results}
        
        # Get all unique document IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        fused_results = []
        for doc_id in all_ids:
            dense_rank = dense_ranks.get(doc_id)
            sparse_rank = sparse_ranks.get(doc_id)
            
            rrf_score = self.calculate_rrf_score(dense_rank, sparse_rank)
            
            fused_results.append({
                id_key: doc_id,
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "rrf_score": rrf_score
            })
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Assign final ranks
        for idx, result in enumerate(fused_results):
            result["rank"] = idx
        
        logger.info(
            f"Fused {len(dense_results)} dense and {len(sparse_results)} sparse results "
            f"into {len(fused_results)} fused results using RRF (k={self.k})"
        )
        
        return fused_results
    
    def get_fusion_statistics(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        fused_results: List[Dict[str, Any]],
        id_key: str = "chunk_id"
    ) -> Dict[str, Any]:
        """
        Calculate fusion statistics.
        
        This method calculates statistics about the fusion process including
        overlap, unique documents, and score distributions.
        
        Args:
            dense_results: List of dense retrieval results
            sparse_results: List of sparse retrieval results
            fused_results: List of fused results
            id_key: Key for document ID (default: "chunk_id")
            
        Returns:
            Dictionary with fusion statistics
        """
        dense_ids = set(result[id_key] for result in dense_results)
        sparse_ids = set(result[id_key] for result in sparse_results)
        fused_ids = set(result[id_key] for result in fused_results)
        
        overlap = dense_ids & sparse_ids
        dense_only = dense_ids - sparse_ids
        sparse_only = sparse_ids - dense_ids
        
        statistics = {
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_results),
            "overlap_count": len(overlap),
            "dense_only_count": len(dense_only),
            "sparse_only_count": len(sparse_only),
            "overlap_ratio": len(overlap) / len(fused_ids) if fused_ids else 0.0,
            "rrf_scores": [result["rrf_score"] for result in fused_results]
        }
        
        logger.info(
            f"Fusion statistics: overlap={len(overlap)}, "
            f"dense_only={len(dense_only)}, sparse_only={len(sparse_only)}"
        )
        
        return statistics
