"""
Hybrid Retrieval Package.

This package provides production-grade hybrid retrieval using Reciprocal
Rank Fusion (RRF) to combine dense and sparse retrieval results.

Components:
- schema: Pydantic schemas for hybrid retrieval results
- rrf: Reciprocal Rank Fusion implementation
- scorer: RRF scorer for fusion calculations
- fusion_service: Service for fusing dense and sparse results
- validator: Validation for hybrid retrieval results
- cache: Caching for hybrid retrieval results
- hybrid_retrieval_service: Main facade for hybrid retrieval

Usage:
    from src.retrieval.hybrid import HybridRetrievalService
    
    # Initialize with dense and sparse services
    hybrid_service = HybridRetrievalService(
        dense_retrieval_service=dense_service,
        sparse_retrieval_service=sparse_service,
        k=60,
        cache_enabled=True
    )
    
    # Search using hybrid retrieval
    results = hybrid_service.search(
        query="Python Backend Developer",
        top_k=10,
        filters={"section": "experience"}
    )
    
    # Results are HybridSearchResult objects with:
    # - candidate_name, resume_id, chunk_id, section
    # - dense_rank, sparse_rank, rrf_score
    # - metadata, matched_chunks, rank

Architecture:
- Facade pattern for clean API
- RRF for rank-based fusion
- Caching for performance
- Validation for data integrity
- Comprehensive logging

SOLID Principles:
- Single Responsibility: Each component has one clear purpose
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Components can be substituted
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depends on abstractions, not concretions
"""

from .schema import (
    HybridSearchResult,
    MatchedChunk,
    RetrievalSource,
    FusionMetrics,
    FusionStrategy
)
from .rrf import ReciprocalRankFusion
from .scorer import RRFScorer
from .fusion_service import (
    FusionService,
    FusionStrategyBase,
    RRFFusionStrategy,
    WeightedFusionStrategy,
    ScoreAveragingFusionStrategy
)
from .validator import HybridRetrievalValidator, ValidationError
from .cache import HybridResultCache
from .hybrid_retrieval_service import HybridRetrievalService

__all__ = [
    # Schema
    "HybridSearchResult",
    "MatchedChunk",
    "RetrievalSource",
    "FusionMetrics",
    "FusionStrategy",
    # RRF
    "ReciprocalRankFusion",
    "RRFScorer",
    # Fusion
    "FusionService",
    "FusionStrategyBase",
    "RRFFusionStrategy",
    "WeightedFusionStrategy",
    "ScoreAveragingFusionStrategy",
    # Validation
    "HybridRetrievalValidator",
    "ValidationError",
    # Cache
    "HybridResultCache",
    # Service
    "HybridRetrievalService",
]
