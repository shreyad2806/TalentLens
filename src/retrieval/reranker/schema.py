"""
Schema module for Reranker.

This module defines the data models used by the reranker service, including
the RerankedResult schema that represents a candidate after reranking with
a cross-encoder model.

Architecture Notes:
- Pure data models following Single Responsibility Principle
- Encapsulates reranked result structure independent of implementation
- Compatible with hybrid retrieval results
- Preserves original retrieval scores for comparison

SOLID Principles Applied:
- Single Responsibility: Data models only, no business logic
- Open/Closed: Can be extended with new fields without modification
- Liskov Substitution: Compatible with base result schemas
- Interface Segregation: Focused schema interfaces
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class RerankEvidence(str, Enum):
    """
    Enumeration for evidence types in reranked results.
    
    This tracks the source of evidence for the reranking decision.
    """
    CROSS_ENCODER = "cross_encoder"
    HYBRID_SCORE = "hybrid_score"
    COMBINED = "combined"


class RerankedResult(BaseModel):
    """
    Schema for a reranked search result.
    
    This class represents a candidate after reranking with a cross-encoder model.
    It preserves the original retrieval scores and adds the rerank score from
    the cross-encoder, allowing for comparison and weighted fusion strategies.
    
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
    
    Architecture Pattern: Data Transfer Object (DTO)
    - Pure data model with validation
    - No business logic
    - Serializable for storage and transmission
    
    Attributes:
        query: The search query used for reranking
        candidate_name: Name of the candidate
        resume_id: ID of the resume
        chunk_id: ID of the chunk
        section: Section of the resume (e.g., skills, experience)
        original_rank: Original rank from hybrid retrieval
        original_score: Original score from hybrid retrieval (RRF score)
        rerank_score: Score from cross-encoder reranking
        final_rank: Final rank after reranking
        metadata: Additional metadata about the candidate
        matched_text: Text that matched in the retrieval
        evidence: Source of evidence for reranking decision
        reranked_at: Timestamp when reranking was performed
    """
    
    query: str = Field(..., description="Search query used for reranking")
    candidate_name: str = Field(..., description="Name of the candidate")
    resume_id: str = Field(..., description="ID of the resume")
    chunk_id: str = Field(..., description="ID of the chunk")
    section: str = Field(..., description="Section of the resume")
    original_rank: Optional[int] = Field(None, description="Original rank from hybrid retrieval")
    original_score: Optional[float] = Field(None, description="Original score from hybrid retrieval")
    rerank_score: float = Field(..., description="Score from cross-encoder reranking")
    final_rank: int = Field(..., description="Final rank after reranking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    matched_text: str = Field(..., description="Text that matched in the retrieval")
    evidence: RerankEvidence = Field(
        default=RerankEvidence.CROSS_ENCODER,
        description="Source of evidence for reranking decision"
    )
    reranked_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when reranking was performed"
    )
    
    @field_validator('rerank_score')
    @classmethod
    def validate_rerank_score(cls, v: float) -> float:
        """
        Validate that rerank score is a valid number.
        
        Cross-encoders typically output scores in a specific range depending
        on the model. We ensure the score is not NaN or infinite.
        
        Args:
            v: Rerank score to validate
            
        Returns:
            Validated rerank score
            
        Raises:
            ValueError: If score is NaN or infinite
        """
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Rerank score must be a valid number, got {v}")
        return v
    
    @field_validator('final_rank')
    @classmethod
    def validate_final_rank(cls, v: int) -> int:
        """
        Validate that final rank is non-negative.
        
        Args:
            v: Final rank to validate
            
        Returns:
            Validated final rank
            
        Raises:
            ValueError: If rank is negative
        """
        if v < 0:
            raise ValueError(f"Final rank must be non-negative, got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RerankedResult to a dictionary.
        
        Returns:
            Dictionary representation of the reranked result
        """
        return self.model_dump()
    
    def get_score_delta(self) -> Optional[float]:
        """
        Calculate the difference between rerank score and original score.
        
        This is useful for understanding how much the cross-encoder changed
        the ranking compared to the original hybrid retrieval.
        
        Returns:
            Difference between rerank score and original score, or None if
            original score is not available
        """
        if self.original_score is not None:
            return self.rerank_score - self.original_score
        return None
    
    def get_rank_delta(self) -> Optional[int]:
        """
        Calculate the change in rank after reranking.
        
        Positive values indicate the candidate moved up in ranking,
        negative values indicate it moved down.
        
        Returns:
            Change in rank, or None if original rank is not available
        """
        if self.original_rank is not None:
            return self.original_rank - self.final_rank
        return None


class RerankMetrics(BaseModel):
    """
    Metrics for reranking performance.
    
    This class tracks performance metrics for the reranking process,
    including latency, cache statistics, and scoring statistics.
    
    Architecture Pattern: Data Transfer Object (DTO)
    - Pure data model with validation
    - No business logic
    - Serializable for logging and analysis
    
    Attributes:
        total_candidates: Total number of candidates reranked
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        cache_hit_rate: Cache hit rate (0.0 to 1.0)
        avg_rerank_score: Average rerank score across all candidates
        min_rerank_score: Minimum rerank score
        max_rerank_score: Maximum rerank score
        avg_score_delta: Average change in score from original to reranked
        total_latency: Total time for reranking in seconds
        avg_latency: Average latency per candidate in seconds
        batch_size: Batch size used for inference
        num_batches: Number of batches processed
    """
    
    total_candidates: int = Field(..., description="Total number of candidates reranked")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate (0.0 to 1.0)")
    avg_rerank_score: float = Field(default=0.0, description="Average rerank score")
    min_rerank_score: float = Field(default=0.0, description="Minimum rerank score")
    max_rerank_score: float = Field(default=0.0, description="Maximum rerank score")
    avg_score_delta: Optional[float] = Field(None, description="Average score delta")
    total_latency: float = Field(..., description="Total reranking latency in seconds")
    avg_latency: float = Field(..., description="Average latency per candidate")
    batch_size: int = Field(default=32, description="Batch size used for inference")
    num_batches: int = Field(default=0, description="Number of batches processed")
    
    @field_validator('cache_hit_rate')
    @classmethod
    def validate_cache_hit_rate(cls, v: float) -> float:
        """
        Validate that cache hit rate is between 0 and 1.
        
        Args:
            v: Cache hit rate to validate
            
        Returns:
            Validated cache hit rate
            
        Raises:
            ValueError: If cache hit rate is outside [0, 1]
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Cache hit rate must be between 0 and 1, got {v}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RerankMetrics to a dictionary.
        
        Returns:
            Dictionary representation of the metrics
        """
        return self.model_dump()
