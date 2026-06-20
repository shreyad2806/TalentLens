"""
Hybrid Retrieval Schema.

This module defines the Pydantic schemas for hybrid retrieval results.
It provides the HybridSearchResult schema that combines dense and sparse
retrieval results using Reciprocal Rank Fusion (RRF).

Architecture Notes:
- HybridSearchResult combines dense and sparse retrieval information
- Preserves matched chunks, metadata, and evidence from both systems
- Tracks retrieval source for each result
- Maintains rank information from both retrieval systems

SOLID Principles Applied:
- Single Responsibility: Handles only hybrid result schema definition
- Open/Closed: Open for new hybrid result types
- Dependency Inversion: Depends on abstract interfaces
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class RetrievalSource(str, Enum):
    """Enumeration of retrieval sources."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class FusionStrategy(str, Enum):
    """Enumeration of fusion strategies."""
    RRF = "rrf"
    WEIGHTED = "weighted"
    SCORE_AVERAGING = "score_averaging"


class MatchedChunk(BaseModel):
    """
    Matched chunk information.
    
    This class represents a chunk that matched the query, including
    its metadata and evidence from the retrieval system.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    section: str = Field(..., description="Section of the resume")
    matched_text: str = Field(..., description="Text that matched the query")
    score: float = Field(..., description="Score from the retrieval system")
    retrieval_source: RetrievalSource = Field(
        ...,
        description="Source of the retrieval (dense, sparse, or hybrid)"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class HybridSearchResult(BaseModel):
    """
    Hybrid search result combining dense and sparse retrieval.
    
    This class represents a hybrid search result that combines information
    from both dense and sparse retrieval systems using Reciprocal Rank Fusion (RRF).
    It preserves matched chunks, metadata, and evidence from both systems.
    
    Fields:
        query: The search query
        candidate_name: Name of the candidate
        resume_id: Unique identifier for the resume
        chunk_id: Unique identifier for the chunk
        section: Section of the resume
        dense_rank: Rank from dense retrieval (None if not in dense results)
        sparse_rank: Rank from sparse retrieval (None if not in sparse results)
        rrf_score: Reciprocal Rank Fusion score
        metadata: Additional metadata about the result
        matched_chunks: List of matched chunks from both retrieval systems
        rank: Final rank after fusion and sorting
    """
    query: str = Field(..., description="The search query")
    candidate_name: str = Field(..., description="Name of the candidate")
    resume_id: str = Field(..., description="Unique identifier for the resume")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    section: str = Field(..., description="Section of the resume")
    dense_rank: Optional[int] = Field(
        default=None,
        description="Rank from dense retrieval (None if not in dense results)"
    )
    sparse_rank: Optional[int] = Field(
        default=None,
        description="Rank from sparse retrieval (None if not in sparse results)"
    )
    rrf_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Reciprocal Rank Fusion score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the result"
    )
    matched_chunks: List[MatchedChunk] = Field(
        default_factory=list,
        description="List of matched chunks from both retrieval systems"
    )
    rank: int = Field(
        default=0,
        ge=0,
        description="Final rank after fusion and sorting"
    )
    
    @field_validator('rrf_score')
    @classmethod
    def validate_rrf_score(cls, v: float) -> float:
        """Validate that RRF score is non-negative."""
        if v < 0:
            raise ValueError('RRF score must be non-negative')
        return v
    
    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        """Validate that rank is non-negative."""
        if v < 0:
            raise ValueError('Rank must be non-negative')
        return v
    
    @field_validator('dense_rank')
    @classmethod
    def validate_dense_rank(cls, v: Optional[int]) -> Optional[int]:
        """Validate that dense rank is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError('Dense rank must be non-negative')
        return v
    
    @field_validator('sparse_rank')
    @classmethod
    def validate_sparse_rank(cls, v: Optional[int]) -> Optional[int]:
        """Validate that sparse rank is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError('Sparse rank must be non-negative')
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class FusionMetrics(BaseModel):
    """
    Metrics for the fusion process.
    
    This class tracks metrics for the fusion process including
    latency, candidate counts, and fusion statistics.
    """
    dense_latency: float = Field(default=0.0, ge=0.0, description="Dense retrieval latency in seconds")
    sparse_latency: float = Field(default=0.0, ge=0.0, description="Sparse retrieval latency in seconds")
    fusion_latency: float = Field(default=0.0, ge=0.0, description="Fusion latency in seconds")
    total_latency: float = Field(default=0.0, ge=0.0, description="Total latency in seconds")
    dense_candidate_count: int = Field(default=0, ge=0, description="Number of candidates from dense retrieval")
    sparse_candidate_count: int = Field(default=0, ge=0, description="Number of candidates from sparse retrieval")
    fused_candidate_count: int = Field(default=0, ge=0, description="Number of candidates after fusion")
    dense_only_count: int = Field(default=0, ge=0, description="Number of candidates only in dense results")
    sparse_only_count: int = Field(default=0, ge=0, description="Number of candidates only in sparse results")
    overlap_count: int = Field(default=0, ge=0, description="Number of candidates in both results")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
