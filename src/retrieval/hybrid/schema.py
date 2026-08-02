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

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models import ResumeMetadata


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
    offset: int = Field(default=0, ge=0, description="Character offset of matched_text in the original chunk text")
    retrieval_source: RetrievalSource = Field(
        ...,
        description="Source of the retrieval (dense, sparse, or hybrid)"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class HybridSearchResult(BaseModel):
    """Hybrid search result carrying the canonical ResumeMetadata unchanged."""

    query: str = Field(..., description="The search query")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    section: str = Field(..., description="Section of the resume")
    dense_rank: int | None = Field(default=None, description="Rank from dense retrieval")
    sparse_rank: int | None = Field(default=None, description="Rank from sparse retrieval")
    rrf_score: float = Field(default=0.0, ge=0.0, description="Reciprocal Rank Fusion score")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    matched_chunks: list[MatchedChunk] = Field(default_factory=list, description="Matched chunks")
    rank: int = Field(default=0, ge=0, description="Final rank")

    @computed_field
    @property
    def resume_id(self) -> str:
        return self.resume_metadata.resume_id

    @computed_field
    @property
    def candidate_name(self) -> str | None:
        return self.resume_metadata.candidate_name

    @computed_field
    @property
    def metadata(self) -> dict[str, Any]:
        """Legacy compatibility: expose resume metadata as a flat dict."""
        return self.resume_metadata.model_dump(mode="json")
    
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
    def validate_dense_rank(cls, v: int | None) -> int | None:
        """Validate that dense rank is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError('Dense rank must be non-negative')
        return v
    
    @field_validator('sparse_rank')
    @classmethod
    def validate_sparse_rank(cls, v: int | None) -> int | None:
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
