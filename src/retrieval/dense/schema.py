"""
Schema definitions for Dense Retrieval Service.

This module defines the data structures used by the dense retrieval service,
including the DenseSearchResult schema that represents a search result with
normalized scores and aggregated candidate information.

Architecture Notes:
- Pydantic schemas for data validation
- Type hints for better IDE support
- Immutable data structures for safety
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models import ResumeMetadata


class DenseSearchResult(BaseModel):
    """Dense search result carrying the canonical ResumeMetadata."""

    query: str = Field(..., description="The original search query")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    section: str = Field(..., description="Section of the resume")
    score: float = Field(..., ge=0.0, le=1.0, description="Raw similarity score")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0.0 - 1.0)")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    matched_text: str = Field(..., description="Text content that matched the query")
    offset: int = Field(default=0, ge=0, description="Character offset of matched_text")
    rank: int = Field(..., ge=0, description="Rank position")

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
    
    @field_validator('normalized_score')
    @classmethod
    def validate_normalized_score(cls, v: float) -> float:
        """
        Validate that normalized score is within valid range.
        
        Args:
            v: Normalized score value
            
        Returns:
            Validated normalized score
            
        Raises:
            ValueError: If score is outside valid range
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Normalized score must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """
        Validate that raw score is within valid range.
        
        Args:
            v: Raw score value
            
        Returns:
            Validated raw score
            
        Raises:
            ValueError: If score is outside valid range
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        """
        Validate that rank is non-negative.
        
        Args:
            v: Rank value
            
        Returns:
            Validated rank
            
        Raises:
            ValueError: If rank is negative
        """
        if v < 0:
            raise ValueError(f"Rank must be non-negative, got {v}")
        return v
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make the model immutable
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AggregatedCandidateResult(BaseModel):
    """
    Schema for an aggregated candidate result.
    
    This schema represents a candidate with aggregated scores from multiple
    chunks, used internally by the candidate aggregator.
    
    Fields:
        candidate_name: Name of the candidate
        resume_id: Unique identifier for the resume
        final_score: Final aggregated score
        section_scores: Dictionary of section-specific scores
        evidence_chunks: List of evidence chunks with their scores
        metadata: Additional metadata
    """
    
    candidate_name: str = Field(..., description="Name of the candidate")
    resume_id: str = Field(..., description="Unique identifier for the resume")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final aggregated score")
    section_scores: dict[str, float] = Field(default_factory=dict, description="Section-specific scores")
    evidence_chunks: list[dict[str, Any]] = Field(default_factory=list, description="Evidence chunks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class RetrievalMetrics(BaseModel):
    """
    Schema for retrieval performance metrics.
    
    This schema captures performance metrics for retrieval operations,
    used for logging and monitoring.
    
    Fields:
        query_latency: Total query latency in seconds
        embedding_latency: Query embedding latency in seconds
        vector_latency: Vector store query latency in seconds
        aggregation_latency: Candidate aggregation latency in seconds
        total_latency: Total end-to-end latency in seconds
        retrieved_chunks: Number of chunks retrieved
        candidates: Number of unique candidates
        cache_hit: Whether the query was served from cache
    """
    
    query_latency: float = Field(..., ge=0.0, description="Total query latency in seconds")
    embedding_latency: float = Field(..., ge=0.0, description="Query embedding latency in seconds")
    vector_latency: float = Field(..., ge=0.0, description="Vector store query latency in seconds")
    aggregation_latency: float = Field(..., ge=0.0, description="Candidate aggregation latency in seconds")
    total_latency: float = Field(..., ge=0.0, description="Total end-to-end latency in seconds")
    retrieved_chunks: int = Field(..., ge=0, description="Number of chunks retrieved")
    candidates: int = Field(..., ge=0, description="Number of unique candidates")
    cache_hit: bool = Field(default=False, description="Whether query was served from cache")
    
    class Config:
        """Pydantic configuration."""
        frozen = True
