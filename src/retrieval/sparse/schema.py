"""
Schema for Sparse Retrieval Service.

This module defines Pydantic schemas for sparse retrieval results and related data structures.

Architecture Notes:
- Pydantic models for data validation
- Frozen models for immutability
- Field validators for data integrity
- Type safety throughout

SOLID Principles Applied:
- Single Responsibility: Schema definitions only
- Open/Closed: Open for extension with new fields
"""

from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models import ResumeMetadata


class SparseSearchResult(BaseModel):
    """Result from sparse BM25 retrieval, carrying the canonical ResumeMetadata."""

    query: str = Field(..., description="The original search query")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    section: str = Field(..., description="Section of the resume")
    bm25_score: float = Field(..., ge=0.0, description="BM25 relevance score")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    matched_terms: list[str] = Field(default_factory=list, description="Terms from query that matched")
    matched_text: str = Field(..., description="Text content that matched the query")
    offset: int = Field(default=0, ge=0, description="Character offset of matched_text in the original chunk text")
    rank: int = Field(..., ge=0, description="Rank position in the results")

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

    @field_validator('bm25_score')
    @classmethod
    def validate_bm25_score(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"BM25 score must be non-negative, got {v}")
        return v

    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Rank must be non-negative, got {v}")
        return v


class BM25Document(BaseModel):
    """Document representation for BM25 indexing. Carries ResumeMetadata unchanged."""

    document_id: str = Field(..., description="Unique document identifier (defaults to chunk_id)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    section: str = Field(..., description="Section name")
    text: str = Field(..., description="Original text content")
    tokens: list[str] = Field(..., description="Tokenized text")
    document_length: int = Field(..., ge=0, description="Number of tokens")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")

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

    @field_validator('document_length')
    @classmethod
    def validate_document_length(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Document length must be non-negative, got {v}")
        return v

    class Config:
        frozen = True


class BM25IndexStats(BaseModel):
    """
    Statistics about the BM25 index.
    
    This schema provides metadata about the BM25 index including vocabulary size,
    document count, and average document length.
    
    Attributes:
        num_documents: Total number of documents in the index
        vocabulary_size: Size of the vocabulary (unique terms)
        average_document_length: Average document length in tokens
        total_tokens: Total number of tokens across all documents
        index_build_time: Time taken to build the index in seconds
    """
    num_documents: int = Field(..., ge=0, description="Total number of documents")
    vocabulary_size: int = Field(..., ge=0, description="Size of vocabulary")
    average_document_length: float = Field(..., ge=0.0, description="Average document length")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens")
    index_build_time: float = Field(..., ge=0.0, description="Index build time in seconds")

    class Config:
        frozen = True


class RetrievalMetrics(BaseModel):
    """
    Performance metrics for sparse retrieval.
    
    This schema tracks performance metrics for sparse retrieval operations including
    latency, cache statistics, and document statistics.
    
    Attributes:
        query_latency: Total query latency in seconds
        tokenization_latency: Tokenization latency in seconds
        scoring_latency: BM25 scoring latency in seconds
        filtering_latency: Metadata filtering latency in seconds
        total_latency: Total end-to-end latency in seconds
        documents_searched: Number of documents searched
        vocabulary_size: Size of the vocabulary
        cache_hit: Whether query was served from cache
        cache_hit_latency: Cache hit latency (if applicable)
    """
    query_latency: float = Field(..., ge=0.0, description="Total query latency in seconds")
    tokenization_latency: float = Field(..., ge=0.0, description="Tokenization latency in seconds")
    scoring_latency: float = Field(..., ge=0.0, description="BM25 scoring latency in seconds")
    filtering_latency: float = Field(..., ge=0.0, description="Metadata filtering latency in seconds")
    total_latency: float = Field(..., ge=0.0, description="Total end-to-end latency in seconds")
    documents_searched: int = Field(..., ge=0, description="Number of documents searched")
    vocabulary_size: int = Field(..., ge=0, description="Size of vocabulary")
    cache_hit: bool = Field(default=False, description="Whether query was served from cache")
    cache_hit_latency: float | None = Field(None, description="Cache hit latency")

    class Config:
        frozen = True
