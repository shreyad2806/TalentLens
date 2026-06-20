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

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class SparseSearchResult(BaseModel):
    """
    Result from sparse BM25 retrieval.
    
    This schema represents a single search result from the BM25 sparse retrieval engine.
    It contains the query, candidate information, BM25 score, and matched terms.
    
    BM25 (Best Matching 25) is a ranking function used by search engines to estimate
    the relevance of documents to a given search query. It's based on the probabilistic
    retrieval framework developed in the 1970s and 1980s.
    
    The BM25 score is calculated using:
    - Term frequency (TF): How often the term appears in the document
    - Inverse document frequency (IDF): How rare the term is across all documents
    - Document length normalization: Accounts for varying document lengths
    - Parameters k1 and b: Tuning parameters for term saturation and length normalization
    
    Attributes:
        query: The original search query
        candidate_name: Name of the candidate
        resume_id: Unique identifier for the resume
        chunk_id: Unique identifier for the chunk
        section: Section of the resume
        bm25_score: BM25 relevance score
        metadata: Additional metadata about the result
        matched_terms: List of terms from the query that matched
        matched_text: Text content that matched the query
        rank: Rank position in the results
    """
    query: str = Field(..., description="The original search query")
    candidate_name: str = Field(..., description="Name of the candidate")
    resume_id: str = Field(..., description="Unique identifier for the resume")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    section: str = Field(..., description="Section of the resume")
    bm25_score: float = Field(..., ge=0.0, description="BM25 relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    matched_terms: List[str] = Field(default_factory=list, description="Terms from query that matched")
    matched_text: str = Field(..., description="Text content that matched the query")
    rank: int = Field(..., ge=0, description="Rank position in the results")

    @field_validator('bm25_score')
    @classmethod
    def validate_bm25_score(cls, v: float) -> float:
        """Validate that BM25 score is non-negative."""
        if v < 0:
            raise ValueError(f"BM25 score must be non-negative, got {v}")
        return v

    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        """Validate that rank is non-negative."""
        if v < 0:
            raise ValueError(f"Rank must be non-negative, got {v}")
        return v

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BM25Document(BaseModel):
    """
    Document representation for BM25 indexing.
    
    This schema represents a document in the BM25 index. Each chunk becomes one
    BM25 document with its tokens, metadata, and statistics.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        resume_id: Unique identifier for the resume
        section: Section of the resume
        candidate_name: Name of the candidate
        text: Original text content
        tokens: Tokenized text
        document_length: Number of tokens in the document
        metadata: Additional metadata
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    resume_id: str = Field(..., description="Resume identifier")
    section: str = Field(..., description="Section name")
    candidate_name: str = Field(..., description="Candidate name")
    text: str = Field(..., description="Original text content")
    tokens: List[str] = Field(..., description="Tokenized text")
    document_length: int = Field(..., ge=0, description="Number of tokens")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('document_length')
    @classmethod
    def validate_document_length(cls, v: int) -> int:
        """Validate that document length is non-negative."""
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
    cache_hit_latency: Optional[float] = Field(None, description="Cache hit latency")

    class Config:
        frozen = True
