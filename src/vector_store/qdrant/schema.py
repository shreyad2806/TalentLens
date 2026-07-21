"""
Schema definitions for Qdrant Vector Store Adapter.

This module defines Pydantic schemas for Qdrant collections, payloads,
filters, and health status.

SOLID Principles Applied:
- Single Responsibility: Schema definitions only
- Open/Closed: Open for extension with new fields
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from ...config import EMBEDDING_DIM


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


class VectorParams(BaseModel):
    """Parameters for vector configuration."""
    size: int = Field(..., description="Vector dimension size")
    distance: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric")


class HnswConfig(BaseModel):
    """HNSW index configuration parameters."""
    m: int = Field(default=16, description="Max number of connections per node")
    ef_construct: int = Field(default=100, description="Index construction parameter")
    full_scan_threshold: int = Field(default=10000, description="Threshold for full scan")


class QdrantCollectionConfig(BaseModel):
    """Configuration for Qdrant collection."""
    collection_name: str = Field(default="talentlens_candidates", description="Collection name")
    vector_size: int = Field(default=EMBEDDING_DIM, description="Vector dimension size")
    distance: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric")
    hnsw_config: Optional[HnswConfig] = Field(default=None, description="HNSW index configuration")
    
    class Config:
        frozen = True


class QdrantPayload(BaseModel):
    """
    Payload schema for Qdrant documents.
    
    This schema defines the structure of metadata stored alongside vectors.
    """
    resume_id: str = Field(..., description="Unique resume identifier")
    candidate_name: str = Field(..., description="Candidate name")
    chunk_id: str = Field(..., description="Chunk identifier within resume")
    section: str = Field(..., description="Resume section (e.g., Skills, Experience)")
    skills: List[str] = Field(default_factory=list, description="Candidate skills")
    experience: Optional[float] = Field(None, description="Years of experience")
    location: Optional[str] = Field(None, description="Candidate location")
    education: Optional[str] = Field(None, description="Education level")
    role: Optional[str] = Field(None, description="Current role")
    salary: Optional[float] = Field(None, description="Expected salary in LPA")
    notice_period: Optional[int] = Field(None, description="Notice period in days")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        frozen = True


class MatchOperator(str, Enum):
    """Match operators for filtering."""
    MATCH = "match"
    MATCH_ANY = "match_any"
    MATCH_TEXT = "match_text"
    RANGE = "range"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IS_NULL = "is_null"
    IS_EMPTY = "is_empty"


class QdrantFilter(BaseModel):
    """
    Filter schema for Qdrant queries.
    
    This schema defines the structure for filtering documents based on
    metadata fields.
    """
    skills: Optional[List[str]] = Field(None, description="Filter by skills (match_any)")
    experience_min: Optional[float] = Field(None, description="Minimum experience (gte)")
    experience_max: Optional[float] = Field(None, description="Maximum experience (lte)")
    location: Optional[str] = Field(None, description="Filter by location (match)")
    education: Optional[str] = Field(None, description="Filter by education (match)")
    role: Optional[str] = Field(None, description="Filter by role (match)")
    salary_min: Optional[float] = Field(None, description="Minimum salary (gte)")
    salary_max: Optional[float] = Field(None, description="Maximum salary (lte)")
    notice_period_max: Optional[int] = Field(None, description="Maximum notice period (lte)")
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """
        Convert to Qdrant filter format.
        
        Returns:
            Dictionary in Qdrant filter format
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchAny
        
        conditions = []
        
        if self.skills:
            conditions.append(
                FieldCondition(
                    key="skills",
                    match=MatchAny(any=self.skills)
                )
            )
        
        if self.experience_min is not None or self.experience_max is not None:
            conditions.append(
                FieldCondition(
                    key="experience",
                    range=Range(
                        gte=self.experience_min,
                        lte=self.experience_max
                    )
                )
            )
        
        if self.location:
            conditions.append(
                FieldCondition(
                    key="location",
                    match=MatchValue(value=self.location)
                )
            )
        
        if self.education:
            conditions.append(
                FieldCondition(
                    key="education",
                    match=MatchValue(value=self.education)
                )
            )
        
        if self.role:
            conditions.append(
                FieldCondition(
                    key="role",
                    match=MatchValue(value=self.role)
                )
            )
        
        if self.salary_min is not None or self.salary_max is not None:
            conditions.append(
                FieldCondition(
                    key="salary",
                    range=Range(
                        gte=self.salary_min,
                        lte=self.salary_max
                    )
                )
            )
        
        if self.notice_period_max is not None:
            conditions.append(
                FieldCondition(
                    key="notice_period",
                    range=Range(lte=self.notice_period_max)
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        return None


class HealthStatus(str, Enum):
    """Health status of Qdrant instance."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class QdrantHealthStatus(BaseModel):
    """Health status of Qdrant adapter."""
    status: HealthStatus = Field(..., description="Overall health status")
    connection_healthy: bool = Field(..., description="Connection to Qdrant is healthy")
    collection_exists: bool = Field(..., description="Collection exists")
    vector_count: int = Field(..., ge=0, description="Number of vectors in collection")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    latency_ms: Optional[float] = Field(None, description="Health check latency in milliseconds")
    
    class Config:
        frozen = True


class SearchResult(BaseModel):
    """Schema for search results from Qdrant."""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    payload: QdrantPayload = Field(..., description="Document payload")
    
    class Config:
        frozen = True


class UpsertResult(BaseModel):
    """Schema for upsert operation results."""
    upserted_count: int = Field(..., ge=0, description="Number of vectors upserted")
    operation_id: Optional[str] = Field(None, description="Operation ID")
    latency_ms: float = Field(..., ge=0, description="Operation latency in milliseconds")
    
    class Config:
        frozen = True


class CollectionInfo(BaseModel):
    """Schema for collection information."""
    name: str = Field(..., description="Collection name")
    vectors_count: int = Field(..., ge=0, description="Number of vectors")
    indexed_vectors_count: int = Field(..., ge=0, description="Number of indexed vectors")
    points_count: int = Field(..., ge=0, description="Number of points")
    segments_count: int = Field(..., ge=0, description="Number of segments")
    status: str = Field(..., description="Collection status")
    config: QdrantCollectionConfig = Field(..., description="Collection configuration")
    
    class Config:
        frozen = True
