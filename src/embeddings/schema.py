"""Embedding Schema module - Pydantic models for embedding records."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from src.models import ResumeMetadata


class EmbeddingRecord(BaseModel):
    """Embedding record carrying the canonical ResumeMetadata unchanged."""

    embedding_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this embedding record")
    chunk_id: str = Field(..., description="ID of the source chunk")
    section: str = Field(..., description="Section type of the source chunk")
    text: str | None = Field(None, description="Chunk text that was embedded")
    vector: list[float] = Field(..., description="The embedding vector as a list of floats")
    vector_dimension: int = Field(..., description="Dimension of the embedding vector")
    model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="Name of the model used")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")

    @field_validator('vector')
    @classmethod
    def validate_vector_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Vector cannot be empty")
        return v

    @field_validator('vector')
    @classmethod
    def validate_vector_no_nan(cls, v: list[float]) -> list[float]:
        import math
        if any(math.isnan(val) for val in v):
            raise ValueError("Vector cannot contain NaN values")
        return v

    @field_validator('vector_dimension')
    @classmethod
    def validate_dimension_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v

    @field_validator('vector_dimension')
    @classmethod
    def validate_dimension_matches_vector(cls, v: int, info) -> int:
        if 'vector' in info.data and len(info.data['vector']) != v:
            raise ValueError(f"Vector dimension {v} does not match actual vector length {len(info.data['vector'])}")
        return v

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
