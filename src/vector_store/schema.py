"""Schema module - VectorRecord schema for vector store abstraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models import ResumeMetadata


class VectorRecord(BaseModel):
    """Schema for a vector record in the vector store."""

    id: str = Field(..., description="Unique identifier for the vector record")
    chunk_id: str = Field(..., description="ID of the chunk this record represents")
    section: str = Field(..., description="Section of the resume")
    text: str | None = Field(None, description="Chunk text that was embedded")
    chunk_text: str | None = Field(None, description="Full chunk text for retrieval/re-ranking")
    original_text: str | None = Field(None, description="Original resume text source if applicable")
    vector: list[float] = Field(..., description="Embedding vector")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")

    @computed_field
    @property
    def resume_id(self) -> str:
        return self.resume_metadata.resume_id

    @computed_field
    @property
    def candidate_name(self) -> Any:
        return self.resume_metadata.candidate_name

    @field_validator('vector')
    @classmethod
    def validate_vector_not_empty(cls, v: list[float]) -> list[float]:
        if not v or len(v) == 0:
            raise ValueError("Vector cannot be empty")
        return v

    @field_validator('vector')
    @classmethod
    def validate_vector_no_nan(cls, v: list[float]) -> list[float]:
        import math
        if any(math.isnan(x) for x in v):
            raise ValueError("Vector cannot contain NaN values")
        return v

    @field_validator('id')
    @classmethod
    def validate_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v

    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk ID cannot be empty")
        return v

    @field_validator('section')
    @classmethod
    def validate_section_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Section cannot be empty")
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
