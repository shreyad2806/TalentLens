"""Schema module - Pydantic data models for Chunk objects."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models import ResumeMetadata


class EmbeddingStatus(str, Enum):
    """Enumeration for embedding status of chunks."""
    PENDING = "pending"
    EMBEDDED = "embedded"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """Legacy chunk-level metadata. Retained for Stage 2 compatibility only."""

    candidate_name: str | None = Field(None, description="Full name of the candidate")
    role: str | None = Field(None, description="Current or primary role")
    experience: int | None = Field(None, description="Years of experience")
    location: str | None = Field(None, description="Geographic location")
    education: str | None = Field(None, description="Education level or institution")
    skills: list[str] = Field(default_factory=list, description="List of skills from the resume")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number")
    summary: str | None = Field(None, description="Professional summary / objective")
    certifications: list[str] | None = Field(default_factory=list, description="Certifications")
    projects: list[str] | None = Field(default_factory=list, description="Project names")
    source_section: str | None = Field(None, description="Original section name")
    extraction_notes: str | None = Field(None, description="Extraction notes")

    def to_dict(self) -> dict[str, Any]:
        return self.dict()


class Chunk(BaseModel):
    """Production-grade Chunk object for resume documents."""

    chunk_id: str = Field(..., description="Unique chunk identifier (UUID)")
    resume_id: str = Field(..., description="Resume identifier")
    candidate_name: str | None = Field(None, description="Candidate name")
    section: str = Field(..., description="Section name")
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Legacy chunk metadata (Stage 2 compatibility)")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    chunk_order: int = Field(..., description="Order within resume")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    embedding_status: EmbeddingStatus = Field(
        default=EmbeddingStatus.PENDING,
        description="Status of embedding process"
    )
    source_document: str | None = Field(None, description="Source document identifier or path")

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    @field_validator('chunk_order')
    @classmethod
    def validate_chunk_order_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Chunk order cannot be negative")
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
