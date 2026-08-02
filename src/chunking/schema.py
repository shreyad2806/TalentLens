"""
Schema module - Data models for resume chunks.

This module defines Pydantic data models for chunks extracted from resume documents.
These models provide type safety, validation, and serialization capabilities.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.models import ResumeMetadata


class ChunkMetadata(BaseModel):
    """
    Metadata associated with a chunk.
    
    This metadata provides context about the chunk's source and the candidate's
    background, which is useful for retrieval and ranking.
    
    Attributes:
        candidate_name: Full name of the candidate
        experience: Years of experience
        location: Geographic location
        role: Current or primary role
        education: Education level or institution
        skills: List of skills extracted from the resume
        email: Email address of the candidate
        phone: Phone number of the candidate
        summary: Professional summary / objective
        source_section: The original section this chunk came from
    """
    candidate_name: str | None = Field(None, description="Full name of the candidate")
    experience: int | None = Field(None, description="Years of experience")
    location: str | None = Field(None, description="Geographic location")
    role: str | None = Field(None, description="Current or primary role")
    education: str | None = Field(None, description="Education level or institution")
    skills: list[str] = Field(default_factory=list, description="List of skills from the resume")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number")
    summary: str | None = Field(None, description="Professional summary / objective")
    source_section: str | None = Field(None, description="Original section name")
    extraction_notes: str | None = Field(None, description="Per-resume extraction log: sources, fallbacks, and missing fields")


class Chunk(BaseModel):
    """
    Represents a semantic chunk from a resume document.
    
    A chunk is a logical unit of text from a resume, typically corresponding
    to a section or subsection. Chunks are designed for RAG ingestion and
    preserve metadata for context-aware retrieval.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        resume_id: Identifier for the resume this chunk belongs to
        candidate_name: Name of the candidate
        section: Section name (e.g., "experience_1", "skills", "summary")
        text: The actual text content of the chunk
        metadata: Additional metadata about the chunk
        chunk_order: Order of this chunk within the resume
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    resume_id: str = Field(..., description="Resume identifier")
    candidate_name: str | None = Field(None, description="Candidate name")
    section: str = Field(..., description="Section name (e.g., 'experience_1', 'skills')")
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    chunk_order: int = Field(..., description="Order within resume")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Chunk to a dictionary.
        
        Returns:
            Dictionary representation of the chunk
        """
        return self.dict()
    
    def to_json(self) -> str:
        """
        Convert the Chunk to a JSON string.
        
        Returns:
            JSON string representation
        """
        import json
        return json.dumps(self.dict(), indent=2)
