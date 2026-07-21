"""
Schema module - Data models for resume chunks.

This module defines Pydantic data models for chunks extracted from resume documents.
These models provide type safety, validation, and serialization capabilities.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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
    candidate_name: Optional[str] = Field(None, description="Full name of the candidate")
    experience: Optional[int] = Field(None, description="Years of experience")
    location: Optional[str] = Field(None, description="Geographic location")
    role: Optional[str] = Field(None, description="Current or primary role")
    education: Optional[str] = Field(None, description="Education level or institution")
    skills: List[str] = Field(default_factory=list, description="List of skills from the resume")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    summary: Optional[str] = Field(None, description="Professional summary / objective")
    source_section: Optional[str] = Field(None, description="Original section name")
    extraction_notes: Optional[str] = Field(None, description="Per-resume extraction log: sources, fallbacks, and missing fields")


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
    candidate_name: Optional[str] = Field(None, description="Candidate name")
    section: str = Field(..., description="Section name (e.g., 'experience_1', 'skills')")
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    chunk_order: int = Field(..., description="Order within resume")
    
    def to_dict(self) -> Dict[str, Any]:
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
