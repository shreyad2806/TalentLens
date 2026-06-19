"""
Schema module - Pydantic data models for Chunk objects.

This module defines production-grade Pydantic models for Chunk objects
with proper validation, serialization, and helper methods.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class EmbeddingStatus(str, Enum):
    """
    Enumeration for embedding status of chunks.
    
    This tracks whether a chunk has been embedded for vector search.
    """
    PENDING = "pending"
    EMBEDDED = "embedded"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """
    Metadata associated with a Chunk object.
    
    This metadata provides context about the chunk's source and the candidate's
    background, which is useful for retrieval and ranking.
    
    Attributes:
        role: Current or primary role
        experience: Years of experience
        location: Geographic location
        education: Education level or institution
        source_section: The original section this chunk came from
    """
    role: Optional[str] = Field(None, description="Current or primary role")
    experience: Optional[int] = Field(None, description="Years of experience")
    location: Optional[str] = Field(None, description="Geographic location")
    education: Optional[str] = Field(None, description="Education level or institution")
    source_section: Optional[str] = Field(None, description="Original section name")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to a dictionary.
        
        Returns:
            Dictionary representation of the metadata
        """
        return self.dict()


class Chunk(BaseModel):
    """
    Production-grade Chunk object for resume documents.
    
    A Chunk is a logical unit of text from a resume, typically corresponding
    to a section or subsection. Chunks are designed for RAG ingestion and
    preserve metadata for context-aware retrieval.
    
    Attributes:
        chunk_id: Unique identifier for this chunk (UUID)
        resume_id: Identifier for the resume this chunk belongs to
        candidate_name: Name of the candidate
        section: Section name (e.g., "experience_1", "skills", "summary")
        text: The actual text content of the chunk
        metadata: Additional metadata about the chunk
        chunk_order: Order of this chunk within the resume
        created_at: Timestamp when chunk was created
        embedding_status: Status of embedding process
        source_document: Source document identifier or path
    """
    chunk_id: str = Field(..., description="Unique chunk identifier (UUID)")
    resume_id: str = Field(..., description="Resume identifier")
    candidate_name: Optional[str] = Field(None, description="Candidate name")
    section: str = Field(..., description="Section name (e.g., 'experience_1', 'skills')")
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    chunk_order: int = Field(..., description="Order within resume")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    embedding_status: EmbeddingStatus = Field(
        default=EmbeddingStatus.PENDING,
        description="Status of embedding process"
    )
    source_document: Optional[str] = Field(None, description="Source document identifier or path")
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """
        Validate that text is not empty.
        
        Args:
            v: Text value to validate
            
        Returns:
            Validated text value
            
        Raises:
            ValueError: If text is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v
    
    @field_validator('chunk_order')
    @classmethod
    def validate_chunk_order_non_negative(cls, v: int) -> int:
        """
        Validate that chunk_order is non-negative.
        
        Args:
            v: Chunk order value to validate
            
        Returns:
            Validated chunk order value
            
        Raises:
            ValueError: If chunk_order is negative
        """
        if v < 0:
            raise ValueError("Chunk order cannot be negative")
        return v
    
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
        return json.dumps(self.dict(), indent=2, default=str)
    
    def summary(self) -> str:
        """
        Get a summary of the chunk.
        
        Returns:
            Summary string with key information
        """
        text_preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return (
            f"Chunk(chunk_id='{self.chunk_id}', section='{self.section}', "
            f"candidate='{self.candidate_name}', text_length={len(self.text)}, "
            f"preview='{text_preview}')"
        )
    
    def is_embedded(self) -> bool:
        """
        Check if the chunk has been embedded.
        
        Returns:
            True if chunk is embedded, False otherwise
        """
        return self.embedding_status == EmbeddingStatus.EMBEDDED
    
    def mark_as_embedded(self) -> None:
        """
        Mark the chunk as embedded.
        
        This method updates the embedding_status to EMBEDDED.
        """
        self.embedding_status = EmbeddingStatus.EMBEDDED
    
    def mark_as_failed(self) -> None:
        """
        Mark the chunk as failed embedding.
        
        This method updates the embedding_status to FAILED.
        """
        self.embedding_status = EmbeddingStatus.FAILED
