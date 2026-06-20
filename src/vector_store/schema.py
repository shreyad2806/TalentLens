"""
Schema module - VectorRecord schema for vector store abstraction.

This module defines the VectorRecord schema used for storing and retrieving
vector embeddings in the vector store abstraction layer.

Architecture Notes:
- This is a pure data model following the Single Responsibility Principle
- Encapsulates vector record structure independent of storage implementation
- Compatible with all vector store adapters (Pinecone, Qdrant, Memory)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class VectorRecord(BaseModel):
    """
    Schema for a vector record in the vector store.
    
    This class represents a single vector record containing the embedding vector
    and associated metadata. It is used by all vector store adapters to ensure
    consistent data structure across different implementations.
    
    Architecture Pattern: Data Transfer Object (DTO)
    - Pure data model with validation
    - No business logic
    - Serializable for storage and transmission
    """
    
    id: str = Field(..., description="Unique identifier for the vector record")
    resume_id: str = Field(..., description="ID of the resume this record belongs to")
    chunk_id: str = Field(..., description="ID of the chunk this record represents")
    candidate_name: str = Field(..., description="Name of the candidate")
    section: str = Field(..., description="Section of the resume (e.g., skills, experience)")
    vector: List[float] = Field(..., description="Embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp when record was created")
    
    @validator('vector')
    def validate_vector_not_empty(cls, v):
        """
        Validate that vector is not empty.
        
        This ensures that we don't store empty vectors which would be useless
        for similarity search operations.
        
        Raises:
            ValueError: If vector is empty
        """
        if not v or len(v) == 0:
            raise ValueError("Vector cannot be empty")
        return v
    
    @validator('vector')
    def validate_vector_no_nan(cls, v):
        """
        Validate that vector contains no NaN values.
        
        NaN values can cause issues with distance calculations and similarity
        search operations in most vector databases.
        
        Raises:
            ValueError: If vector contains NaN values
        """
        import math
        if any(math.isnan(x) for x in v):
            raise ValueError("Vector cannot contain NaN values")
        return v
    
    @validator('id')
    def validate_id_not_empty(cls, v):
        """
        Validate that ID is not empty.
        
        Empty IDs would cause issues with upsert and delete operations.
        
        Raises:
            ValueError: If ID is empty
        """
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v
    
    @validator('resume_id')
    def validate_resume_id_not_empty(cls, v):
        """
        Validate that resume_id is not empty.
        
        Empty resume IDs would cause issues with resume-level operations.
        
        Raises:
            ValueError: If resume_id is empty
        """
        if not v or not v.strip():
            raise ValueError("Resume ID cannot be empty")
        return v
    
    @validator('chunk_id')
    def validate_chunk_id_not_empty(cls, v):
        """
        Validate that chunk_id is not empty.
        
        Empty chunk IDs would cause issues with chunk-level operations.
        
        Raises:
            ValueError: If chunk_id is empty
        """
        if not v or not v.strip():
            raise ValueError("Chunk ID cannot be empty")
        return v
    
    @validator('candidate_name')
    def validate_candidate_name_not_empty(cls, v):
        """
        Validate that candidate_name is not empty.
        
        Empty candidate names would make the data less useful for filtering.
        
        Raises:
            ValueError: If candidate_name is empty
        """
        if not v or not v.strip():
            raise ValueError("Candidate name cannot be empty")
        return v
    
    @validator('section')
    def validate_section_not_empty(cls, v):
        """
        Validate that section is not empty.
        
        Empty sections would make the data less useful for filtering.
        
        Raises:
            ValueError: If section is empty
        """
        if not v or not v.strip():
            raise ValueError("Section cannot be empty")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the VectorRecord to a dictionary.
        
        Returns:
            Dictionary representation of the record
        """
        return {
            'id': self.id,
            'resume_id': self.resume_id,
            'chunk_id': self.chunk_id,
            'candidate_name': self.candidate_name,
            'section': self.section,
            'vector': self.vector,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    def to_json(self) -> str:
        """
        Convert the VectorRecord to JSON string.
        
        Returns:
            JSON string representation of the record
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def dimension(self) -> int:
        """
        Get the dimension of the vector.
        
        Returns:
            Dimension of the vector
        """
        return len(self.vector)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
