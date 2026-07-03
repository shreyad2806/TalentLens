"""
Embedding Schema module - Pydantic models for embedding records.

This module defines the data models for embedding records, providing type safety,
validation, and serialization capabilities for the embedding layer.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import UUID, uuid4


class EmbeddingRecord(BaseModel):
    """
    Embedding record representing a vector embedding of a chunk.
    
    This model encapsulates all information about an embedding, including:
    - Unique identifiers for tracking
    - Source chunk information
    - The embedding vector itself
    - Model metadata
    - Creation timestamp
    
    Attributes:
        embedding_id: Unique identifier for this embedding record
        chunk_id: ID of the source chunk
        resume_id: ID of the source resume
        candidate_name: Name of the candidate
        section: Section type of the source chunk
        vector: The embedding vector as a list of floats
        vector_dimension: Dimension of the embedding vector
        model_name: Name of the model used to generate the embedding
        created_at: Timestamp when the embedding was created
        metadata: Additional metadata about the embedding
    """
    
    embedding_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this embedding record")
    chunk_id: UUID = Field(..., description="ID of the source chunk")
    resume_id: str = Field(..., description="ID of the source resume")
    candidate_name: str = Field(..., description="Name of the candidate")
    section: str = Field(..., description="Section type of the source chunk")
    vector: List[float] = Field(..., description="The embedding vector as a list of floats")
    vector_dimension: int = Field(..., description="Dimension of the embedding vector")
    model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="Name of the model used to generate the embedding")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the embedding was created")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the embedding")
    
    @field_validator('vector')
    @classmethod
    def validate_vector_not_empty(cls, v: List[float]) -> List[float]:
        """Validate that the vector is not empty."""
        if not v:
            raise ValueError("Vector cannot be empty")
        return v
    
    @field_validator('vector')
    @classmethod
    def validate_vector_no_nan(cls, v: List[float]) -> List[float]:
        """Validate that the vector contains no NaN values."""
        import math
        if any(math.isnan(val) for val in v):
            raise ValueError("Vector cannot contain NaN values")
        return v
    
    @field_validator('vector_dimension')
    @classmethod
    def validate_dimension_positive(cls, v: int) -> int:
        """Validate that the vector dimension is positive."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
    
    @field_validator('vector_dimension')
    @classmethod
    def validate_dimension_matches_vector(cls, v: int, info) -> int:
        """Validate that the dimension matches the actual vector length."""
        if 'vector' in info.data and len(info.data['vector']) != v:
            raise ValueError(f"Vector dimension {v} does not match actual vector length {len(info.data['vector'])}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the embedding record to a dictionary.
        
        Returns:
            Dictionary representation of the embedding record
        """
        return {
            'embedding_id': str(self.embedding_id),
            'chunk_id': str(self.chunk_id),
            'resume_id': self.resume_id,
            'candidate_name': self.candidate_name,
            'section': self.section,
            'vector': self.vector,
            'vector_dimension': self.vector_dimension,
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the embedding record to JSON string.
        
        Returns:
            JSON string representation of the embedding record
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def summary(self) -> str:
        """
        Generate a summary of the embedding record.
        
        Returns:
            Summary string with key information
        """
        return (f"EmbeddingRecord(id={self.embedding_id}, chunk_id={self.chunk_id}, "
                f"resume_id={self.resume_id}, section={self.section}, "
                f"dimension={self.vector_dimension}, model={self.model_name})")
