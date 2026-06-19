"""
Schema module - BM25Document schema for BM25 retrieval.

This module defines the BM25Document schema used for storing and retrieving
documents in the BM25 sparse retrieval index.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class BM25Document(BaseModel):
    """
    Schema for a BM25 document.
    
    Each BM25 document represents a chunk from a resume that has been indexed
    for sparse retrieval using the BM25 algorithm.
    """
    
    document_id: str = Field(..., description="Unique identifier for the BM25 document")
    chunk_id: str = Field(..., description="ID of the chunk this document represents")
    resume_id: str = Field(..., description="ID of the resume this chunk belongs to")
    candidate_name: str = Field(..., description="Name of the candidate")
    section: str = Field(..., description="Section of the resume (e.g., skills, experience)")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    token_count: int = Field(default=0, description="Number of tokens in the document")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp when document was created")
    
    @validator('text')
    def validate_text_not_empty(cls, v):
        """Validate that text is not empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v
    
    @validator('token_count')
    def validate_token_count(cls, v):
        """Validate that token count is non-negative."""
        if v < 0:
            raise ValueError("Token count must be non-negative")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the BM25Document to a dictionary.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            'document_id': self.document_id,
            'chunk_id': self.chunk_id,
            'resume_id': self.resume_id,
            'candidate_name': self.candidate_name,
            'section': self.section,
            'text': self.text,
            'metadata': self.metadata,
            'token_count': self.token_count,
            'created_at': self.created_at
        }
    
    def to_json(self) -> str:
        """
        Convert the BM25Document to JSON string.
        
        Returns:
            JSON string representation of the document
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
