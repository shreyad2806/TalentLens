"""
Chunk Service module - Unified orchestration layer for chunking.

This module provides the main ChunkService class that orchestrates the entire
chunking pipeline: semantic chunking, chunk generation, and validation.
"""

from typing import List, Optional
import uuid
import logging

from ..resume_parser.schema import ResumeDocument
from .chunk_generator import ChunkGenerator
from .chunk_validator import ChunkValidator
from .schema import Chunk

logger = logging.getLogger(__name__)


class ChunkService:
    """
    Unified chunking service for resume documents.
    
    This class orchestrates the entire chunking pipeline:
    1. Use SemanticChunker to break document into logical sections
    2. Use ChunkGenerator to create Chunk objects with proper metadata
    3. Use ChunkValidator to filter invalid chunks
    
    The service provides a clean, high-level interface for chunking resumes
    while keeping the underlying components modular and testable.
    """
    
    def __init__(self, max_chunk_length: Optional[int] = None):
        """
        Initialize the chunk service with component modules.
        
        Args:
            max_chunk_length: Maximum allowed chunk length in characters.
                            If None, uses validator's default.
        """
        self.chunk_generator = ChunkGenerator()
        self.chunk_validator = ChunkValidator(max_chunk_length=max_chunk_length)
    
    def generate_chunks(self, document: ResumeDocument, resume_id: Optional[str] = None) -> List[Chunk]:
        """
        Generate semantic chunks from a ResumeDocument.
        
        This is the main entry point for chunking resumes. It handles the entire
        pipeline from document to validated chunks.
        
        Args:
            document: The ResumeDocument to chunk
            resume_id: Unique identifier for the resume. If None, generates a UUID.
            
        Returns:
            List of valid Chunk objects
            
        Raises:
            ValueError: If document is None
        """
        if document is None:
            raise ValueError("ResumeDocument cannot be None")
        
        # Generate resume_id if not provided
        if resume_id is None:
            resume_id = str(uuid.uuid4())
        
        # Step 1: Generate chunks using ChunkGenerator
        chunks = self.chunk_generator.generate_chunks(document, resume_id)
        
        # Step 2: Validate chunks using ChunkValidator
        valid_chunks = self.chunk_validator.validate_chunks(chunks)
        
        # Step 3: Get validation stats (optional, for logging/debugging)
        stats = self.chunk_validator.get_validation_stats(chunks, valid_chunks)

        # Log validation stats
        if stats['filtered_chunks'] > 0:
            logger.info(f"Filtered {stats['filtered_chunks']} invalid chunks ({stats['filter_rate']})")

        return valid_chunks
    
    def generate_chunks_with_stats(self, document: ResumeDocument, 
                                   resume_id: Optional[str] = None) -> dict:
        """
        Generate chunks and return with validation statistics.
        
        This method is useful for debugging and monitoring the chunking process.
        
        Args:
            document: The ResumeDocument to chunk
            resume_id: Unique identifier for the resume. If None, generates a UUID.
            
        Returns:
            Dictionary with 'chunks' and 'stats' keys
        """
        if document is None:
            raise ValueError("ResumeDocument cannot be None")
        
        # Generate resume_id if not provided
        if resume_id is None:
            resume_id = str(uuid.uuid4())
        
        # Generate chunks
        chunks = self.chunk_generator.generate_chunks(document, resume_id)
        
        # Validate chunks
        valid_chunks = self.chunk_validator.validate_chunks(chunks)
        
        # Get statistics
        stats = self.chunk_validator.get_validation_stats(chunks, valid_chunks)
        
        return {
            'chunks': valid_chunks,
            'stats': stats
        }
