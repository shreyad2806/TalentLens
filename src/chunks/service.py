"""
Service module - ChunkService as the main entry point for chunk creation.

This module implements the ChunkService class that provides a clean interface
for creating Chunk objects from ResumeDocument. It orchestrates the factory
and validator components.
"""

import uuid
from typing import List, Optional
import logging

from ..resume_parser.schema import ResumeDocument
from .factory import ChunkFactory
from .validator import ChunkValidator
from .schema import Chunk

logger = logging.getLogger(__name__)


class ChunkService:
    """
    Service for creating Chunk objects from ResumeDocument.
    
    This class provides the main entry point for chunk creation. It orchestrates
    the ChunkFactory and ChunkValidator components to ensure that only valid
    Chunk objects are returned.
    
    The service follows the Single Responsibility Principle by focusing on
    orchestration, while delegating creation to the factory and validation
    to the validator.
    """
    
    def __init__(self, max_chunk_length: Optional[int] = None):
        """
        Initialize the ChunkService with component modules.
        
        Args:
            max_chunk_length: Maximum allowed chunk length in characters.
                            If None, uses validator's default.
        """
        self.chunk_factory = ChunkFactory()
        self.chunk_validator = ChunkValidator(max_chunk_length=max_chunk_length)
    
    def create_chunks(self, document: ResumeDocument, 
                     resume_id: Optional[str] = None,
                     source_document: Optional[str] = None) -> List[Chunk]:
        """
        Create Chunk objects from a ResumeDocument.
        
        This is the main entry point for chunk creation. It handles the entire
        process from document to validated chunks.
        
        Args:
            document: The ResumeDocument to convert
            resume_id: Unique identifier for the resume. If None, generates a UUID.
            source_document: Optional source document identifier or path
            
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
        
        # Step 1: Create chunks using ChunkFactory
        chunks = self.chunk_factory.create_chunks(
            document=document,
            resume_id=resume_id,
            source_document=source_document
        )
        
        # Step 2: Validate chunks using ChunkValidator
        valid_chunks = self.chunk_validator.validate_chunks(chunks)

        # Step 3: Get validation stats (optional, for logging/debugging)
        stats = self.chunk_validator.get_validation_stats(chunks, valid_chunks)

        # Log validation stats
        if stats['filtered_chunks'] > 0:
            logger.info(f"Filtered {stats['filtered_chunks']} invalid chunks ({stats['filter_rate']})")

        return valid_chunks
    
    def create_chunks_with_stats(self, document: ResumeDocument,
                                 resume_id: Optional[str] = None,
                                 source_document: Optional[str] = None) -> dict:
        """
        Create chunks and return with validation statistics.
        
        This method is useful for debugging and monitoring the chunk creation process.
        
        Args:
            document: The ResumeDocument to convert
            resume_id: Unique identifier for the resume. If None, generates a UUID.
            source_document: Optional source document identifier or path
            
        Returns:
            Dictionary with 'chunks' and 'stats' keys
        """
        if document is None:
            raise ValueError("ResumeDocument cannot be None")
        
        # Generate resume_id if not provided
        if resume_id is None:
            resume_id = str(uuid.uuid4())
        
        # Create chunks
        chunks = self.chunk_factory.create_chunks(
            document=document,
            resume_id=resume_id,
            source_document=source_document
        )
        
        # Validate chunks
        valid_chunks = self.chunk_validator.validate_chunks(chunks)
        
        # Get statistics
        stats = self.chunk_validator.get_validation_stats(chunks, valid_chunks)
        
        return {
            'chunks': valid_chunks,
            'stats': stats
        }
