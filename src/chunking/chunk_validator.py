"""
Chunk Validator module - Validates and filters chunks.

This module validates chunks to ensure quality and removes invalid chunks
before they are used for RAG ingestion.
"""

from typing import List, Set
from .schema import Chunk


class ChunkValidator:
    """
    Validator for chunks generated from resume documents.
    
    This class validates chunks to ensure they meet quality standards:
    - No empty chunks
    - No duplicate chunks
    - Valid section names
    - No oversized chunks
    
    Invalid chunks are filtered out before being returned.
    """
    
    # Valid section names
    VALID_SECTIONS = {
        'summary',
        'skills',
        'experience_1', 'experience_2', 'experience_3', 'experience_4', 'experience_5',
        'project_1', 'project_2', 'project_3', 'project_4', 'project_5',
        'education_1', 'education_2', 'education_3', 'education_4', 'education_5',
        'certifications',
        'languages'
    }
    
    # Maximum chunk length (characters)
    MAX_CHUNK_LENGTH = 5000
    
    def __init__(self, max_chunk_length: int = None):
        """
        Initialize the chunk validator.
        
        Args:
            max_chunk_length: Maximum allowed chunk length in characters.
                            If None, uses default MAX_CHUNK_LENGTH.
        """
        self.max_chunk_length = max_chunk_length or self.MAX_CHUNK_LENGTH
    
    def validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Validate a list of chunks and return only valid chunks.
        
        This method applies all validation rules:
        1. Remove empty chunks
        2. Remove duplicate chunks
        3. Remove chunks with invalid sections
        4. Remove oversized chunks
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of valid chunks
        """
        # Step 1: Remove empty chunks
        valid_chunks = self._filter_empty_chunks(chunks)
        
        # Step 2: Remove duplicate chunks
        valid_chunks = self._filter_duplicate_chunks(valid_chunks)
        
        # Step 3: Remove chunks with invalid sections
        valid_chunks = self._filter_invalid_sections(valid_chunks)
        
        # Step 4: Remove oversized chunks
        valid_chunks = self._filter_oversized_chunks(valid_chunks)
        
        return valid_chunks
    
    def _filter_empty_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with empty or whitespace-only text.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with non-empty text
        """
        return [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
    
    def _filter_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks based on text content.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of unique chunks
        """
        seen_texts: Set[str] = set()
        unique_chunks = []
        
        for chunk in chunks:
            text_normalized = chunk.text.strip().lower()
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _filter_invalid_sections(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with invalid section names.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with valid section names
        """
        return [chunk for chunk in chunks if chunk.section in self.VALID_SECTIONS]
    
    def _filter_oversized_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks that exceed the maximum length.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks within size limits
        """
        return [chunk for chunk in chunks if len(chunk.text) <= self.max_chunk_length]
    
    def get_validation_stats(self, chunks: List[Chunk], valid_chunks: List[Chunk]) -> dict:
        """
        Get statistics about the validation process.
        
        Args:
            chunks: Original list of chunks
            valid_chunks: Validated list of chunks
            
        Returns:
            Dictionary with validation statistics
        """
        return {
            'total_chunks': len(chunks),
            'valid_chunks': len(valid_chunks),
            'filtered_chunks': len(chunks) - len(valid_chunks),
            'filter_rate': f"{((len(chunks) - len(valid_chunks)) / len(chunks) * 100):.1f}%" if chunks else "0%"
        }
