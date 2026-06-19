"""
Validator module - ChunkValidator for validating Chunk objects.

This module implements validation logic for Chunk objects to ensure data quality
before they are used for embedding or retrieval.
"""

from typing import List, Set, Optional
from .schema import Chunk


class ChunkValidator:
    """
    Validator for Chunk objects.
    
    This class validates Chunk objects to ensure they meet quality standards:
    - No duplicate chunk IDs
    - No duplicate text content
    - No empty chunks
    - No empty metadata
    - Valid section names
    - Chunk size within configurable limits
    
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
    
    # Default maximum chunk length (characters)
    DEFAULT_MAX_CHUNK_LENGTH = 5000
    
    def __init__(self, max_chunk_length: Optional[int] = None):
        """
        Initialize the ChunkValidator.
        
        Args:
            max_chunk_length: Maximum allowed chunk length in characters.
                            If None, uses DEFAULT_MAX_CHUNK_LENGTH.
        """
        self.max_chunk_length = max_chunk_length or self.DEFAULT_MAX_CHUNK_LENGTH
    
    def validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Validate a list of chunks and return only valid chunks.
        
        This method applies all validation rules in order:
        1. Remove chunks with duplicate IDs
        2. Remove chunks with duplicate text
        3. Remove empty chunks
        4. Remove chunks with empty metadata
        5. Remove chunks with invalid sections
        6. Remove oversized chunks
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of valid chunks
        """
        # Step 1: Remove duplicate chunk IDs
        valid_chunks = self._filter_duplicate_ids(chunks)
        
        # Step 2: Remove duplicate text
        valid_chunks = self._filter_duplicate_text(valid_chunks)
        
        # Step 3: Remove empty chunks
        valid_chunks = self._filter_empty_chunks(valid_chunks)
        
        # Step 4: Remove chunks with empty metadata
        valid_chunks = self._filter_empty_metadata(valid_chunks)
        
        # Step 5: Remove chunks with invalid sections
        valid_chunks = self._filter_invalid_sections(valid_chunks)
        
        # Step 6: Remove oversized chunks
        valid_chunks = self._filter_oversized_chunks(valid_chunks)
        
        return valid_chunks
    
    def _filter_duplicate_ids(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with duplicate chunk IDs.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with unique IDs
        """
        seen_ids: Set[str] = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _filter_duplicate_text(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with duplicate text content.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with unique text
        """
        seen_texts: Set[str] = set()
        unique_chunks = []
        
        for chunk in chunks:
            text_normalized = chunk.text.strip().lower()
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _filter_empty_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with empty or whitespace-only text.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with non-empty text
        """
        return [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
    
    def _filter_empty_metadata(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove chunks with empty metadata.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            List of chunks with non-empty metadata
        """
        return [chunk for chunk in chunks if chunk.metadata is not None]
    
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
        if not chunks:
            return {
                'total_chunks': 0,
                'valid_chunks': 0,
                'filtered_chunks': 0,
                'filter_rate': '0%'
            }
        
        filtered_count = len(chunks) - len(valid_chunks)
        filter_rate = (filtered_count / len(chunks)) * 100 if chunks else 0
        
        return {
            'total_chunks': len(chunks),
            'valid_chunks': len(valid_chunks),
            'filtered_chunks': filtered_count,
            'filter_rate': f"{filter_rate:.1f}%"
        }
    
    def validate_single_chunk(self, chunk: Chunk) -> bool:
        """
        Validate a single chunk.
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if chunk is valid, False otherwise
        """
        # Check for empty text
        if not chunk.text or not chunk.text.strip():
            return False
        
        # Check for empty metadata
        if not chunk.metadata:
            return False
        
        # Check for invalid section
        if chunk.section not in self.VALID_SECTIONS:
            return False
        
        # Check for oversized chunk
        if len(chunk.text) > self.max_chunk_length:
            return False
        
        return True
