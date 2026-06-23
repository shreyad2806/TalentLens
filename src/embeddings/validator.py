"""
Validator module - Embedding quality validation.

This module provides validation logic for embedding records to ensure quality
before they are used for vector database storage or retrieval.

Validations include:
- Empty text detection
- Duplicate embedding detection
- Incorrect dimension validation
- NaN value detection
- Empty vector detection
"""

from typing import List, Optional, Set
from .schema import EmbeddingRecord
from ..config import EMBEDDING_DIM
import math


class EmbeddingValidator:
    """
    Validator for embedding records.
    
    This class provides methods to validate embedding records and filter out
    invalid embeddings based on various quality criteria.
    
    The validator helps ensure that only high-quality embeddings are stored
    in the vector database, improving retrieval accuracy and system reliability.
    """
    
    def __init__(self, expected_dimension: Optional[int] = None):
        """
        Initialize the embedding validator.
        
        Args:
            expected_dimension: Expected dimension of embedding vectors. If None, uses config default.
        """
        self.expected_dimension = expected_dimension or EMBEDDING_DIM
    
    def validate_text_not_empty(self, text: str) -> bool:
        """
        Validate that the text is not empty.
        
        Args:
            text: The text to validate
            
        Returns:
            True if text is not empty, False otherwise
        """
        return text is not None and len(text.strip()) > 0
    
    def validate_vector_not_empty(self, vector: List[float]) -> bool:
        """
        Validate that the vector is not empty.
        
        Args:
            vector: The vector to validate
            
        Returns:
            True if vector is not empty, False otherwise
        """
        return vector is not None and len(vector) > 0
    
    def validate_vector_dimension(self, vector: List[float]) -> bool:
        """
        Validate that the vector has the correct dimension.
        
        Args:
            vector: The vector to validate
            
        Returns:
            True if vector has correct dimension, False otherwise
        """
        return len(vector) == self.expected_dimension
    
    def validate_vector_no_nan(self, vector: List[float]) -> bool:
        """
        Validate that the vector contains no NaN values.
        
        Args:
            vector: The vector to validate
            
        Returns:
            True if vector has no NaN values, False otherwise
        """
        return not any(math.isnan(val) for val in vector)
    
    def validate_vector_no_inf(self, vector: List[float]) -> bool:
        """
        Validate that the vector contains no infinite values.
        
        Args:
            vector: The vector to validate
            
        Returns:
            True if vector has no infinite values, False otherwise
        """
        return not any(math.isinf(val) for val in vector)
    
    def validate_embedding_record(self, record: EmbeddingRecord) -> bool:
        """
        Validate a complete embedding record.
        
        This method performs all validations on the embedding record.
        
        Args:
            record: The embedding record to validate
            
        Returns:
            True if the record is valid, False otherwise
        """
        # Validate vector is not empty
        if not self.validate_vector_not_empty(record.vector):
            return False
        
        # Validate vector dimension
        if not self.validate_vector_dimension(record.vector):
            return False
        
        # Validate no NaN values
        if not self.validate_vector_no_nan(record.vector):
            return False
        
        # Validate no infinite values
        if not self.validate_vector_no_inf(record.vector):
            return False
        
        return True
    
    def filter_duplicates(self, records: List[EmbeddingRecord]) -> List[EmbeddingRecord]:
        """
        Filter out duplicate embedding records based on chunk_id.
        
        Args:
            records: List of embedding records to filter
            
        Returns:
            List of unique embedding records
        """
        seen_chunk_ids: Set[str] = set()
        unique_records: List[EmbeddingRecord] = []
        
        for record in records:
            chunk_id_str = str(record.chunk_id)
            if chunk_id_str not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id_str)
                unique_records.append(record)
        
        return unique_records
    
    def filter_invalid(self, records: List[EmbeddingRecord]) -> List[EmbeddingRecord]:
        """
        Filter out invalid embedding records.
        
        This method applies all validation rules and returns only valid records.
        
        Args:
            records: List of embedding records to filter
            
        Returns:
            List of valid embedding records
        """
        valid_records: List[EmbeddingRecord] = []
        
        for record in records:
            if self.validate_embedding_record(record):
                valid_records.append(record)
        
        return valid_records
    
    def validate_and_filter(self, records: List[EmbeddingRecord]) -> List[EmbeddingRecord]:
        """
        Validate and filter embedding records.
        
        This method combines duplicate filtering and validation filtering.
        
        Args:
            records: List of embedding records to validate and filter
            
        Returns:
            List of valid, unique embedding records
        """
        # First filter duplicates
        unique_records = self.filter_duplicates(records)
        
        # Then filter invalid records
        valid_records = self.filter_invalid(unique_records)
        
        return valid_records
    
    def get_validation_stats(self, original_count: int, filtered_count: int) -> dict:
        """
        Get validation statistics.
        
        Args:
            original_count: Original number of records
            filtered_count: Number of records after filtering
            
        Returns:
            Dictionary with validation statistics
        """
        filtered_out = original_count - filtered_count
        filter_rate = (filtered_out / original_count) if original_count > 0 else 0.0
        
        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'filtered_out': filtered_out,
            'filter_rate': filter_rate
        }
