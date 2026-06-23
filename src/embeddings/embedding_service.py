"""
Embedding Service module - Main entry point for embedding generation.

This module provides the EmbeddingService class as the main interface for
generating embeddings from Chunk objects. The service orchestrates the
entire embedding pipeline: validation, model loading, embedding generation,
and record creation.

The outside world should only call EmbeddingService.
"""

from typing import List, Optional
from .vectorizer import Vectorizer
from .validator import EmbeddingValidator
from .schema import EmbeddingRecord
from ..config import EMBEDDING_DIM, EMBEDDING_MODEL


class EmbeddingService:
    """
    Main service for generating embeddings from Chunk objects.
    
    This class provides a clean, high-level interface for generating embeddings.
    It orchestrates the entire embedding pipeline:
    1. Validate input chunks
    2. Load model (lazily, singleton)
    3. Generate embeddings (with caching)
    4. Create EmbeddingRecord objects
    5. Validate output embeddings
    6. Return valid records
    
    The service follows SOLID principles and provides a simple interface
    for the outside world.
    """
    
    def __init__(self, expected_dimension: Optional[int] = None):
        """
        Initialize the embedding service.
        
        Args:
            expected_dimension: Expected dimension of embedding vectors. If None, uses config default.
        """
        self.vectorizer = Vectorizer()
        self.validator = EmbeddingValidator(expected_dimension=expected_dimension or EMBEDDING_DIM)
        self._print_model_info()
    
    def _print_model_info(self) -> None:
        """
        Print embedding model information.
        
        This method prints the model name, dimension, and load time
        to provide visibility into the embedding configuration.
        """
        model_loader = self.vectorizer.model_loader
        diagnostics = model_loader.get_diagnostics()
        
        print("\n" + "="*60)
        print("🧠 Embedding Model Configuration")
        print("="*60)
        print(f"Model: {diagnostics['model_name']}")
        print(f"Dimension: {EMBEDDING_DIM}")
        print(f"Load Time: {diagnostics['load_time']:.2f}s")
        print(f"Device: {diagnostics['device']}")
        print(f"Memory Usage: {diagnostics['memory_usage_mb']:.2f} MB")
        print("="*60)
        print("🚀 Embedding Model Ready")
        print("="*60 + "\n")
    
    def embed_chunk(self, chunk) -> Optional[EmbeddingRecord]:
        """
        Generate an embedding for a single Chunk object.
        
        This method:
        1. Validates the chunk (checks for empty text)
        2. Generates embedding using the vectorizer
        3. Validates the resulting embedding record
        4. Returns the valid embedding record
        
        Args:
            chunk: Chunk object to embed
            
        Returns:
            EmbeddingRecord if successful, None if validation fails
        """
        # Validate chunk text is not empty
        if not self.validator.validate_text_not_empty(chunk.text):
            return None
        
        # Generate embedding record
        embedding_record = self.vectorizer.vectorize_chunk(chunk)
        
        # Validate the embedding record
        if not self.validator.validate_embedding_record(embedding_record):
            return None
        
        return embedding_record
    
    def embed_chunks(self, chunks: List) -> List[EmbeddingRecord]:
        """
        Generate embeddings for multiple Chunk objects.
        
        This method:
        1. Validates all chunks (checks for empty text)
        2. Generates embeddings for all chunks using the vectorizer
        3. Validates all resulting embedding records
        4. Filters out duplicates
        5. Returns only valid, unique embedding records
        
        Args:
            chunks: List of Chunk objects to embed
            
        Returns:
            List of valid, unique EmbeddingRecord objects
        """
        # Generate embedding records for all chunks
        embedding_records: List[EmbeddingRecord] = []
        
        for chunk in chunks:
            # Validate chunk text is not empty
            if not self.validator.validate_text_not_empty(chunk.text):
                continue
            
            # Generate embedding record
            embedding_record = self.vectorizer.vectorize_chunk(chunk)
            embedding_records.append(embedding_record)
        
        # Validate and filter records
        valid_records = self.validator.validate_and_filter(embedding_records)
        
        return valid_records
    
    def embed_chunks_with_stats(self, chunks: List) -> dict:
        """
        Generate embeddings for multiple Chunk objects with statistics.
        
        This method is similar to embed_chunks but also returns statistics
        about the embedding process, including cache hit rate and filter rate.
        
        Args:
            chunks: List of Chunk objects to embed
            
        Returns:
            Dictionary with:
                - 'embeddings': List of valid EmbeddingRecord objects
                - 'stats': Dictionary with embedding statistics
        """
        original_count = len(chunks)
        
        # Generate embedding records
        embedding_records = self.embed_chunks(chunks)
        
        # Get statistics
        validation_stats = self.validator.get_validation_stats(original_count, len(embedding_records))
        cache_stats = self.vectorizer.get_cache_stats()
        
        return {
            'embeddings': embedding_records,
            'stats': {
                'validation': validation_stats,
                'cache': cache_stats
            }
        }
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.vectorizer.get_cache_stats()
    
    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        
        This method clears all cached embeddings to free memory.
        """
        self.vectorizer.cache.clear()
