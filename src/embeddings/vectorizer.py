"""
Vectorizer module - Chunk to EmbeddingRecord conversion.

This module provides the Vectorizer class that converts Chunk objects into
EmbeddingRecord objects. Each Chunk is converted to exactly one EmbeddingRecord
with its corresponding embedding vector.

The vectorizer does NOT concatenate chunks - one Chunk becomes one Vector.
"""

from typing import List
from .schema import EmbeddingRecord
from .model_loader import get_model_loader
from .cache import get_embedding_cache


class Vectorizer:
    """
    Vectorizer for converting Chunk objects to EmbeddingRecord objects.
    
    This class handles the conversion of Chunk objects into EmbeddingRecord objects
    by generating embedding vectors for each chunk. The vectorizer uses the
    singleton model loader and embedding cache for efficient processing.
    
    The vectorizer follows the principle: One Chunk → One Vector.
    Chunks are never concatenated; each chunk gets its own embedding.
    """
    
    def __init__(self):
        """
        Initialize the vectorizer.
        """
        self.model_loader = get_model_loader()
        self.cache = get_embedding_cache()
    
    def vectorize_chunk(self, chunk) -> EmbeddingRecord:
        """
        Convert a single Chunk object to an EmbeddingRecord.
        
        This method:
        1. Checks the cache for existing embedding
        2. If not cached, generates new embedding
        3. Creates EmbeddingRecord with all metadata
        4. Caches the result
        
        Args:
            chunk: Chunk object to vectorize
            
        Returns:
            EmbeddingRecord with the chunk's embedding vector
        """
        # Check cache first
        cached_embedding = self.cache.get(chunk.text)
        
        if cached_embedding is not None:
            # Use cached embedding
            vector = cached_embedding
        else:
            # Generate new embedding
            model = self.model_loader.get_model()
            vector = model.encode(chunk.text).tolist()
            
            # Cache the result
            self.cache.set(chunk.text, vector)
        
        # Create EmbeddingRecord
        embedding_record = EmbeddingRecord(
            chunk_id=chunk.chunk_id,
            resume_id=chunk.resume_id,
            candidate_name=chunk.candidate_name,
            section=chunk.section,
            vector=vector,
            vector_dimension=len(vector),
            model_name=self.model_loader.get_model_name(),
            metadata={
                'chunk_order': chunk.chunk_order,
                'source_section': chunk.metadata.source_section if chunk.metadata else None,
                'text_length': len(chunk.text)
            }
        )
        
        return embedding_record
    
    def vectorize_chunks(self, chunks: List) -> List[EmbeddingRecord]:
        """
        Convert multiple Chunk objects to EmbeddingRecord objects.
        
        This method processes each chunk individually, generating one embedding
        per chunk. Chunks are never concatenated.
        
        Args:
            chunks: List of Chunk objects to vectorize
            
        Returns:
            List of EmbeddingRecord objects
        """
        embedding_records: List[EmbeddingRecord] = []
        
        for chunk in chunks:
            embedding_record = self.vectorize_chunk(chunk)
            embedding_records.append(embedding_record)
        
        return embedding_records
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
