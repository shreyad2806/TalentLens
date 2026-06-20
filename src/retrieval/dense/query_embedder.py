"""
Query Embedder for Dense Retrieval Service.

This module provides functionality to generate embeddings for recruiter queries
using the existing BGE-M3 model without reloading it.

Architecture Notes:
- Reuses existing embedding infrastructure
- Leverages embedding cache
- No model reloading
- Efficient query embedding

SOLID Principles Applied:
- Single Responsibility: Handles only query embedding
- Dependency Inversion: Depends on embedding service abstraction
- Interface Segregation: Focused embedding interface
"""

import logging
from typing import List, Optional
from src.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """
    Generates embeddings for recruiter queries.
    
    This class provides a simple interface for generating embeddings for
    recruiter queries using the existing BGE-M3 model. It reuses the
    existing embedding infrastructure and leverages the embedding cache
    to avoid recomputing embeddings for identical queries.
    
    Architecture Pattern: Facade Pattern
    - Simplifies embedding generation
    - Reuses existing infrastructure
    - Leverages caching
    - No model reloading
    
    Features:
        - Reuses existing BGE-M3 model
        - Leverages embedding cache
        - No model reloading
        - Efficient query embedding
    """
    
    def __init__(self, expected_dimension: int = 1024):
        """
        Initialize the query embedder.
        
        Args:
            expected_dimension: Expected dimension of embedding vectors (default: 1024)
        """
        # Initialize embedding service (this will reuse the existing model)
        try:
            self.embedding_service = EmbeddingService(expected_dimension=expected_dimension)
            self.dimension = expected_dimension
            logger.info("QueryEmbedder initialized with existing embedding service")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a recruiter query.
        
        This method generates an embedding for the given query using the
        existing BGE-M3 model. The embedding is cached by the embedding
        service to avoid recomputation for identical queries.
        
        Args:
            query: Recruiter query to embed
            
        Returns:
            Embedding vector for the query
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If embedding generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Create a mock chunk object for the query
        # The embedding service expects a Chunk object, so we create a minimal one
        from src.chunks.schema import Chunk, ChunkMetadata
        
        query_chunk = Chunk(
            chunk_id="query",
            resume_id="query",
            text=query,
            section="query",
            candidate_name="query",
            metadata=ChunkMetadata(),
            chunk_order=0
        )
        
        try:
            # Generate embedding using the embedding service
            embedding_record = self.embedding_service.embed_chunks([query_chunk])
            
            if not embedding_record:
                raise RuntimeError("Failed to generate embedding")
            
            # Extract the embedding vector
            embedding = embedding_record[0].embedding
            
            logger.debug(f"Generated embedding for query: {query[:50]}... (dimension: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}") from e
    
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries: List of recruiter queries to embed
            
        Returns:
            List of embedding vectors for the queries
        """
        if not queries:
            return []
        
        # Create mock chunk objects for the queries
        from src.chunks.schema import Chunk, ChunkMetadata
        
        query_chunks = []
        for i, query in enumerate(queries):
            if not query or not query.strip():
                raise ValueError(f"Query at index {i} cannot be empty")
            
            query_chunk = Chunk(
                chunk_id=f"query_{i}",
                resume_id="query",
                text=query,
                section="query",
                candidate_name="query",
                metadata=ChunkMetadata(),
                chunk_order=i
            )
            query_chunks.append(query_chunk)
        
        try:
            # Generate embeddings using the embedding service
            embedding_records = self.embedding_service.embed_chunks(query_chunks)
            
            if not embedding_record:
                raise RuntimeError("Failed to generate embeddings")
            
            # Extract the embedding vectors
            embeddings = [record.embedding for record in embedding_records]
            
            logger.info(f"Generated embeddings for {len(queries)} queries")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed queries: {e}")
            raise RuntimeError(f"Query embeddings failed: {e}") from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.dimension
