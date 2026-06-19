"""
Embeddings Package - Production Embedding Layer

This package provides a production-ready embedding layer for the RAG ingestion pipeline.
It handles model loading, embedding generation, validation, caching, and vectorization
of Chunk objects into EmbeddingRecord objects ready for vector database storage.

Components:
- schema: Pydantic models for embedding records
- model_loader: Singleton model loader with lazy loading
- cache: In-memory cache for duplicate text avoidance
- validator: Embedding quality validation
- vectorizer: Chunk to EmbeddingRecord conversion
- embedding_service: Main service orchestrating the embedding pipeline

The embedding layer follows SOLID principles and is designed to be modular,
testable, and production-ready.
"""

from .schema import EmbeddingRecord
from .embedding_service import EmbeddingService

__all__ = [
    "EmbeddingRecord",
    "EmbeddingService",
]
