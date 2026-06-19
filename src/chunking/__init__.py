"""
Chunking Package - Semantic section chunking for resume documents.

This package provides a clean, extensible chunking layer for breaking down
ResumeDocument objects into semantic chunks suitable for RAG ingestion.

Modules:
    - schema: Data models for chunks
    - semantic_chunker: Semantic section detection and chunking
    - chunk_generator: Chunk object generation from ResumeDocument
    - chunk_validator: Chunk validation and filtering
    - chunk_service: Unified orchestration layer

The chunking layer preserves metadata and follows logical resume boundaries
instead of using fixed-size token chunking.
"""

from .schema import Chunk
from .chunk_service import ChunkService

__all__ = [
    "Chunk",
    "ChunkService",
]
