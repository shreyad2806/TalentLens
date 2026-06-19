"""
Chunks Package - Production-grade Chunk Object layer.

This package provides a clean, production-ready Chunk Object layer for the
AI ingestion pipeline. It converts semantic sections from ResumeDocument into
structured Chunk objects with proper validation and serialization.

Modules:
    - schema: Pydantic data models for Chunk objects
    - factory: ChunkFactory for creating Chunk objects from semantic sections
    - validator: ChunkValidator for validating Chunk objects
    - service: ChunkService as the main entry point for chunk creation

The Chunk Object layer follows SOLID principles and maintains backward compatibility.
"""

from .schema import Chunk, ChunkMetadata
from .service import ChunkService

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkService",
]
