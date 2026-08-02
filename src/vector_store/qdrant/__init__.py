"""
Qdrant Vector Store Adapter Package.

This package provides a production-ready Qdrant adapter for vector storage,
supporting collection management, metadata filtering, and health monitoring.

Components:
- schema: Pydantic schemas for Qdrant collections and payloads
- collection_manager: Collection lifecycle management
- health_check: Health monitoring and validation
- qdrant_adapter: Main adapter implementing vector store interface

Usage:
    from src.vector_store.qdrant import QdrantAdapter
    
    adapter = QdrantAdapter()
    adapter.create_collection()
    adapter.upsert_vectors(vectors)
    results = adapter.search(query_vector)
"""

from .collection_manager import CollectionManager
from .health_check import HealthCheck
from .qdrant_adapter import QdrantAdapter
from .schema import (
    QdrantCollectionConfig,
    QdrantFilter,
    QdrantHealthStatus,
    QdrantPayload,
)

__all__ = [
    "CollectionManager",
    "HealthCheck",
    "QdrantAdapter",
    "QdrantCollectionConfig",
    "QdrantFilter",
    "QdrantHealthStatus",
    "QdrantPayload",
]
