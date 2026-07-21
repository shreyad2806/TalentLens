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

from .qdrant_adapter import QdrantAdapter
from .collection_manager import CollectionManager
from .health_check import HealthCheck
from .schema import (
    QdrantCollectionConfig,
    QdrantPayload,
    QdrantFilter,
    QdrantHealthStatus,
)

__all__ = [
    "QdrantAdapter",
    "CollectionManager",
    "HealthCheck",
    "QdrantCollectionConfig",
    "QdrantPayload",
    "QdrantFilter",
    "QdrantHealthStatus",
]
