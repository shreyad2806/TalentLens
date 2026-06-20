"""
Vector Store Abstraction Layer - Production-ready vector store interface.

This package provides a clean abstraction layer for vector store operations,
supporting multiple backends (Pinecone, Qdrant, Memory) through a unified interface.

Architecture Overview:
- Abstract Interface (VectorStore): Contract all adapters must implement
- VectorRecord Schema: Data model for vector records
- VectorStoreService: Main entry point for application code
- VectorStoreFactory: Creates appropriate adapter based on configuration
- Config: Environment-based configuration management
- Validator: Validates vector records before storage

SOLID Principles Applied:
- Single Responsibility: Each class has one reason to change
- Open/Closed: Open for extension (new adapters), closed for modification
- Liskov Substitution: Adapters are substitutable for VectorStore interface
- Interface Segregation: Focused, cohesive interfaces
- Dependency Inversion: High-level modules depend on abstractions, not concretions

Design Patterns Used:
- Adapter Pattern: Adapts different vector store APIs to common interface
- Factory Pattern: Creates adapter instances based on configuration
- Facade Pattern: VectorStoreService simplifies complex subsystem
- Strategy Pattern: Different storage strategies are interchangeable

Usage:
    from src.vector_store import VectorStoreService, VectorRecord
    
    # Create service (uses configuration from environment)
    service = VectorStoreService()
    
    # Upsert records
    service.upsert(records)
    
    # Query for similar vectors
    results = service.query(vector, k=10)
    
    # Delete records
    service.delete(ids)
    
    # Fetch records
    record = service.fetch(id)
    
    # Count records
    count = service.count()

Configuration:
    Set VECTOR_STORE_PROVIDER environment variable:
    - pinecone: Use Pinecone vector database
    - qdrant: Use Qdrant vector database
    - memory: Use in-memory storage (default, for testing)

Security Note:
    Application code should NEVER directly call Pinecone or other adapters.
    All vector store operations must go through VectorStoreService.
"""

from .schema import VectorRecord
from .interface import VectorStore, VectorStoreError
from .service import VectorStoreService
from .factory import VectorStoreFactory, create_vector_store
from .config import VectorStoreConfig, VectorStoreProvider, get_config
from .validator import VectorStoreValidator, ValidationError

__all__ = [
    # Schema
    'VectorRecord',
    
    # Interface
    'VectorStore',
    'VectorStoreError',
    
    # Service
    'VectorStoreService',
    
    # Factory
    'VectorStoreFactory',
    'create_vector_store',
    
    # Configuration
    'VectorStoreConfig',
    'VectorStoreProvider',
    'get_config',
    
    # Validator
    'VectorStoreValidator',
    'ValidationError'
]
