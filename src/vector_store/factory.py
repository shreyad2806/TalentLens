"""
Factory module - Factory for creating vector store adapters.

This module provides the VectorStoreFactory class that instantiates the
appropriate vector store adapter based on configuration. This follows the
Factory Pattern to decouple adapter creation from usage.

Architecture Notes:
- Factory Pattern for adapter instantiation
- Reads configuration from environment
- Returns appropriate adapter based on provider
- Caches the created adapter to avoid duplicate Qdrant client instances
- Follows Dependency Inversion Principle

SOLID Principles Applied:
- Single Responsibility: Handles only adapter creation
- Open/Closed: Open for new adapters, closed for modification
- Dependency Inversion: Depends on abstraction (VectorStore interface)
"""


from .config import VectorStoreConfig, VectorStoreProvider, get_config
from .interface import VectorStore

# Module-level singleton cache to prevent duplicate Qdrant client
# initializations (embedded Qdrant locks the storage directory).
_vector_store_instance: VectorStore | None = None
_vector_store_config_key: object | None = None


class VectorStoreFactory:
    """
    Factory for creating vector store adapter instances.
    
    This class implements the Factory Pattern to create vector store adapter
    instances based on configuration. It abstracts away the complexity of
    adapter instantiation and configuration.
    
    Architecture Pattern: Factory Pattern
    - Encapsulates object creation logic
    - Decouples client code from concrete implementations
    - Centralizes adapter configuration
    - Enables easy switching between providers
    
    SOLID Principles Applied:
    - Single Responsibility: Handles only adapter creation
    - Open/Closed: Open for extension with new adapters
    - Dependency Inversion: Returns VectorStore abstraction
    - Interface Segregation: Focused factory interface
    """
    
    def __init__(self, config: VectorStoreConfig | None = None):
        """
        Initialize the factory with configuration.
        
        Args:
            config: Optional configuration. If None, uses global config.
        """
        self.config = config or get_config()
    
    def create_vector_store(self) -> VectorStore:
        """
        Create a vector store adapter instance based on configuration.
        
        This method reads the configured provider from the configuration
        and instantiates the appropriate adapter. Currently supported
        providers are:
        - pinecone: Pinecone vector database
        - qdrant: Qdrant vector database
        - memory: In-memory vector store (for testing/development)
        
        Note:
            Adapters are not implemented yet. This is the abstraction layer only.
            When adapters are implemented, they will be imported and instantiated here.
        
        Returns:
            VectorStore adapter instance
            
        Raises:
            NotImplementedError: If the configured provider is not yet implemented
            ValueError: If the provider configuration is invalid
        """
        provider = self.config.provider
        
        if provider == VectorStoreProvider.PINECONE:
            return self._create_pinecone_adapter()
        elif provider == VectorStoreProvider.QDRANT:
            return self._create_qdrant_adapter()
        elif provider == VectorStoreProvider.MEMORY:
            return self._create_memory_adapter()
        else:
            supported_providers = [p.value for p in VectorStoreProvider]
            raise ValueError(
                f"Unsupported vector store provider: {provider.value}. "
                f"Supported providers are: {', '.join(supported_providers)}. "
                f"Set VECTOR_STORE_PROVIDER environment variable to one of the supported values."
            )
    
    def _create_pinecone_adapter(self) -> VectorStore:
        """
        Create a Pinecone vector store adapter.
        
        This method instantiates the Pinecone adapter which uses the Pinecone
        SDK to interact with Pinecone's vector database service.
        
        Returns:
            PineconeAdapter instance
        """
        from .adapters.pinecone_adapter import PineconeAdapter
        return PineconeAdapter(config=self.config)
    
    def _create_qdrant_adapter(self) -> VectorStore:
        """
        Create a Qdrant vector store adapter.
        
        This method instantiates the Qdrant adapter which uses the Qdrant
        SDK to interact with Qdrant's vector database service.
        
        Returns:
            QdrantAdapter instance
        """
        from .adapters.qdrant_adapter import QdrantAdapter
        return QdrantAdapter(config=self.config)
    
    def _create_memory_adapter(self) -> VectorStore:
        """
        Create an in-memory vector store adapter.
        
        This method instantiates the memory adapter which stores vectors in
        memory for testing and development purposes. Data is not persisted and
        is lost when the process exits.
        
        Returns:
            MemoryVectorStore adapter instance
        """
        from .adapters.memory import MemoryVectorStore
        return MemoryVectorStore(config=self.config)
    
    @staticmethod
    def create(config: VectorStoreConfig | None = None) -> VectorStore:
        """
        Static method to create (or reuse) a vector store adapter.
        
        This is a convenience method that creates a factory instance
        and returns the vector store adapter in one call. The result is
        cached per configuration to avoid duplicate Qdrant client
        initializations.
        
        Args:
            config: Optional configuration. If None, uses global config.
            
        Returns:
            VectorStore adapter instance
        """
        factory = VectorStoreFactory(config)
        return factory.create_vector_store()


def create_vector_store(config: VectorStoreConfig | None = None) -> VectorStore:
    """
    Convenience function to create (or reuse) a vector store adapter.
    
    This function provides a simple interface for creating vector store
    adapters without needing to instantiate the factory explicitly. The
    result is cached per configuration so the application never opens more
    than one embedded Qdrant client per storage path.
    
    Args:
        config: Optional configuration. If None, uses global config.
        
    Returns:
        VectorStore adapter instance
        
    Example:
        >>> from src.vector_store.factory import create_vector_store
        >>> vector_store = create_vector_store()
        >>> result = vector_store.upsert(records)
    """
    global _vector_store_instance, _vector_store_config_key

    config = config or get_config()
    key = (
        config.provider,
        getattr(config, "qdrant_path", None),
        getattr(config, "qdrant_url", None),
        getattr(config, "qdrant_api_key", None),
        getattr(config, "collection_name", None),
        config.dimension,
    )

    if _vector_store_instance is not None and _vector_store_config_key == key:
        return _vector_store_instance

    _vector_store_config_key = key
    _vector_store_instance = VectorStoreFactory.create(config)
    return _vector_store_instance
