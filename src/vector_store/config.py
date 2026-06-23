"""
Config module - Configuration management for vector store.

This module provides configuration management for the vector store abstraction
layer, reading settings from environment variables and providing sensible defaults.

Architecture Notes:
- Configuration follows the Single Responsibility Principle
- Environment-based configuration enables easy switching between providers
- Default values ensure the system works without explicit configuration
- Type-safe configuration with validation

SOLID Principles Applied:
- Single Responsibility: Handles only configuration management
- Open/Closed: Can be extended with new configuration options without modification
"""

import os
from typing import Optional
from enum import Enum

# Import embedding dimension from config to ensure alignment
from ..config import EMBEDDING_DIM


class VectorStoreProvider(Enum):
    """
    Supported vector store providers.
    
    This enum defines the supported vector store implementations that can be
    used through the abstraction layer. New providers can be added here as
    they are implemented.
    
    Architecture Pattern: Enumeration for Type Safety
    - Provides type-safe provider selection
    - Prevents typos in provider names
    - Easy to extend with new providers
    """
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    MEMORY = "memory"


class VectorStoreConfig:
    """
    Configuration class for vector store settings.
    
    This class manages all configuration for the vector store abstraction layer,
    reading from environment variables and providing sensible defaults.
    
    Architecture Pattern: Configuration Object Pattern
    - Centralized configuration management
    - Environment-based configuration
    - Type-safe access to configuration values
    - Validation of configuration values
    
    SOLID Principles Applied:
    - Single Responsibility: Handles only configuration
    - Dependency Inversion: Depends on abstractions (environment variables)
    """
    
    # Environment variable names
    ENV_VECTOR_STORE_PROVIDER = "VECTOR_STORE_PROVIDER"
    ENV_VECTOR_STORE_DIMENSION = "VECTOR_STORE_DIMENSION"
    ENV_VECTOR_STORE_METRIC = "VECTOR_STORE_METRIC"
    
    # Default values
    DEFAULT_PROVIDER = VectorStoreProvider.QDRANT
    DEFAULT_DIMENSION = EMBEDDING_DIM  # Use configured embedding dimension
    DEFAULT_METRIC = "cosine"
    
    def __init__(self):
        """
        Initialize the configuration from environment variables.
        
        Reads configuration from environment variables and applies defaults
        where values are not provided.
        """
        self._provider = self._load_provider()
        self._dimension = self._load_dimension()
        self._metric = self._load_metric()
    
    def _load_provider(self) -> VectorStoreProvider:
        """
        Load the vector store provider from environment.
        
        Returns:
            VectorStoreProvider enum value
            
        Raises:
            ValueError: If provider value is invalid
        """
        provider_str = os.getenv(self.ENV_VECTOR_STORE_PROVIDER, self.DEFAULT_PROVIDER.value)
        
        try:
            return VectorStoreProvider(provider_str.lower())
        except ValueError:
            raise ValueError(
                f"Invalid vector store provider: {provider_str}. "
                f"Valid options: {[p.value for p in VectorStoreProvider]}"
            )
    
    def _load_dimension(self) -> int:
        """
        Load the vector dimension from environment.
        
        Returns:
            Vector dimension as integer
            
        Raises:
            ValueError: If dimension is not a positive integer
        """
        dimension_str = os.getenv(self.ENV_VECTOR_STORE_DIMENSION, str(self.DEFAULT_DIMENSION))
        
        try:
            dimension = int(dimension_str)
            if dimension <= 0:
                raise ValueError("Dimension must be a positive integer")
            return dimension
        except ValueError as e:
            raise ValueError(f"Invalid vector dimension: {dimension_str}. Must be a positive integer.") from e
    
    def _load_metric(self) -> str:
        """
        Load the distance metric from environment.
        
        Returns:
            Distance metric string
            
        Note:
            Common metrics: cosine, euclidean, dotproduct
        """
        return os.getenv(self.ENV_VECTOR_STORE_METRIC, self.DEFAULT_METRIC)
    
    @property
    def provider(self) -> VectorStoreProvider:
        """
        Get the configured vector store provider.
        
        Returns:
            VectorStoreProvider enum value
        """
        return self._provider
    
    @property
    def dimension(self) -> int:
        """
        Get the configured vector dimension.
        
        Returns:
            Vector dimension as integer
        """
        return self._dimension
    
    @property
    def metric(self) -> str:
        """
        Get the configured distance metric.
        
        Returns:
            Distance metric string
        """
        return self._metric
    
    def is_pinecone(self) -> bool:
        """
        Check if Pinecone is configured as the provider.
        
        Returns:
            True if provider is Pinecone, False otherwise
        """
        return self._provider == VectorStoreProvider.PINECONE
    
    def is_qdrant(self) -> bool:
        """
        Check if Qdrant is configured as the provider.
        
        Returns:
            True if provider is Qdrant, False otherwise
        """
        return self._provider == VectorStoreProvider.QDRANT
    
    def is_memory(self) -> bool:
        """
        Check if in-memory storage is configured as the provider.
        
        Returns:
            True if provider is Memory, False otherwise
        """
        return self._provider == VectorStoreProvider.MEMORY
    
    def __repr__(self) -> str:
        """
        Get string representation of the configuration.
        
        Returns:
            String representation
        """
        return (
            f"VectorStoreConfig(provider={self._provider.value}, "
            f"dimension={self._dimension}, metric={self._metric})"
        )
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'provider': self._provider.value,
            'dimension': self._dimension,
            'metric': self._metric
        }


# Global configuration instance
# This follows the Singleton pattern for configuration
_config_instance: Optional[VectorStoreConfig] = None


def get_config() -> VectorStoreConfig:
    """
    Get the global vector store configuration instance.
    
    This function implements the Singleton pattern for configuration,
    ensuring that configuration is loaded only once and reused.
    
    Architecture Pattern: Singleton Pattern
    - Single source of truth for configuration
    - Lazy initialization
    - Thread-safe in practice (Python GIL)
    
    Returns:
        VectorStoreConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = VectorStoreConfig()
    return _config_instance


def reset_config() -> None:
    """
    Reset the global configuration instance.
    
    This is primarily useful for testing, allowing the configuration
    to be reloaded with different environment variables.
    
    Note:
        This should not be used in production code.
    """
    global _config_instance
    _config_instance = None
