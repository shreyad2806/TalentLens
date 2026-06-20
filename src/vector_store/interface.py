"""
Interface module - Abstract VectorStore interface for adapter pattern.

This module defines the abstract VectorStore interface that all vector store
adapters must implement. This follows the Dependency Inversion Principle and
the Adapter Pattern, allowing the application to depend on abstractions rather
than concrete implementations.

Architecture Notes:
- Abstract Base Class (ABC) defines the contract
- All adapters (Pinecone, Qdrant, Memory) must implement this interface
- Enables easy switching between vector store providers
- Follows Interface Segregation Principle - focused, cohesive interface
- Follows Liskov Substitution Principle - adapters are substitutable
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .schema import VectorRecord


class VectorStore(ABC):
    """
    Abstract interface for vector store implementations.
    
    This class defines the contract that all vector store adapters must implement.
    It provides a unified API for vector operations regardless of the underlying
    storage technology (Pinecone, Qdrant, Memory, etc.).
    
    Architecture Pattern: Adapter Pattern + Strategy Pattern
    - Adapter Pattern: Adapts different vector store APIs to a common interface
    - Strategy Pattern: Different storage strategies are interchangeable
    - Dependency Inversion: High-level modules depend on this abstraction
    
    SOLID Principles Applied:
    - Single Responsibility: Defines only the vector store contract
    - Open/Closed: Open for extension (new adapters), closed for modification
    - Liskov Substitution: All adapters are substitutable for this interface
    - Interface Segregation: Focused interface with only necessary methods
    - Dependency Inversion: Application depends on this abstraction, not concretions
    """
    
    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Insert or update vector records in the store.
        
        This method performs an "upsert" operation - if a record with the same
        ID exists, it will be updated; otherwise, a new record will be inserted.
        
        Args:
            records: List of VectorRecord objects to upsert
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - upserted_count: int
            - errors: List of error messages (if any)
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def query(self, vector: List[float], k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar vectors.
        
        Performs a similarity search to find the k most similar vectors to the
        query vector. Optional filters can be applied to narrow results.
        
        Args:
            vector: Query vector to search for
            k: Number of results to return (default: 10)
            filters: Optional metadata filters (e.g., {'section': 'skills'})
            
        Returns:
            List of dictionaries containing:
            - id: str
            - score: float (similarity score)
            - record: VectorRecord
            - metadata: Dict[str, Any]
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vector records by their IDs.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - deleted_count: int
            - errors: List of error messages (if any)
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def delete_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Delete all vector records for a specific resume.
        
        This is a convenience method for deleting all chunks associated with
        a single resume.
        
        Args:
            resume_id: ID of the resume to delete
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - deleted_count: int
            - errors: List of error messages (if any)
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        Fetch a single vector record by its ID.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def fetch_resume(self, resume_id: str) -> List[VectorRecord]:
        """
        Fetch all vector records for a specific resume.
        
        This is a convenience method for retrieving all chunks associated with
        a single resume.
        
        Args:
            resume_id: ID of the resume to fetch
            
        Returns:
            List of VectorRecord objects for the resume
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of vector records in the store.
        
        Returns:
            Total number of records
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def clear(self) -> Dict[str, Any]:
        """
        Clear all vector records from the store.
        
        This is a destructive operation that cannot be undone.
        
        Returns:
            Dictionary with operation results including:
            - success: bool
            - cleared_count: int
            - errors: List of error messages (if any)
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the vector store.
        
        This method should verify connectivity and operational status of the
        underlying vector store implementation.
        
        Returns:
            Dictionary with health status including:
            - healthy: bool
            - status: str (e.g., 'healthy', 'degraded', 'unhealthy')
            - message: str (description of status)
            - latency_ms: float (optional, response time)
            
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the vector store connection and release resources.
        
        This method should be called when the vector store is no longer needed
        to properly release resources and close connections.
        
        Raises:
            NotImplementedError: If not implemented by adapter
            VectorStoreError: If operation fails
        """
        pass


class VectorStoreError(Exception):
    """
    Base exception for vector store operations.
    
    All vector store adapters should raise this exception (or subclasses)
    for errors that occur during vector store operations.
    
    Architecture Pattern: Custom Exception Hierarchy
    - Provides consistent error handling across all adapters
    - Allows application to catch vector store errors specifically
    - Can be extended with adapter-specific exceptions
    """
    
    def __init__(self, message: str, adapter_name: str = "unknown", original_error: Optional[Exception] = None):
        """
        Initialize the vector store error.
        
        Args:
            message: Error message
            adapter_name: Name of the adapter that raised the error
            original_error: Original exception if this wraps another error
        """
        self.message = message
        self.adapter_name = adapter_name
        self.original_error = original_error
        super().__init__(f"[{adapter_name}] {message}")
