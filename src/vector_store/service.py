"""
Service module - Main entry point for vector store operations.

This module provides the VectorStoreService class, which is the main entry
point for application code to interact with the vector store. It hides the
complexity of adapter selection and provides a simple, clean API.

Architecture Notes:
- Facade Pattern: Simplifies interface to complex subsystem
- Delegates to selected adapter
- Application code NEVER directly calls Pinecone or other adapters
- Follows Single Responsibility Principle

SOLID Principles Applied:
- Single Responsibility: Coordinates vector store operations
- Dependency Inversion: Depends on VectorStore abstraction
- Interface Segregation: Exposes only necessary methods
"""

import logging
from typing import List, Dict, Any, Optional
from .interface import VectorStore, VectorStoreError
from .schema import VectorRecord
from .factory import VectorStoreFactory, create_vector_store
from .config import get_config
from .validator import VectorStoreValidator, ValidationError

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Main service for vector store operations.
    
    This class is the primary entry point for application code to interact
    with the vector store. It implements the Facade Pattern to provide a
    simplified interface to the complex vector store subsystem.
    
    Architecture Pattern: Facade Pattern
    - Provides a simplified interface to the vector store subsystem
    - Hides complexity of adapter selection and configuration
    - Delegates operations to the selected adapter
    - Application code depends on this service, not on adapters directly
    
    SOLID Principles Applied:
    - Single Responsibility: Coordinates vector store operations
    - Open/Closed: Can be extended with new operations without modification
    - Liskov Substitution: Works with any VectorStore implementation
    - Interface Segregation: Exposes only necessary methods
    - Dependency Inversion: Depends on VectorStore abstraction, not concretions
    
    Security Note:
    - Application code should NEVER directly call Pinecone or other adapters
    - All vector store operations must go through this service
    - This ensures consistent validation, error handling, and logging
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the vector store service.
        
        Args:
            vector_store: Optional VectorStore instance. If None, creates one
                         using the factory based on configuration.
        """
        if vector_store is None:
            self.vector_store = create_vector_store()
        else:
            self.vector_store = vector_store
        
        self.config = get_config()
        self.validator = VectorStoreValidator(expected_dimension=self.config.dimension)
        
        logger.info(f"VectorStoreService initialized with provider: {self.config.provider.value}")
    
    def upsert(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Insert or update vector records.
        
        This method validates the records before upserting them to the vector
        store. Invalid records are rejected with clear error messages.
        
        Args:
            records: List of VectorRecord objects to upsert
            
        Returns:
            Dictionary with operation results:
            - success: bool
            - upserted_count: int
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info(f"Upserting {len(records)} records to vector store")
        
        # Validate records
        validation_result = self.validator.validate_records(records)
        
        if not validation_result['valid']:
            error_messages = [str(e) for e in validation_result['errors']]
            logger.error(f"Validation failed: {error_messages}")
            return {
                'success': False,
                'upserted_count': 0,
                'errors': error_messages
            }
        
        # Upsert to vector store
        try:
            result = self.vector_store.upsert(records)
            logger.info(f"Successfully upserted {result.get('upserted_count', 0)} records")
            return result
        except VectorStoreError as e:
            logger.error(f"Vector store upsert failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upsert: {e}")
            raise VectorStoreError(f"Upsert operation failed: {str(e)}") from e
    
    def query(self, vector: List[float], k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar vectors.
        
        This method validates the query vector before performing the search.
        
        Args:
            vector: Query vector to search for
            k: Number of results to return (default: 10)
            filters: Optional metadata filters
            
        Returns:
            List of dictionaries containing search results
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info(f"Querying vector store with k={k}")
        
        # Validate query vector
        try:
            self.validator.validate_query_vector(vector)
        except ValidationError as e:
            logger.error(f"Query vector validation failed: {e}")
            raise VectorStoreError(f"Invalid query vector: {str(e)}") from e
        
        # Query vector store
        try:
            results = self.vector_store.query(vector, k=k, filters=filters)
            logger.info(f"Query returned {len(results)} results")
            return results
        except VectorStoreError as e:
            logger.error(f"Vector store query failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise VectorStoreError(f"Query operation failed: {str(e)}") from e
    
    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vector records by their IDs.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            Dictionary with operation results:
            - success: bool
            - deleted_count: int
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info(f"Deleting {len(ids)} records from vector store")
        
        try:
            result = self.vector_store.delete(ids)
            logger.info(f"Successfully deleted {result.get('deleted_count', 0)} records")
            return result
        except VectorStoreError as e:
            logger.error(f"Vector store delete failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during delete: {e}")
            raise VectorStoreError(f"Delete operation failed: {str(e)}") from e
    
    def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        Fetch a single vector record by its ID.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info(f"Fetching record with ID: {id}")
        
        try:
            record = self.vector_store.fetch(id)
            if record:
                logger.info(f"Successfully fetched record: {id}")
            else:
                logger.warning(f"Record not found: {id}")
            return record
        except VectorStoreError as e:
            logger.error(f"Vector store fetch failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during fetch: {e}")
            raise VectorStoreError(f"Fetch operation failed: {str(e)}") from e
    
    def count(self) -> int:
        """
        Get the total number of vector records in the store.
        
        Returns:
            Total number of records
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info("Counting records in vector store")
        
        try:
            count = self.vector_store.count()
            logger.info(f"Vector store contains {count} records")
            return count
        except VectorStoreError as e:
            logger.error(f"Vector store count failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during count: {e}")
            raise VectorStoreError(f"Count operation failed: {str(e)}") from e
    
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the vector store.
        
        Returns:
            Dictionary with health status
            
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info("Checking vector store health")
        
        try:
            health_status = self.vector_store.health()
            logger.info(f"Vector store health: {health_status.get('status', 'unknown')}")
            return health_status
        except VectorStoreError as e:
            logger.error(f"Vector store health check failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            raise VectorStoreError(f"Health check failed: {str(e)}") from e
    
    def close(self) -> None:
        """
        Close the vector store connection and release resources.
        
        Raises:
            VectorStoreError: If the operation fails
        """
        logger.info("Closing vector store connection")
        
        try:
            self.vector_store.close()
            logger.info("Vector store connection closed successfully")
        except VectorStoreError as e:
            logger.error(f"Vector store close failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during close: {e}")
            raise VectorStoreError(f"Close operation failed: {str(e)}") from e
    
    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            Self for use in with statements
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Ensures the vector store connection is closed when exiting the context.
        """
        self.close()
