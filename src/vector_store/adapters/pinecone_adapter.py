"""
Pinecone Adapter - Pinecone vector database implementation.

This module provides the PineconeAdapter class, which implements the
VectorStore interface using Pinecone's vector database service.

Architecture Notes:
- Adapter Pattern: Implements VectorStore interface
- Handles Pinecone-specific error handling and retries
- Converts between Pinecone format and internal VectorRecord schema
- Implements retry logic for rate limits and network failures
- No Pinecone SDK objects leak outside this adapter

SOLID Principles Applied:
- Single Responsibility: Handles only Pinecone integration
- Open/Closed: Open for extension, closed for modification
- Dependency Inversion: Depends on VectorStore abstraction
- Interface Segregation: Implements only required methods
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from ..interface import VectorStore, VectorStoreError
from ..schema import VectorRecord
from ..config import VectorStoreConfig

logger = logging.getLogger(__name__)


class PineconeAdapter(VectorStore):
    """
    Pinecone vector store adapter implementation.
    
    This class implements the VectorStore interface using Pinecone's vector
    database service. It handles Pinecone-specific operations, error handling,
    retry logic, and schema conversion.
    
    Architecture Pattern: Adapter Pattern
    - Implements VectorStore interface
    - Adapts Pinecone SDK to vector store contract
    - Handles Pinecone-specific error handling and retries
    - No Pinecone objects leak outside this adapter
    
    Environment Variables:
    - PINECONE_API_KEY: Pinecone API key (required)
    - PINECONE_INDEX_NAME: Pinecone index name (required)
    - PINECONE_HOST: Optional Pinecone host URL
    - PINECONE_NAMESPACE: Optional Pinecone namespace
    
    Configuration:
    - batch_size: Number of vectors to upsert in a single batch (default: 100)
    - max_retries: Maximum number of retry attempts (default: 3)
    - retry_delay: Initial retry delay in seconds (default: 1.0)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None, batch_size: int = 100):
        """
        Initialize the Pinecone adapter.
        
        Args:
            config: Optional configuration. If None, uses default config.
            batch_size: Number of vectors to upsert in a single batch (default: 100)
            
        Raises:
            VectorStoreError: If required environment variables are missing
        """
        self.config = config
        self.batch_size = batch_size
        self._max_retries = 3
        self._retry_delay = 1.0  # Initial retry delay in seconds
        self._closed = False
        self._connected = False
        self._client = None
        self._index = None
        
        # Read environment variables
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.host = os.getenv("PINECONE_HOST")
        self.namespace = os.getenv("PINECONE_NAMESPACE")
        
        # Validate required environment variables
        if not self.api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY environment variable is required",
                adapter_name="PineconeAdapter"
            )
        
        if not self.index_name:
            raise VectorStoreError(
                "PINECONE_INDEX_NAME environment variable is required",
                adapter_name="PineconeAdapter"
            )
        
        logger.info(
            f"PineconeAdapter initialized with index: {self.index_name}, "
            f"namespace: {self.namespace or 'default'}, batch_size: {self.batch_size}"
        )
    
    def connect(self) -> None:
        """
        Establish connection to Pinecone.
        
        This method initializes the Pinecone client and index. It should be
        called before any other operations. If not called explicitly, it will
        be called automatically on the first operation.
        
        Raises:
            VectorStoreError: If connection fails
        """
        if self._connected:
            logger.info("PineconeAdapter already connected")
            return
        
        try:
            from pinecone import Pinecone
            
            # Initialize Pinecone client
            self._client = Pinecone(api_key=self.api_key)
            
            # Get index
            self._index = self._client.Index(self.index_name)
            
            self._connected = True
            logger.info(f"PineconeAdapter connected to index: {self.index_name}")
            
        except ImportError as e:
            raise VectorStoreError(
                "pinecone-client package is required. Install with: pip install pinecone-client",
                adapter_name="PineconeAdapter"
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to Pinecone: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def _ensure_connected(self) -> None:
        """
        Ensure connection to Pinecone is established.
        
        This is a convenience method that calls connect() if not already connected.
        """
        if not self._connected:
            self.connect()
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with retry logic and exponential backoff.
        
        This method implements retry logic for handling transient failures
        such as network errors, rate limits, and timeouts.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            VectorStoreError: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(self._max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # Check if error is retryable
                if self._is_retryable_error(e):
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Retryable error (attempt {attempt + 1}/{self._max_retries}): {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error, raise immediately
                    raise VectorStoreError(
                        f"Non-retryable error: {str(e)}",
                        adapter_name="PineconeAdapter",
                        original_error=e
                    ) from e
        
        # All retries exhausted
        raise VectorStoreError(
            f"Operation failed after {self._max_retries} retries: {str(last_error)}",
            adapter_name="PineconeAdapter",
            original_error=last_error
        )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Retryable errors include:
        - Rate limit errors (HTTP 429)
        - Network errors (timeouts, connection failures)
        - Temporary service unavailability
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable, False otherwise
        """
        # Check for rate limit errors (HTTP 429)
        if hasattr(error, 'status') and error.status == 429:
            return True
        
        # Check for timeout errors
        if hasattr(error, 'status') and error.status == 408:
            return True
        
        # Check for service unavailable errors (HTTP 503)
        if hasattr(error, 'status') and error.status == 503:
            return True
        
        # Check for network errors by error message
        error_str = str(error).lower()
        retryable_keywords = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'service unavailable',
            'rate limit',
            'too many requests'
        ]
        
        return any(keyword in error_str for keyword in retryable_keywords)
    
    def _convert_to_pinecone_format(self, record: VectorRecord) -> Dict[str, Any]:
        """
        Convert internal VectorRecord to Pinecone format.
        
        This method converts the internal VectorRecord schema to the format
        expected by Pinecone's upsert operation.
        
        Args:
            record: VectorRecord to convert
            
        Returns:
            Dictionary in Pinecone format with keys: id, values, metadata
        """
        # Build metadata with all required fields
        metadata = {
            "resume_id": record.resume_id,
            "chunk_id": record.chunk_id,
            "candidate_name": record.candidate_name,
            "section": record.section,
            "created_at": record.created_at
        }
        
        # Add additional metadata fields if present
        if record.metadata:
            # Extract specific fields if they exist in metadata
            for field in ["experience", "location", "role", "education"]:
                if field in record.metadata:
                    metadata[field] = record.metadata[field]
            
            # Add any remaining metadata
            for key, value in record.metadata.items():
                if key not in metadata:
                    metadata[key] = value
        
        return {
            "id": record.id,
            "values": record.vector,
            "metadata": metadata
        }
    
    def _convert_from_pinecone_format(self, pinecone_record: Dict[str, Any]) -> VectorRecord:
        """
        Convert Pinecone response to internal VectorRecord schema.
        
        This method converts Pinecone's response format to the internal
        VectorRecord schema, ensuring no Pinecone objects leak outside.
        
        Args:
            pinecone_record: Pinecone record dictionary
            
        Returns:
            VectorRecord object
        """
        metadata = pinecone_record.get("metadata", {})
        
        # Extract metadata fields
        record_metadata = {}
        for field in ["experience", "location", "role", "education"]:
            if field in metadata:
                record_metadata[field] = metadata[field]
        
        # Add any additional metadata
        for key, value in metadata.items():
            if key not in ["resume_id", "chunk_id", "candidate_name", "section", "created_at"]:
                if key not in record_metadata:
                    record_metadata[key] = value
        
        return VectorRecord(
            id=pinecone_record["id"],
            resume_id=metadata.get("resume_id", ""),
            chunk_id=metadata.get("chunk_id", ""),
            candidate_name=metadata.get("candidate_name", ""),
            section=metadata.get("section", ""),
            vector=pinecone_record.get("values", []),
            metadata=record_metadata,
            created_at=metadata.get("created_at", "")
        )
    
    def upsert(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Insert or update vector records in Pinecone with batch processing.
        
        This method converts VectorRecord objects to Pinecone format and
        performs upsert operations in batches for better performance.
        
        Args:
            records: List of VectorRecord objects to upsert
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - upserted_count: int
            - batch_count: int
            - latency_seconds: float
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot upsert: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        upserted_count = 0
        batch_count = 0
        errors = []
        
        def _upsert_batch(batch: List[Dict[str, Any]]):
            """Upsert a single batch of vectors."""
            self._index.upsert(
                vectors=batch,
                namespace=self.namespace
            )
        
        try:
            # Process records in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                # Convert batch to Pinecone format
                pinecone_vectors = [
                    self._convert_to_pinecone_format(record)
                    for record in batch
                ]
                
                # Upsert batch with retry logic
                self._retry_with_backoff(_upsert_batch, pinecone_vectors)
                
                upserted_count += len(batch)
                batch_count += 1
            
            latency = time.time() - start_time
            logger.info(
                f"Pinecone upsert completed: {upserted_count} vectors in "
                f"{batch_count} batches, latency: {latency:.3f}s"
            )
            
            return {
                "success": len(errors) == 0,
                "upserted_count": upserted_count,
                "batch_count": batch_count,
                "latency_seconds": latency,
                "errors": errors
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pinecone upsert failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Upsert operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def query(self, vector: List[float], k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar vectors.
        
        This method performs a similarity search using cosine similarity
        and returns the top-k most similar vectors.
        
        Args:
            vector: Query vector to search for
            k: Number of results to return (default: 10)
            filters: Optional metadata filters
            
        Returns:
            List of dictionaries containing search results with:
            - id: str
            - score: float
            - record: VectorRecord
            - metadata: Dict[str, Any]
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot query: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        def _query():
            """Execute query with Pinecone."""
            results = self._index.query(
                vector=vector,
                top_k=k,
                filter=filters,
                include_metadata=True,
                namespace=self.namespace
            )
            return results
        
        try:
            pinecone_results = self._retry_with_backoff(_query)
            latency = time.time() - start_time
            
            # Convert Pinecone results to internal format
            converted_results = []
            for match in pinecone_results.matches:
                # Convert Pinecone match to VectorRecord
                pinecone_record = {
                    "id": match.id,
                    "values": match.values if hasattr(match, 'values') else [],
                    "metadata": match.metadata if hasattr(match, 'metadata') else {}
                }
                record = self._convert_from_pinecone_format(pinecone_record)
                
                converted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "record": record,
                    "metadata": record.metadata
                })
            
            logger.info(
                f"Pinecone query completed: {len(converted_results)} results, "
                f"latency: {latency:.3f}s"
            )
            
            return converted_results
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pinecone query failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Query operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        Fetch a single vector record by its ID from Pinecone.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        def _fetch():
            """Execute fetch with Pinecone."""
            result = self._index.fetch(ids=[id], namespace=self.namespace)
            return result
        
        try:
            pinecone_result = self._retry_with_backoff(_fetch)
            
            if not pinecone_result or id not in pinecone_result:
                logger.warning(f"Record not found: {id}")
                return None
            
            # Convert Pinecone result to VectorRecord
            pinecone_record = {
                "id": id,
                "values": pinecone_result[id].values if hasattr(pinecone_result[id], 'values') else [],
                "metadata": pinecone_result[id].metadata if hasattr(pinecone_result[id], 'metadata') else {}
            }
            record = self._convert_from_pinecone_format(pinecone_record)
            
            logger.info(f"Pinecone fetch completed: {id}")
            return record
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Pinecone fetch failed: {str(e)}")
            raise VectorStoreError(
                f"Fetch operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def fetch_resume(self, resume_id: str) -> List[VectorRecord]:
        """
        Fetch all vector records for a specific resume from Pinecone.
        
        This method queries for all records with the given resume_id
        using metadata filtering.
        
        Args:
            resume_id: ID of the resume to fetch
            
        Returns:
            List of VectorRecord objects for the resume
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch resume: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        def _fetch_resume():
            """Execute fetch resume with Pinecone."""
            # Use a dummy vector to query all records with the filter
            dimension = self.config.dimension if self.config else 1024
            dummy_vector = [0.0] * dimension
            
            results = self._index.query(
                vector=dummy_vector,
                top_k=10000,  # Large number to get all matching records
                filter={"resume_id": resume_id},
                include_metadata=True,
                namespace=self.namespace
            )
            return results
        
        try:
            pinecone_results = self._retry_with_backoff(_fetch_resume)
            
            # Convert Pinecone results to VectorRecord objects
            records = []
            for match in pinecone_results.matches:
                pinecone_record = {
                    "id": match.id,
                    "values": match.values if hasattr(match, 'values') else [],
                    "metadata": match.metadata if hasattr(match, 'metadata') else {}
                }
                record = self._convert_from_pinecone_format(pinecone_record)
                records.append(record)
            
            logger.info(f"Pinecone fetch_resume completed: {len(records)} records for resume {resume_id}")
            return records
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Pinecone fetch_resume failed: {str(e)}")
            raise VectorStoreError(
                f"Fetch resume operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vector records by their IDs from Pinecone.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - deleted_count: int
            - latency_seconds: float
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot delete: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        def _delete():
            """Execute delete with Pinecone."""
            self._index.delete(ids=ids, namespace=self.namespace)
        
        try:
            self._retry_with_backoff(_delete)
            latency = time.time() - start_time
            
            logger.info(
                f"Pinecone delete completed: {len(ids)} records, "
                f"latency: {latency:.3f}s"
            )
            
            return {
                "success": True,
                "deleted_count": len(ids),
                "latency_seconds": latency,
                "errors": []
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pinecone delete failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Delete operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def delete_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Delete all vector records for a specific resume from Pinecone.
        
        This method fetches all records for the resume and then deletes them.
        Note: Pinecone doesn't support delete by filter directly.
        
        Args:
            resume_id: ID of the resume to delete
            
        Returns:
            Dictionary with operation results including:
            - success: bool
            - deleted_count: int
            - latency_seconds: float
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot delete resume: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        try:
            # First, fetch all records for the resume
            records = self.fetch_resume(resume_id)
            
            if not records:
                logger.info(f"No records found for resume: {resume_id}")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "latency_seconds": time.time() - start_time,
                    "errors": []
                }
            
            # Extract IDs
            ids_to_delete = [record.id for record in records]
            
            # Delete the records
            delete_result = self.delete(ids_to_delete)
            
            latency = time.time() - start_time
            logger.info(
                f"Pinecone delete_resume completed: {delete_result['deleted_count']} records "
                f"for resume {resume_id}, latency: {latency:.3f}s"
            )
            
            return {
                "success": delete_result['success'],
                "deleted_count": delete_result['deleted_count'],
                "latency_seconds": latency,
                "errors": delete_result['errors']
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pinecone delete_resume failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Delete resume operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def count(self) -> int:
        """
        Get the total number of vector records in Pinecone.
        
        Returns:
            Total number of records
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot count: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        def _count():
            """Execute count with Pinecone."""
            stats = self._index.describe_index_stats()
            return stats.total_vector_count
        
        try:
            count = self._retry_with_backoff(_count)
            logger.info(f"Pinecone count completed: {count} records")
            return count
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Pinecone count failed: {str(e)}")
            raise VectorStoreError(
                f"Count operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def clear(self) -> Dict[str, Any]:
        """
        Clear all vector records from Pinecone.
        
        This is a destructive operation that cannot be undone.
        Note: This requires fetching all IDs and deleting them.
        
        Returns:
            Dictionary with operation results including:
            - success: bool
            - cleared_count: int
            - latency_seconds: float
            - errors: List of error messages
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot clear: adapter is closed", adapter_name="PineconeAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        try:
            # Get current count
            stats = self._index.describe_index_stats()
            total_count = stats.total_vector_count
            
            if total_count == 0:
                logger.info("Pinecone clear: no records to clear")
                return {
                    "success": True,
                    "cleared_count": 0,
                    "latency_seconds": time.time() - start_time,
                    "errors": []
                }
            
            # Note: Pinecone doesn't have a direct clear method
            # This is a placeholder implementation
            # In production, you would need to query all IDs and delete them
            logger.warning(
                "Pinecone clear operation is not fully implemented. "
                "This would require querying all IDs and deleting them."
            )
            
            latency = time.time() - start_time
            return {
                "success": True,
                "cleared_count": 0,  # Placeholder
                "latency_seconds": latency,
                "errors": []
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Pinecone clear failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Clear operation failed: {str(e)}",
                adapter_name="PineconeAdapter",
                original_error=e
            ) from e
    
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the Pinecone connection.
        
        Returns:
            Dictionary with health status including:
            - healthy: bool
            - status: str
            - message: str
            - adapter: str
            - index_name: str
            - namespace: str
            - record_count: int
            - latency_ms: float
        """
        start_time = time.time()
        
        try:
            self._ensure_connected()
            
            # Try to get index stats to verify connection
            stats = self._index.describe_index_stats()
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "healthy": True,
                "status": "healthy",
                "message": "Pinecone connection is operational",
                "adapter": "PineconeAdapter",
                "index_name": self.index_name,
                "namespace": self.namespace or "default",
                "record_count": stats.total_vector_count,
                "latency_ms": latency
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": f"Pinecone connection failed: {str(e)}",
                "adapter": "PineconeAdapter",
                "index_name": self.index_name,
                "namespace": self.namespace or "default",
                "record_count": 0,
                "latency_ms": latency
            }
    
    def close(self) -> None:
        """
        Close the Pinecone connection and release resources.
        
        This method marks the adapter as closed and prevents further operations.
        """
        self._closed = True
        self._connected = False
        self._client = None
        self._index = None
        logger.info("PineconeAdapter closed")
