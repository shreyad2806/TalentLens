"""
Qdrant Adapter - Qdrant vector database implementation.

This module provides the QdrantAdapter class, which implements the
VectorStore interface using Qdrant's vector database service.

Architecture Notes:
- Adapter Pattern: Implements VectorStore interface
- Handles Qdrant-specific error handling and retries
- Converts between Qdrant format and internal VectorRecord schema
- Implements retry logic for rate limits and network failures
- No Qdrant SDK objects leak outside this adapter

SOLID Principles Applied:
- Single Responsibility: Handles only Qdrant integration
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


class QdrantAdapter(VectorStore):
    """
    Qdrant vector store adapter implementation.
    
    This class implements the VectorStore interface using Qdrant's vector
    database service. It handles Qdrant-specific operations, error handling,
    retry logic, and schema conversion.
    
    Architecture Pattern: Adapter Pattern
    - Implements VectorStore interface
    - Adapts Qdrant SDK to vector store contract
    - Handles Qdrant-specific error handling and retries
    - No Qdrant objects leak outside this adapter
    
    Environment Variables:
    - QDRANT_HOST: Qdrant host URL (default: localhost)
    - QDRANT_PORT: Qdrant port (default: 6333)
    - QDRANT_COLLECTION: Qdrant collection name (required)
    - QDRANT_API_KEY: Optional Qdrant API key
    
    Configuration:
    - batch_size: Number of vectors to upsert in a single batch (default: 100)
    - max_retries: Maximum number of retry attempts (default: 3)
    - retry_delay: Initial retry delay in seconds (default: 1.0)
    - timeout: Request timeout in seconds (default: 30)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None, batch_size: int = 100):
        """
        Initialize the Qdrant adapter.
        
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
        self._timeout = 30  # Request timeout in seconds
        self._closed = False
        self._connected = False
        self._client = None
        self._collection_name = None
        
        # Read environment variables
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = os.getenv("QDRANT_PORT", "6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        # Validate required environment variables
        if not self.collection_name:
            raise VectorStoreError(
                "QDRANT_COLLECTION environment variable is required",
                adapter_name="QdrantAdapter"
            )
        
        logger.info(
            f"QdrantAdapter initialized with host: {self.host}:{self.port}, "
            f"collection: {self.collection_name}, batch_size: {self.batch_size}"
        )
    
    def connect(self) -> None:
        """
        Establish connection to Qdrant.
        
        This method initializes the Qdrant client and creates the collection
        if it doesn't exist. It should be called before any other operations.
        If not called explicitly, it will be called automatically on the first operation.
        
        Raises:
            VectorStoreError: If connection fails
        """
        if self._connected:
            logger.info("QdrantAdapter already connected")
            return
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            # Initialize Qdrant client
            self._client = QdrantClient(
                host=self.host,
                port=int(self.port),
                api_key=self.api_key,
                timeout=self._timeout
            )
            
            # Check if collection exists, create if not
            collections = self._client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self._create_collection()
            
            self._connected = True
            logger.info(f"QdrantAdapter connected to collection: {self.collection_name}")
            
        except ImportError as e:
            raise VectorStoreError(
                "qdrant-client package is required. Install with: pip install qdrant-client",
                adapter_name="QdrantAdapter"
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to Qdrant: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def _create_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't exist.
        
        This method creates a new collection with the appropriate configuration
        based on the configured vector dimension.
        """
        from qdrant_client.models import Distance, VectorParams
        
        dimension = self.config.dimension if self.config else 1024
        
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"Created collection {self.collection_name} with dimension {dimension}")
    
    def _ensure_connected(self) -> None:
        """
        Ensure connection to Qdrant is established.
        
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
                        adapter_name="QdrantAdapter",
                        original_error=e
                    ) from e
        
        # All retries exhausted
        raise VectorStoreError(
            f"Operation failed after {self._max_retries} retries: {str(last_error)}",
            adapter_name="QdrantAdapter",
            original_error=last_error
        )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Retryable errors include:
        - Rate limit errors
        - Network errors (timeouts, connection failures)
        - Temporary service unavailability
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable, False otherwise
        """
        # Check for timeout errors
        error_str = str(error).lower()
        retryable_keywords = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'service unavailable',
            'rate limit',
            'too many requests',
            'unavailable'
        ]
        
        return any(keyword in error_str for keyword in retryable_keywords)
    
    def _convert_to_qdrant_format(self, record: VectorRecord):
        """
        Convert internal VectorRecord to Qdrant PointStruct.
        
        This method converts the internal VectorRecord schema to the format
        expected by Qdrant's upsert operation.
        
        Args:
            record: VectorRecord to convert
            
        Returns:
            Qdrant PointStruct object
        """
        from qdrant_client.models import PointStruct
        
        # Build payload with all required fields
        payload = {
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
                    payload[field] = record.metadata[field]
            
            # Add any remaining metadata
            for key, value in record.metadata.items():
                if key not in payload:
                    payload[key] = value
        
        return PointStruct(
            id=record.id,
            vector=record.vector,
            payload=payload
        )
    
    def _convert_from_qdrant_format(self, qdrant_point) -> VectorRecord:
        """
        Convert Qdrant response to internal VectorRecord schema.
        
        This method converts Qdrant's response format to the internal
        VectorRecord schema, ensuring no Qdrant objects leak outside.
        
        Args:
            qdrant_point: Qdrant point object
            
        Returns:
            VectorRecord object
        """
        payload = qdrant_point.payload
        
        # Extract metadata fields
        record_metadata = {}
        for field in ["experience", "location", "role", "education"]:
            if field in payload:
                record_metadata[field] = payload[field]
        
        # Add any additional metadata
        for key, value in payload.items():
            if key not in ["resume_id", "chunk_id", "candidate_name", "section", "created_at"]:
                if key not in record_metadata:
                    record_metadata[key] = value
        
        return VectorRecord(
            id=str(qdrant_point.id),
            resume_id=payload.get("resume_id", ""),
            chunk_id=payload.get("chunk_id", ""),
            candidate_name=payload.get("candidate_name", ""),
            section=payload.get("section", ""),
            vector=qdrant_point.vector,
            metadata=record_metadata,
            created_at=payload.get("created_at", "")
        )
    
    def upsert(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Insert or update vector records in Qdrant with batch processing.
        
        This method converts VectorRecord objects to Qdrant format and
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
            raise VectorStoreError("Cannot upsert: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        upserted_count = 0
        batch_count = 0
        errors = []
        
        # Validate dimensions
        dimension = self.config.dimension if self.config else 1024
        for record in records:
            if len(record.vector) != dimension:
                errors.append(
                    f"Dimension mismatch for record {record.id}: "
                    f"expected {dimension}, got {len(record.vector)}"
                )
        
        if errors:
            return {
                "success": False,
                "upserted_count": 0,
                "batch_count": 0,
                "latency_seconds": time.time() - start_time,
                "errors": errors
            }
        
        def _upsert_batch(batch: List):
            """Upsert a single batch of vectors."""
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        try:
            # Process records in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                # Convert batch to Qdrant format
                qdrant_points = [
                    self._convert_to_qdrant_format(record)
                    for record in batch
                ]
                
                # Upsert batch with retry logic
                self._retry_with_backoff(_upsert_batch, qdrant_points)
                
                upserted_count += len(batch)
                batch_count += 1
            
            latency = time.time() - start_time
            logger.info(
                f"Qdrant upsert completed: {upserted_count} vectors in "
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
            logger.error(f"Qdrant upsert failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Upsert operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def query(self, vector: List[float], k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query Qdrant for similar vectors.
        
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
            raise VectorStoreError("Cannot query: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        def _query():
            """Execute query with Qdrant."""
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            query_filter = None
            if filters:
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
                query_filter = Filter(must=conditions)
            
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
                query_filter=query_filter
            )
            return results
        
        try:
            qdrant_results = self._retry_with_backoff(_query)
            latency = time.time() - start_time
            
            # Convert Qdrant results to internal format
            converted_results = []
            for scored_point in qdrant_results:
                record = self._convert_from_qdrant_format(score_point)
                
                converted_results.append({
                    "id": str(score_point.id),
                    "score": scored_point.score,
                    "record": record,
                    "metadata": record.metadata
                })
            
            logger.info(
                f"Qdrant query completed: {len(converted_results)} results, "
                f"latency: {latency:.3f}s"
            )
            
            return converted_results
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Qdrant query failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Query operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        Fetch a single vector record by its ID from Qdrant.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        def _fetch():
            """Execute fetch with Qdrant."""
            from qdrant_client.models import PointIdsList
            
            results = self._client.retrieve(
                collection_name=self.collection_name,
                ids=PointIdsList(points=[id])
            )
            return results
        
        try:
            qdrant_results = self._retry_with_backoff(_fetch)
            
            if not qdrant_results:
                logger.warning(f"Record not found: {id}")
                return None
            
            # Convert Qdrant result to VectorRecord
            record = self._convert_from_qdrant_format(qdrant_results[0])
            
            logger.info(f"Qdrant fetch completed: {id}")
            return record
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Qdrant fetch failed: {str(e)}")
            raise VectorStoreError(
                f"Fetch operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def fetch_resume(self, resume_id: str) -> List[VectorRecord]:
        """
        Fetch all vector records for a specific resume from Qdrant.
        
        This method queries for all records with the given resume_id
        using scroll API with filtering.
        
        Args:
            resume_id: ID of the resume to fetch
            
        Returns:
            List of VectorRecord objects for the resume
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch resume: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        def _fetch_resume():
            """Execute fetch resume with Qdrant."""
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter for resume_id
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="resume_id",
                        match=MatchValue(value=resume_id)
                    )
                ]
            )
            
            # Use scroll to get all matching records
            results = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=10000  # Large limit to get all records
            )
            
            return results[0]  # scroll returns (points, next_page_offset)
        
        try:
            qdrant_results = self._retry_with_backoff(_fetch_resume)
            
            # Convert Qdrant results to VectorRecord objects
            records = []
            for point in qdrant_results:
                record = self._convert_from_qdrant_format(point)
                records.append(record)
            
            logger.info(f"Qdrant fetch_resume completed: {len(records)} records for resume {resume_id}")
            return records
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Qdrant fetch_resume failed: {str(e)}")
            raise VectorStoreError(
                f"Fetch resume operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vector records by their IDs from Qdrant.
        
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
            raise VectorStoreError("Cannot delete: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        def _delete():
            """Execute delete with Qdrant."""
            from qdrant_client.models import PointIdsList
            
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids)
            )
        
        try:
            self._retry_with_backoff(_delete)
            latency = time.time() - start_time
            
            logger.info(
                f"Qdrant delete completed: {len(ids)} records, "
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
            logger.error(f"Qdrant delete failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Delete operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def delete_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Delete all vector records for a specific resume from Qdrant.
        
        This method uses Qdrant's delete by filter capability.
        
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
            raise VectorStoreError("Cannot delete resume: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        def _delete_resume():
            """Execute delete resume with Qdrant."""
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter for resume_id
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="resume_id",
                        match=MatchValue(value=resume_id)
                    )
                ]
            )
            
            # Delete by filter
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter
            )
        
        try:
            self._retry_with_backoff(_delete_resume)
            latency = time.time() - start_time
            
            logger.info(
                f"Qdrant delete_resume completed for resume {resume_id}, "
                f"latency: {latency:.3f}s"
            )
            
            return {
                "success": True,
                "deleted_count": 0,  # Qdrant doesn't return count for delete by filter
                "latency_seconds": latency,
                "errors": []
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Qdrant delete_resume failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Delete resume operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def count(self) -> int:
        """
        Get the total number of vector records in Qdrant.
        
        Returns:
            Total number of records
            
        Raises:
            VectorStoreError: If operation fails
        """
        if self._closed:
            raise VectorStoreError("Cannot count: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        def _count():
            """Execute count with Qdrant."""
            collection_info = self._client.get_collection(self.collection_name)
            return collection_info.points_count
        
        try:
            count = self._retry_with_backoff(_count)
            logger.info(f"Qdrant count completed: {count} records")
            return count
            
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Qdrant count failed: {str(e)}")
            raise VectorStoreError(
                f"Count operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def clear(self) -> Dict[str, Any]:
        """
        Clear all vector records from Qdrant.
        
        This is a destructive operation that cannot be undone.
        
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
            raise VectorStoreError("Cannot clear: adapter is closed", adapter_name="QdrantAdapter")
        
        self._ensure_connected()
        
        start_time = time.time()
        
        try:
            # Get current count
            collection_info = self._client.get_collection(self.collection_name)
            total_count = collection_info.points_count
            
            if total_count == 0:
                logger.info("Qdrant clear: no records to clear")
                return {
                    "success": True,
                    "cleared_count": 0,
                    "latency_seconds": time.time() - start_time,
                    "errors": []
                }
            
            # Delete all points
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=self._client.filter(must=[])
            )
            
            latency = time.time() - start_time
            logger.info(f"Qdrant clear completed: {total_count} records, latency: {latency:.3f}s")
            
            return {
                "success": True,
                "cleared_count": total_count,
                "latency_seconds": latency,
                "errors": []
            }
            
        except VectorStoreError:
            raise
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Qdrant clear failed after {latency:.3f}s: {str(e)}")
            raise VectorStoreError(
                f"Clear operation failed: {str(e)}",
                adapter_name="QdrantAdapter",
                original_error=e
            ) from e
    
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the Qdrant connection.
        
        Returns:
            Dictionary with health status including:
            - healthy: bool
            - status: str
            - message: str
            - adapter: str
            - collection_name: str
            - record_count: int
            - latency_ms: float
        """
        start_time = time.time()
        
        try:
            self._ensure_connected()
            
            # Try to get collection info to verify connection
            collection_info = self._client.get_collection(self.collection_name)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "healthy": True,
                "status": "healthy",
                "message": "Qdrant connection is operational",
                "adapter": "QdrantAdapter",
                "collection_name": self.collection_name,
                "record_count": collection_info.points_count,
                "latency_ms": latency
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": f"Qdrant connection failed: {str(e)}",
                "adapter": "QdrantAdapter",
                "collection_name": self.collection_name,
                "record_count": 0,
                "latency_ms": latency
            }
    
    def close(self) -> None:
        """
        Close the Qdrant connection and release resources.
        
        This method marks the adapter as closed and prevents further operations.
        """
        self._closed = True
        self._connected = False
        self._client = None
        logger.info("QdrantAdapter closed")
