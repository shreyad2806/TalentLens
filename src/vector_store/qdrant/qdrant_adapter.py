"""
Qdrant Adapter for Vector Store.

This module provides a production-ready Qdrant adapter implementing the vector
store interface with support for collection management, metadata filtering,
and health monitoring.

SOLID Principles Applied:
- Single Responsibility: Handles only Qdrant vector operations
- Open/Closed: Open for extension with new features
- Dependency Inversion: Depends on QdrantClient abstraction
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    SearchRequest,
    VectorParams,
    Distance,
    Batch,
)

from .schema import (
    QdrantCollectionConfig,
    QdrantPayload,
    QdrantFilter,
    QdrantHealthStatus,
    SearchResult,
    UpsertResult,
    CollectionInfo,
)

from .collection_manager import CollectionManager
from .health_check import HealthCheck
from ...config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """
    Production-ready Qdrant adapter for vector storage.
    
    This adapter provides a complete interface for vector storage operations
    including collection management, vector upsert, search with metadata filtering,
    and health monitoring.
    
    Capabilities:
        - create_collection(): Create a new collection
        - delete_collection(): Delete the collection
        - upsert_vectors(): Insert or update vectors
        - search(): Search for similar vectors
        - search_with_filters(): Search with metadata filters
        - batch_insert(): Batch insert vectors
        - count(): Get vector count
        - health_check(): Perform health check
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: str = "Cosine"
    ):
        """
        Initialize the Qdrant adapter.
        
        Args:
            url: Qdrant server URL (from QDRANT_URL env var if None)
            api_key: Qdrant API key (from QDRANT_API_KEY env var if None)
            collection_name: Collection name (from QDRANT_COLLECTION env var if None)
            vector_size: Vector dimension size (default: from config EMBEDDING_DIM)
            distance: Distance metric - Cosine, Euclid, or Dot (default: Cosine)
        """
        # Use configured embedding dimension if not provided
        if vector_size is None:
            vector_size = EMBEDDING_DIM
        # Get configuration from environment variables
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "talentlens_candidates")
        
        # Map distance string to enum
        from .schema import DistanceMetric
        distance_map = {
            "Cosine": DistanceMetric.COSINE,
            "Euclid": DistanceMetric.EUCLID,
            "Dot": DistanceMetric.DOT,
        }
        distance_enum = distance_map.get(distance, DistanceMetric.COSINE)
        
        # Create collection config
        self.config = QdrantCollectionConfig(
            collection_name=self.collection_name,
            vector_size=vector_size,
            distance=distance_enum
        )
        
        # Initialize Qdrant client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
        
        # Initialize collection manager and health check
        self.collection_manager = CollectionManager(self.client, self.config)
        self.health_checker = HealthCheck(self.client, self.config)
        
        logger.info(
            f"QdrantAdapter initialized - URL: {self.url}, "
            f"Collection: {self.collection_name}, "
            f"Vector Size: {vector_size}, "
            f"Distance: {distance}"
        )
    
    def create_collection(self) -> bool:
        """
        Create a new collection with the configured schema.
        
        Returns:
            True if collection created successfully, False otherwise
        """
        try:
            success = self.collection_manager.create_collection()
            if success:
                logger.info(f"Collection {self.collection_name} created successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if collection deleted successfully, False otherwise
        """
        try:
            success = self.collection_manager.delete_collection()
            if success:
                logger.info(f"Collection {self.collection_name} deleted successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> UpsertResult:
        """
        Insert or update vectors in the collection.
        
        Args:
            vectors: List of vector embeddings
            payloads: List of payload dictionaries
            ids: Optional list of point IDs (auto-generated if None)
        
        Returns:
            UpsertResult with operation details
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(vectors) != len(payloads):
                raise ValueError("Vectors and payloads must have the same length")
            
            # Generate IDs if not provided
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            # Create point structures
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Upsert points
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = UpsertResult(
                upserted_count=len(vectors),
                operation_id=str(operation_info.status),
                latency_ms=latency_ms
            )
            
            logger.info(
                f"Upserted {len(vectors)} vectors in {latency_ms:.2f}ms "
                f"(operation_id: {operation_info.status})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector embedding
            top_k: Number of results to return (default: 10)
            score_threshold: Minimum score threshold (optional)
        
        Returns:
            List of SearchResult objects
        """
        try:
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload_dict = result.payload
                payload = QdrantPayload(**payload_dict)
                
                search_result = SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=payload
                )
                results.append(search_result)
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            raise
    
    def search_with_filters(
        self,
        query_vector: List[float],
        filters: QdrantFilter,
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors with metadata filters.
        
        Args:
            query_vector: Query vector embedding
            filters: QdrantFilter object with filter conditions
            top_k: Number of results to return (default: 10)
            score_threshold: Minimum score threshold (optional)
        
        Returns:
            List of SearchResult objects
        """
        try:
            # Convert filters to Qdrant filter format
            qdrant_filter = filters.to_qdrant_filter()
            
            # Perform search with filters
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload_dict = result.payload
                payload = QdrantPayload(**payload_dict)
                
                search_result = SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=payload
                )
                results.append(search_result)
            
            logger.debug(f"Search with filters returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search with filters: {str(e)}")
            raise
    
    def batch_insert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> UpsertResult:
        """
        Batch insert vectors in chunks.
        
        Args:
            vectors: List of vector embeddings
            payloads: List of payload dictionaries
            batch_size: Number of vectors per batch (default: 100)
        
        Returns:
            UpsertResult with operation details
        """
        start_time = time.time()
        total_upserted = 0
        
        try:
            # Validate inputs
            if len(vectors) != len(payloads):
                raise ValueError("Vectors and payloads must have the same length")
            
            # Generate IDs if not provided
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                result = self.upsert_vectors(batch_vectors, batch_payloads, batch_ids)
                total_upserted += result.upserted_count
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = UpsertResult(
                upserted_count=total_upserted,
                operation_id="batch_insert",
                latency_ms=latency_ms
            )
            
            logger.info(
                f"Batch insert complete: {total_upserted} vectors in {latency_ms:.2f}ms "
                f"(batch_size: {batch_size})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to batch insert: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Get the number of vectors in the collection.
        
        Returns:
            Number of vectors in the collection
        """
        try:
            count = self.health_checker.check_vector_count()
            logger.debug(f"Vector count: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get vector count: {str(e)}")
            return 0
    
    def health_check(self) -> QdrantHealthStatus:
        """
        Perform comprehensive health check.
        
        Returns:
            QdrantHealthStatus with health information
        """
        try:
            health_status = self.health_checker.perform_health_check()
            return health_status
        except Exception as e:
            logger.error(f"Failed to perform health check: {str(e)}")
            return QdrantHealthStatus(
                status="unhealthy",
                connection_healthy=False,
                collection_exists=False,
                vector_count=0,
                error_message=str(e),
                latency_ms=None
            )
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """
        Get detailed information about the collection.
        
        Returns:
            CollectionInfo if collection exists, None otherwise
        """
        try:
            return self.collection_manager.get_collection_info()
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return None
    
    def delete_points(self, point_ids: List[str]) -> bool:
        """
        Delete specific points from the collection.
        
        Args:
            point_ids: List of point IDs to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} points from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all vectors from the collection without deleting it.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            return self.collection_manager.clear_collection()
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
    
    def print_health_status(self) -> None:
        """Print formatted health status to console."""
        health_status = self.health_check()
        self.health_checker.print_health_status(health_status)
    
    def is_healthy(self) -> bool:
        """
        Quick health check.
        
        Returns:
            True if system is healthy, False otherwise
        """
        return self.health_checker.is_healthy()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the adapter.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = self.health_checker.get_diagnostics()
        diagnostics.update({
            "url": self.url,
            "api_key_provided": self.api_key is not None,
            "config": {
                "collection_name": self.collection_name,
                "vector_size": self.config.vector_size,
                "distance": self.config.distance.value
            }
        })
        return diagnostics
