"""
Health Check Module for Qdrant Vector Store.

This module provides health monitoring and validation for the Qdrant adapter,
including connection checks, collection validation, and vector count verification.

SOLID Principles Applied:
- Single Responsibility: Handles only health monitoring
- Open/Closed: Open for new health checks
"""

import logging
import time
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient

from .schema import (
    QdrantHealthStatus,
    HealthStatus,
    QdrantCollectionConfig,
)

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Health checker for Qdrant adapter.
    
    This class provides comprehensive health checks for the Qdrant instance,
    including connection validation, collection existence verification,
    and vector count monitoring.
    """
    
    def __init__(
        self,
        client: QdrantClient,
        config: Optional[QdrantCollectionConfig] = None
    ):
        """
        Initialize the health checker.
        
        Args:
            client: QdrantClient instance
            config: Collection configuration (uses default if None)
        """
        self.client = client
        self.config = config or QdrantCollectionConfig()
        logger.info(f"HealthCheck initialized for collection: {self.config.collection_name}")
    
    def check_connection(self) -> bool:
        """
        Check if Qdrant connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            start_time = time.time()
            self.client.get_collections()
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Qdrant connection check successful (latency: {latency_ms:.2f}ms)")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}")
            return False
    
    def check_collection_exists(self) -> bool:
        """
        Check if the configured collection exists.
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            exists = self.config.collection_name in collection_names
            logger.debug(f"Collection {self.config.collection_name} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Failed to check collection existence: {str(e)}")
            return False
    
    def check_vector_count(self) -> int:
        """
        Get the number of vectors in the collection.
        
        Returns:
            Number of vectors (0 if collection doesn't exist or error occurs)
        """
        try:
            if not self.check_collection_exists():
                return 0
            
            info = self.client.get_collection(collection_name=self.config.collection_name)
            count = info.points_count
            logger.debug(f"Vector count for {self.config.collection_name}: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get vector count: {str(e)}")
            return 0
    
    def check_indexed_vector_count(self) -> int:
        """
        Get the number of indexed vectors in the collection.
        
        Returns:
            Number of indexed vectors (0 if collection doesn't exist or error occurs)
        """
        try:
            if not self.check_collection_exists():
                return 0
            
            info = self.client.get_collection(collection_name=self.config.collection_name)
            count = info.indexed_vectors_count
            logger.debug(f"Indexed vector count for {self.config.collection_name}: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get indexed vector count: {str(e)}")
            return 0
    
    def check_collection_status(self) -> str:
        """
        Get the status of the collection.
        
        Returns:
            Collection status string
        """
        try:
            if not self.check_collection_exists():
                return "not_found"
            
            info = self.client.get_collection(collection_name=self.config.collection_name)
            status = str(info.status)
            logger.debug(f"Collection status for {self.config.collection_name}: {status}")
            return status
        except Exception as e:
            logger.error(f"Failed to get collection status: {str(e)}")
            return "error"
    
    def perform_health_check(self) -> QdrantHealthStatus:
        """
        Perform comprehensive health check.
        
        This method checks connection, collection existence, vector count,
        and overall health status of the Qdrant adapter.
        
        Returns:
            QdrantHealthStatus with comprehensive health information
        """
        start_time = time.time()
        
        # Check connection
        connection_healthy = self.check_connection()
        
        # Check collection
        collection_exists = self.check_collection_exists()
        
        # Get vector count
        vector_count = self.check_vector_count()
        
        # Determine overall health status
        if not connection_healthy:
            status = HealthStatus.UNHEALTHY
            error_message = "Connection to Qdrant failed"
        elif not collection_exists:
            status = HealthStatus.UNHEALTHY
            error_message = f"Collection {self.config.collection_name} does not exist"
        elif vector_count == 0:
            status = HealthStatus.DEGRADED
            error_message = "Collection exists but contains no vectors"
        else:
            status = HealthStatus.HEALTHY
            error_message = None
        
        latency_ms = (time.time() - start_time) * 1000
        
        health_status = QdrantHealthStatus(
            status=status,
            connection_healthy=connection_healthy,
            collection_exists=collection_exists,
            vector_count=vector_count,
            error_message=error_message,
            latency_ms=latency_ms
        )
        
        logger.info(
            f"Health check complete - Status: {status.value}, "
            f"Connection: {connection_healthy}, "
            f"Collection: {collection_exists}, "
            f"Vectors: {vector_count}, "
            f"Latency: {latency_ms:.2f}ms"
        )
        
        return health_status
    
    def print_health_status(self, health_status: QdrantHealthStatus) -> None:
        """
        Print formatted health status to console.
        
        Args:
            health_status: Health status to print
        """
        print("\n" + "="*70)
        print("🔍 Qdrant Health Status")
        print("="*70)
        
        # Overall status
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNKNOWN: "❓"
        }
        emoji = status_emoji.get(health_status.status, "❓")
        print(f"\n{emoji} Overall Status: {health_status.status.value.upper()}")
        
        # Connection
        conn_status = "✅ Healthy" if health_status.connection_healthy else "❌ Unhealthy"
        print(f"   Connection: {conn_status}")
        
        # Collection
        coll_status = "✅ Exists" if health_status.collection_exists else "❌ Not Found"
        print(f"   Collection: {coll_status}")
        
        # Vectors
        print(f"   Vector Count: {health_status.vector_count}")
        
        # Latency
        if health_status.latency_ms:
            print(f"   Check Latency: {health_status.latency_ms:.2f}ms")
        
        # Error message
        if health_status.error_message:
            print(f"\n❌ Error: {health_status.error_message}")
        
        print("="*70 + "\n")
    
    def is_healthy(self) -> bool:
        """
        Quick health check.
        
        Returns:
            True if system is healthy, False otherwise
        """
        health_status = self.perform_health_check()
        return health_status.status == HealthStatus.HEALTHY
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the Qdrant instance.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            "collection_name": self.config.collection_name,
            "connection_healthy": self.check_connection(),
            "collection_exists": self.check_collection_exists(),
            "collection_status": self.check_collection_status(),
            "vector_count": self.check_vector_count(),
            "indexed_vector_count": self.check_indexed_vector_count(),
            "config": {
                "vector_size": self.config.vector_size,
                "distance": self.config.distance.value,
                "hnsw_config": self.config.hnsw_config.model_dump() if self.config.hnsw_config else None
            }
        }
        return diagnostics
