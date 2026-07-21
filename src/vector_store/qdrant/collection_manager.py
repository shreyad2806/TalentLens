"""
Collection Manager for Qdrant Vector Store.

This module provides collection lifecycle management for Qdrant, including
creation, deletion, and configuration management.

SOLID Principles Applied:
- Single Responsibility: Handles only collection management
- Open/Closed: Open for new collection configurations
"""

import logging
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    CollectionParams,
    PayloadSchemaType,
    KeywordIndexParams,
    FloatIndexParams,
    IntegerIndexParams,
)

from .schema import (
    QdrantCollectionConfig,
    DistanceMetric,
    HnswConfig,
    CollectionInfo,
)

logger = logging.getLogger(__name__)


class CollectionManager:
    """
    Manager for Qdrant collection lifecycle.
    
    This class handles creation, deletion, and configuration of Qdrant
    collections with proper schema definition and indexing.
    """
    
    def __init__(
        self,
        client: QdrantClient,
        config: Optional[QdrantCollectionConfig] = None
    ):
        """
        Initialize the collection manager.
        
        Args:
            client: QdrantClient instance
            config: Collection configuration (uses default if None)
        """
        self.client = client
        self.config = config or QdrantCollectionConfig()
        logger.info(f"CollectionManager initialized for collection: {self.config.collection_name}")
    
    def create_collection(self) -> bool:
        """
        Create a new collection with the configured schema.
        
        This method creates a collection with proper vector configuration,
        HNSW indexing, and payload schema for metadata filtering.
        
        Returns:
            True if collection created successfully, False otherwise
        """
        try:
            # Check if collection already exists
            if self.collection_exists():
                logger.info(f"Collection {self.config.collection_name} already exists")
                return True
            
            # Map distance metric
            distance_map = {
                DistanceMetric.COSINE: Distance.COSINE,
                DistanceMetric.EUCLID: Distance.EUCLID,
                DistanceMetric.DOT: Distance.DOT,
            }
            
            # Create HNSW config
            hnsw_config = None
            if self.config.hnsw_config:
                hnsw_config = HnswConfigDiff(
                    m=self.config.hnsw_config.m,
                    ef_construct=self.config.hnsw_config.ef_construct,
                    full_scan_threshold=self.config.hnsw_config.full_scan_threshold
                )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map[self.config.distance],
                    hnsw_config=hnsw_config
                ),
                optimizers_config=None,
                replication_factor=1,
                write_consistency_factor=1,
                on_disk_payload=True,
                hnsw_config=hnsw_config
            )
            
            # Create payload indexes for filtering performance
            self._create_payload_indexes()
            
            logger.info(f"Collection {self.config.collection_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.config.collection_name}: {str(e)}")
            return False
    
    def _create_payload_indexes(self) -> None:
        """
        Create payload indexes for efficient filtering.
        
        This method creates indexes on frequently filtered fields
        to improve query performance.
        """
        try:
            # Index skills for skill filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="skills",
                field_schema=PayloadSchemaType.KEYWORD,
                field_params=KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.debug("Created index on skills field")
            
            # Index location for location filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="location",
                field_schema=PayloadSchemaType.KEYWORD,
                field_params=KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.debug("Created index on location field")
            
            # Index experience for range filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="experience",
                field_schema=PayloadSchemaType.FLOAT,
                field_params=FloatIndexParams(
                    type="float",
                    is_tenant=False
                )
            )
            logger.debug("Created index on experience field")
            
            # Index role for role filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="role",
                field_schema=PayloadSchemaType.KEYWORD,
                field_params=KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.debug("Created index on role field")
            
            # Index education for education filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="education",
                field_schema=PayloadSchemaType.KEYWORD,
                field_params=KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.debug("Created index on education field")
            
            # Index salary for range filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="salary",
                field_schema=PayloadSchemaType.FLOAT,
                field_params=FloatIndexParams(
                    type="float",
                    is_tenant=False
                )
            )
            logger.debug("Created index on salary field")
            
            # Index notice_period for range filtering
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="notice_period",
                field_schema=PayloadSchemaType.INTEGER,
                field_params=IntegerIndexParams(
                    type="integer",
                    is_tenant=False
                )
            )
            logger.debug("Created index on notice_period field")
            
            logger.info("All payload indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some payload indexes: {str(e)}")
            # Don't fail the collection creation if indexes fail
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if collection deleted successfully, False otherwise
        """
        try:
            if not self.collection_exists():
                logger.info(f"Collection {self.config.collection_name} does not exist")
                return True
            
            self.client.delete_collection(collection_name=self.config.collection_name)
            logger.info(f"Collection {self.config.collection_name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {self.config.collection_name}: {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
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
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """
        Get detailed information about the collection.
        
        Returns:
            CollectionInfo if collection exists, None otherwise
        """
        try:
            if not self.collection_exists():
                return None
            
            info = self.client.get_collection(collection_name=self.config.collection_name)
            
            collection_info = CollectionInfo(
                name=self.config.collection_name,
                vectors_count=info.vectors_count,
                indexed_vectors_count=info.indexed_vectors_count,
                points_count=info.points_count,
                segments_count=info.segments_count,
                status=str(info.status),
                config=self.config
            )
            
            logger.debug(f"Collection info retrieved for {self.config.collection_name}")
            return collection_info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return None
    
    def update_collection_config(self, new_config: QdrantCollectionConfig) -> bool:
        """
        Update collection configuration.
        
        Note: Some configuration changes may require collection recreation.
        
        Args:
            new_config: New collection configuration
        
        Returns:
            True if configuration updated successfully, False otherwise
        """
        try:
            self.config = new_config
            logger.info(f"Collection configuration updated for {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update collection configuration: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all vectors from the collection without deleting it.
        
        Returns:
            True if collection cleared successfully, False otherwise
        """
        try:
            if not self.collection_exists():
                logger.info(f"Collection {self.config.collection_name} does not exist")
                return True
            
            # Delete all points
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=None  # Delete all points
            )
            
            logger.info(f"Collection {self.config.collection_name} cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection {self.config.collection_name}: {str(e)}")
            return False
