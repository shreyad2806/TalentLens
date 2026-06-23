"""
Qdrant Adapter - Qdrant vector database implementation.

This module provides the QdrantAdapter class, which implements the
VectorStore interface using the production Qdrant adapter from the
qdrant/ subdirectory.

Architecture Notes:
- Adapter Pattern: Implements VectorStore interface
- Delegates to production QdrantAdapter from qdrant/ subdirectory
- Converts between Qdrant format and internal VectorRecord schema
- No Qdrant SDK objects leak outside this adapter

SOLID Principles Applied:
- Single Responsibility: Handles only interface adaptation
- Open/Closed: Open for extension, closed for modification
- Dependency Inversion: Depends on VectorStore abstraction
- Interface Segregation: Implements only required methods
"""

import os
import logging
from typing import List, Dict, Any, Optional
from ..interface import VectorStore, VectorStoreError
from ..schema import VectorRecord
from ..config import VectorStoreConfig

logger = logging.getLogger(__name__)


class QdrantAdapter(VectorStore):
    """
    Qdrant vector store adapter implementation.
    
    This class implements the VectorStore interface by delegating to the
    production QdrantAdapter from the qdrant/ subdirectory. It handles
    schema conversion between VectorRecord and QdrantPayload.
    
    Architecture Pattern: Adapter Pattern + Delegation
    - Implements VectorStore interface
    - Delegates to production QdrantAdapter
    - Converts between VectorRecord and QdrantPayload
    - No Qdrant objects leak outside this adapter
    
    Environment Variables:
    - QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    - QDRANT_COLLECTION: Qdrant collection name (default: talentlens_candidates)
    - QDRANT_API_KEY: Optional Qdrant API key
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the Qdrant adapter.
        
        Args:
            config: VectorStoreConfig instance (uses environment if None)
        """
        self.config = config or VectorStoreConfig()
        
        # Get Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "talentlens_candidates")
        
        # Import and initialize production Qdrant adapter
        from ..qdrant import QdrantAdapter as ProductionQdrantAdapter
        
        self._adapter = ProductionQdrantAdapter(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection,
            vector_size=self.config.dimension,
            distance="Cosine"
        )
        
        logger.info(f"QdrantAdapter initialized - Collection: {qdrant_collection}")
    
    def _vector_record_to_payload(self, record: VectorRecord) -> Dict[str, Any]:
        """Convert VectorRecord to QdrantPayload dictionary."""
        return {
            "resume_id": record.resume_id,
            "candidate_name": record.candidate_name,
            "chunk_id": record.chunk_id,
            "section": record.section,
            "skills": record.metadata.get("skills", []),
            "experience": record.metadata.get("experience"),
            "location": record.metadata.get("location"),
            "education": record.metadata.get("education"),
            "role": record.metadata.get("role"),
            "salary": record.metadata.get("salary"),
            "notice_period": record.metadata.get("notice_period"),
            "metadata": record.metadata
        }
    
    def _search_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert SearchResult to dictionary format."""
        return {
            "id": result.id,
            "score": result.score,
            "record": VectorRecord(
                id=result.id,
                resume_id=result.payload.resume_id,
                chunk_id=result.payload.chunk_id,
                candidate_name=result.payload.candidate_name,
                section=result.payload.section,
                vector=[],  # Vector not returned in search
                metadata=result.payload.metadata
            ),
            "metadata": result.payload.metadata
        }
    
    def upsert(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Insert or update vector records in the store.
        
        Args:
            records: List of VectorRecord objects to upsert
            
        Returns:
            Dictionary with operation results
        """
        try:
            vectors = [record.vector for record in records]
            payloads = [self._vector_record_to_payload(record) for record in records]
            ids = [record.id for record in records]
            
            result = self._adapter.upsert_vectors(vectors, payloads, ids)
            
            return {
                "success": True,
                "upserted_count": result.upserted_count,
                "errors": [],
                "latency_seconds": result.latency_ms / 1000
            }
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            return {
                "success": False,
                "upserted_count": 0,
                "errors": [str(e)],
                "latency_seconds": 0
            }
    
    def query(self, vector: List[float], k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar vectors.
        
        Args:
            vector: Query vector to search for
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search result dictionaries
        """
        try:
            if filters:
                from ..qdrant import QdrantFilter
                qdrant_filter = QdrantFilter(
                    skills=filters.get("skills"),
                    experience_min=filters.get("experience_min"),
                    experience_max=filters.get("experience_max"),
                    location=filters.get("location"),
                    education=filters.get("education"),
                    role=filters.get("role")
                )
                results = self._adapter.search_with_filters(vector, qdrant_filter, top_k=k)
            else:
                results = self._adapter.search(vector, top_k=k)
            
            return [self._search_result_to_dict(result) for result in results]
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise VectorStoreError(f"Query failed: {str(e)}", "QdrantAdapter", e)
    
    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """
        Delete vector records by their IDs.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            Dictionary with operation results
        """
        try:
            success = self._adapter.delete_points(ids)
            return {
                "success": success,
                "deleted_count": len(ids) if success else 0,
                "errors": []
            }
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            return {
                "success": False,
                "deleted_count": 0,
                "errors": [str(e)]
            }
    
    def delete_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Delete all vector records for a specific resume.
        
        Args:
            resume_id: ID of the resume to delete
            
        Returns:
            Dictionary with operation results
        """
        # This would require a more complex query to find all chunks for a resume
        # For now, we'll return not implemented
        return {
            "success": False,
            "deleted_count": 0,
            "errors": ["delete_resume not yet implemented for Qdrant adapter"]
        }
    
    def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        Fetch a single vector record by its ID.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
        """
        # This would require implementing a fetch by ID in the production adapter
        # For now, we'll return None
        return None
    
    def fetch_resume(self, resume_id: str) -> List[VectorRecord]:
        """
        Fetch all vector records for a specific resume.
        
        Args:
            resume_id: ID of the resume to fetch
            
        Returns:
            List of VectorRecord objects for the resume
        """
        # This would require implementing a fetch by resume_id in the production adapter
        # For now, we'll return empty list
        return []
    
    def count(self) -> int:
        """
        Get the total number of vector records in the store.
        
        Returns:
            Total number of records
        """
        try:
            return self._adapter.count()
        except Exception as e:
            logger.error(f"Count failed: {str(e)}")
            return 0
    
    def clear(self) -> Dict[str, Any]:
        """
        Clear all vector records from the store.
        
        Returns:
            Dictionary with operation results
        """
        try:
            success = self._adapter.clear_collection()
            # Recreate collection after clearing
            if success:
                self._adapter.create_collection()
            return {
                "success": success,
                "cleared_count": 0,  # Count before clear not tracked
                "errors": []
            }
        except Exception as e:
            logger.error(f"Clear failed: {str(e)}")
            return {
                "success": False,
                "cleared_count": 0,
                "errors": [str(e)]
            }
    
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the vector store.
        
        Returns:
            Dictionary with health status
        """
        try:
            health_status = self._adapter.health_check()
            return {
                "healthy": health_status.status.value == "healthy",
                "status": health_status.status.value,
                "message": health_status.error_message or "OK",
                "latency_ms": health_status.latency_ms,
                "record_count": health_status.vector_count
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": str(e),
                "latency_ms": 0,
                "record_count": 0
            }
    
    def close(self) -> None:
        """
        Close the vector store connection and release resources.
        """
        # Qdrant client doesn't need explicit closing
        pass
