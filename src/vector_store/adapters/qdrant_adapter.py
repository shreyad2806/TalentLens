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

import logging
import os
from typing import Any

from ...resume_parser.normalizer import MetadataNormalizer
from ..config import VectorStoreConfig
from ..interface import VectorStore, VectorStoreError
from ..qdrant.schema import QdrantPayload
from ..schema import VectorRecord

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
    
    def __init__(self, config: VectorStoreConfig | None = None):
        """
        Initialize the Qdrant adapter.
        
        Args:
            config: VectorStoreConfig instance (uses environment if None)
        """
        self.config = config or VectorStoreConfig()
        
        # Get Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_path = os.getenv("QDRANT_PATH", "data/vector_store/qdrant")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "talentlens_candidates")
        
        # Import and initialize production Qdrant adapter
        from ..qdrant import QdrantAdapter as ProductionQdrantAdapter
        
        self._adapter = ProductionQdrantAdapter(
            url=qdrant_url,
            path=qdrant_path,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection,
            vector_size=self.config.dimension,
            distance="Cosine"
        )
        
        logger.info(f"QdrantAdapter initialized - Collection: {qdrant_collection}")
        
        # Ensure the collection exists before any upsert/query operations.
        # This is idempotent: it returns True if the collection already exists.
        self._adapter.create_collection()
    
    def _vector_record_to_payload(self, record: VectorRecord) -> dict[str, Any]:
        """Convert VectorRecord to QdrantPayload dictionary."""
        resume_metadata = record.resume_metadata.model_dump(mode="json")
        # Normalize skills once for Qdrant's exact, case-sensitive MatchAny filter.
        resume_metadata["skills"] = MetadataNormalizer.normalize_skills_for_qdrant(
            resume_metadata.get("skills")
        )
        return {
            "resume_id": record.resume_id,
            "candidate_name": record.candidate_name,
            "chunk_id": record.chunk_id,
            "section": record.section,
            "text": record.text,
            "chunk_text": record.chunk_text,
            "original_text": record.original_text,
            "skills": resume_metadata.get("skills", []),
            "experience": resume_metadata.get("experience"),
            "location": resume_metadata.get("location"),
            "education": resume_metadata.get("education"),
            "role": resume_metadata.get("role"),
            "salary": resume_metadata.get("salary"),
            "notice_period": resume_metadata.get("notice_period"),
            "metadata": resume_metadata
        }
    
    def _search_result_to_dict(self, result) -> dict[str, Any]:
        """Convert SearchResult to dictionary format."""
        payload_dict = result.payload
        if isinstance(payload_dict, QdrantPayload):
            payload_dict = payload_dict.model_dump(mode="json")
        return {
            "id": result.id,
            "score": result.score,
            "metadata": payload_dict
        }
    
    def upsert(self, records: list[VectorRecord]) -> dict[str, Any]:
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
            logger.error(f"Upsert failed: {e!s}")
            return {
                "success": False,
                "upserted_count": 0,
                "errors": [str(e)],
                "latency_seconds": 0
            }
    
    def query(self, vector: list[float], k: int = 10, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
                    skills=MetadataNormalizer.normalize_skills_for_qdrant(filters.get("skills")),
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
            logger.error(f"Query failed: {e!s}")
            raise VectorStoreError(f"Query failed: {e!s}", "QdrantAdapter", e)
    
    def delete(self, ids: list[str]) -> dict[str, Any]:
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
            logger.error(f"Delete failed: {e!s}")
            return {
                "success": False,
                "deleted_count": 0,
                "errors": [str(e)]
            }
    
    def delete_resume(self, resume_id: str) -> dict[str, Any]:
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
    
    def _payload_to_vector_record(self, payload: dict[str, Any], id: str, vector: list[float]) -> VectorRecord:
        """Convert a Qdrant payload back into a VectorRecord."""
        from src.models import ResumeMetadata
        metadata = payload.get("metadata", {})
        resume_metadata = ResumeMetadata.model_validate(metadata)
        return VectorRecord(
            id=id,
            chunk_id=payload.get("chunk_id", id),
            section=payload.get("section", "unknown"),
            text=payload.get("text"),
            chunk_text=payload.get("chunk_text"),
            original_text=payload.get("original_text"),
            vector=vector,
            resume_metadata=resume_metadata
        )

    def fetch(self, id: str) -> VectorRecord | None:
        """
        Fetch a single vector record by its ID.

        Args:
            id: Record ID to fetch

        Returns:
            VectorRecord if found, None otherwise
        """
        try:
            points = self._adapter.client.retrieve(
                collection_name=self._adapter.collection_name,
                ids=[id],
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                return None
            point = points[0]
            payload = point.payload if hasattr(point, "payload") else {}
            if isinstance(payload, QdrantPayload):
                payload = payload.model_dump(mode="json")
            vector = point.vector if hasattr(point, "vector") else []
            return self._payload_to_vector_record(payload, str(point.id), vector)
        except Exception as e:
            logger.error(f"Fetch failed for id={id}: {e!s}")
            return None

    def fetch_resume(self, resume_id: str) -> list[VectorRecord]:
        """
        Fetch all vector records for a specific resume.

        Args:
            resume_id: ID of the resume to fetch

        Returns:
            List of VectorRecord objects for the resume
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            response = self._adapter.client.scroll(
                collection_name=self._adapter.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="resume_id", match=MatchValue(value=resume_id))]
                ),
                with_payload=True,
                with_vectors=True,
            )
            points = response[0] if isinstance(response, tuple) else response
            records = []
            for point in points:
                payload = point.payload if hasattr(point, "payload") else {}
                if isinstance(payload, QdrantPayload):
                    payload = payload.model_dump(mode="json")
                vector = point.vector if hasattr(point, "vector") else []
                records.append(self._payload_to_vector_record(payload, str(point.id), vector))
            return records
        except Exception as e:
            logger.error(f"fetch_resume failed for resume_id={resume_id}: {e!s}")
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
            logger.error(f"Count failed: {e!s}")
            return 0
    
    def clear(self) -> dict[str, Any]:
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
            logger.error(f"Clear failed: {e!s}")
            return {
                "success": False,
                "cleared_count": 0,
                "errors": [str(e)]
            }
    
    def health(self) -> dict[str, Any]:
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
            logger.error(f"Health check failed: {e!s}")
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": str(e),
                "latency_ms": 0,
                "record_count": 0
            }
    
    def save(self, path: str | None = None) -> dict[str, Any]:
        """Qdrant local-mode persistence is handled by the client; this is a no-op."""
        return {
            "success": True,
            "path": path or self._adapter.path,
            "vectors_saved": self._adapter.count(),
            "message": "Qdrant persistence is managed automatically by the client"
        }
    
    def load(self, path: str | None = None) -> dict[str, Any]:
        """Qdrant local-mode data is loaded automatically on client init."""
        return {
            "success": True,
            "path": path or self._adapter.path,
            "vectors_restored": self._adapter.count(),
            "message": "Qdrant data is loaded automatically on adapter initialization"
        }
    
    def serialize(self) -> dict[str, Any]:
        """Serialization is not applicable for the Qdrant adapter."""
        return {"success": False, "message": "Qdrant does not support manual serialization"}
    
    def deserialize(self, data: dict[str, Any]) -> None:
        """Deserialization is not applicable for the Qdrant adapter."""
    
    def integrity_check(self) -> dict[str, Any]:
        """Validate Qdrant collection integrity."""
        errors = []
        warnings = []
        
        try:
            count = self._adapter.count()
            health = self._adapter.health_check()
            
            if not health.collection_exists:
                errors.append("Qdrant collection does not exist")
            
            if health.vector_count != count:
                warnings.append(f"Health count ({health.vector_count}) differs from count() ({count})")
            
            return {
                "valid": len(errors) == 0 and health.collection_exists,
                "dimension": self.config.dimension,
                "count": count,
                "metadata_count": count,
                "collection_exists": health.collection_exists,
                "connection_healthy": health.connection_healthy,
                "errors": errors,
                "warnings": warnings
            }
        except Exception as e:
            return {
                "valid": False,
                "dimension": self.config.dimension,
                "count": 0,
                "metadata_count": 0,
                "errors": [str(e)],
                "warnings": []
            }
    
    def close(self) -> None:
        """
        Close the vector store connection and release resources.
        """
        # Qdrant client doesn't need explicit closing
