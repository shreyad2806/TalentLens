"""
Memory Adapter - In-memory vector store implementation.

This module provides the MemoryVectorStore adapter, which implements the
VectorStore interface using an in-memory Python dictionary. This is useful
for testing and development purposes.

Architecture Notes:
- Adapter Pattern: Implements VectorStore interface
- No external dependencies - pure Python
- Data is not persisted - lost when process exits
- Uses cosine similarity for vector search
"""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from ...models import ResumeMetadata
from ..config import VectorStoreConfig
from ..interface import VectorStore, VectorStoreError
from ..schema import VectorRecord


class MemoryVectorStore(VectorStore):
    """
    In-memory vector store implementation.
    
    This class implements the VectorStore interface using a Python dictionary
    for storage. It provides a simple, dependency-free implementation suitable
    for testing and development.
    
    Architecture Pattern: Adapter Pattern
    - Implements VectorStore interface
    - Adapts dictionary storage to vector store contract
    - No external dependencies
    
    Storage Structure:
    {
        "id": {
            "vector": [float],
            "metadata": dict
        }
    }
    """
    
    def __init__(self, config: VectorStoreConfig | None = None):
        """
        Initialize the memory vector store.
        
        Args:
            config: Optional configuration. If None, uses default config.
        """
        self.config = config
        self._store: dict[str, dict[str, Any]] = {}
        self._closed = False
    
    def upsert(self, records: list[VectorRecord]) -> dict[str, Any]:
        """
        Insert or update vector records in the store.
        
        Args:
            records: List of VectorRecord objects to upsert
            
        Returns:
            Dictionary with operation results
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot upsert: store is closed", adapter_name="MemoryVectorStore")
        
        upserted_count = 0
        errors = []
        
        for record in records:
            try:
                self._store[record.id] = {
                    "vector": record.vector,
                    "chunk_id": record.chunk_id,
                    "section": record.section,
                    "resume_metadata": record.resume_metadata.model_dump(mode="json")
                }
                upserted_count += 1
            except Exception as e:
                errors.append(f"Failed to upsert record {record.id}: {e!s}")
        
        return {
            "success": len(errors) == 0,
            "upserted_count": upserted_count,
            "errors": errors
        }
    
    def query(self, vector: list[float], k: int = 10, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Query the vector store for similar vectors using cosine similarity.
        
        Args:
            vector: Query vector to search for
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of dictionaries containing search results
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot query: store is closed", adapter_name="MemoryVectorStore")
        
        if not self._store:
            return []
        
        # Calculate cosine similarity for all vectors
        results = []
        query_norm = math.sqrt(sum(x * x for x in vector))
        
        if query_norm == 0:
            return []
        
        for record_id, data in self._store.items():
            # Apply filters if provided
            if filters:
                if not self._apply_filters(data["resume_metadata"], filters):
                    continue

            # Calculate cosine similarity
            stored_vector = data["vector"]
            stored_norm = math.sqrt(sum(x * x for x in stored_vector))

            if stored_norm == 0:
                continue

            dot_product = sum(x * y for x, y in zip(vector, stored_vector))
            cosine_similarity = dot_product / (query_norm * stored_norm)

            results.append({
                "id": record_id,
                "score": cosine_similarity,
                "metadata": {**data["resume_metadata"], "chunk_id": data["chunk_id"], "section": data["section"]}
            })
        
        # Sort by score descending and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    def delete(self, ids: list[str]) -> dict[str, Any]:
        """
        Delete vector records by their IDs.
        
        Args:
            ids: List of record IDs to delete
            
        Returns:
            Dictionary with operation results
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot delete: store is closed", adapter_name="MemoryVectorStore")
        
        deleted_count = 0
        errors = []
        
        for record_id in ids:
            if record_id in self._store:
                del self._store[record_id]
                deleted_count += 1
            else:
                errors.append(f"Record not found: {record_id}")
        
        return {
            "success": len(errors) == 0,
            "deleted_count": deleted_count,
            "errors": errors
        }
    
    def delete_resume(self, resume_id: str) -> dict[str, Any]:
        """
        Delete all vector records for a specific resume.
        
        Args:
            resume_id: ID of the resume to delete
            
        Returns:
            Dictionary with operation results
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot delete resume: store is closed", adapter_name="MemoryVectorStore")
        
        deleted_count = 0
        ids_to_delete = []
        
        for record_id, data in self._store.items():
            if data["resume_metadata"].get("resume_id") == resume_id:
                ids_to_delete.append(record_id)
        
        for record_id in ids_to_delete:
            del self._store[record_id]
            deleted_count += 1
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "errors": []
        }
    
    def fetch(self, id: str) -> VectorRecord | None:
        """
        Fetch a single vector record by its ID.
        
        Args:
            id: Record ID to fetch
            
        Returns:
            VectorRecord if found, None otherwise
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch: store is closed", adapter_name="MemoryVectorStore")
        
        if id not in self._store:
            return None
        
        data = self._store[id]
        resume_metadata = ResumeMetadata.model_validate(data["resume_metadata"])

        return VectorRecord(
            id=id,
            chunk_id=data["chunk_id"],
            section=data["section"],
            vector=data["vector"],
            resume_metadata=resume_metadata
        )
    
    def fetch_resume(self, resume_id: str) -> list[VectorRecord]:
        """
        Fetch all vector records for a specific resume.
        
        Args:
            resume_id: ID of the resume to fetch
            
        Returns:
            List of VectorRecord objects for the resume
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot fetch resume: store is closed", adapter_name="MemoryVectorStore")
        
        records = []
        
        for record_id, data in self._store.items():
            if data["resume_metadata"].get("resume_id") == resume_id:
                records.append(VectorRecord(
                    id=record_id,
                    chunk_id=data["chunk_id"],
                    section=data["section"],
                    vector=data["vector"],
                    resume_metadata=ResumeMetadata.model_validate(data["resume_metadata"])
                ))

        return records
    
    def count(self) -> int:
        """
        Get the total number of vector records in the store.
        
        Returns:
            Total number of records
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot count: store is closed", adapter_name="MemoryVectorStore")
        
        return len(self._store)
    
    def clear(self) -> dict[str, Any]:
        """
        Clear all vector records from the store.
        
        Returns:
            Dictionary with operation results
            
        Raises:
            VectorStoreError: If store is closed
        """
        if self._closed:
            raise VectorStoreError("Cannot clear: store is closed", adapter_name="MemoryVectorStore")
        
        cleared_count = len(self._store)
        self._store.clear()
        
        return {
            "success": True,
            "cleared_count": cleared_count,
            "errors": []
        }
    
    def health(self) -> dict[str, Any]:
        """
        Check the health status of the vector store.
        
        Returns:
            Dictionary with health status
        """
        return {
            "healthy": not self._closed,
            "status": "healthy" if not self._closed else "unhealthy",
            "message": "Memory vector store is operational" if not self._closed else "Memory vector store is closed",
            "adapter": "MemoryVectorStore",
            "record_count": len(self._store)
        }
    
    def close(self) -> None:
        """
        Close the vector store connection and release resources.
        
        For the memory adapter, this simply marks the store as closed.
        """
        self._closed = True
        self._store.clear()
    
    def _apply_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """
        Apply metadata filters before similarity search.

        Supports the same filter keys used by the Qdrant adapter:
        - skills: list intersection (candidate has at least one required skill)
        - location: case-insensitive exact/substring match
        - experience_min/experience_max: numeric range against metadata experience_years/experience
        - unknown keys are ignored so callers do not break the search
        """
        def _normalize(value: Any) -> str:
            return str(value).strip().lower()

        for key, value in filters.items():
            if key == "skills":
                candidate_skills = metadata.get("skills", [])
                if not candidate_skills:
                    return False
                required = {_normalize(s) for s in value}
                candidate = {_normalize(s) for s in candidate_skills}
                if not (required & candidate):
                    return False

            elif key == "location":
                candidate_location = metadata.get("location")
                if not candidate_location:
                    return False
                if _normalize(value) not in _normalize(candidate_location):
                    return False

            elif key == "experience_min":
                exp = metadata.get("experience_years") or metadata.get("experience")
                if exp is None or float(exp) < float(value):
                    return False

            elif key == "experience_max":
                exp = metadata.get("experience_years") or metadata.get("experience")
                if exp is None or float(exp) > float(value):
                    return False

            elif key in metadata:
                if metadata[key] != value:
                    return False

        return True
    
    def serialize(self) -> dict[str, Any]:
        """Convert the in-memory store to a serializable dictionary."""
        records = []
        for record_id, data in self._store.items():
            records.append({
                "id": record_id,
                "vector": data["vector"],
                "chunk_id": data["chunk_id"],
                "section": data["section"],
                "resume_metadata": data["resume_metadata"]
            })
        
        dimension = self.config.dimension if self.config else 384
        if records:
            dimension = len(records[0]["vector"])
        
        # Simple checksum of the JSON representation of the records
        payload = json.dumps(records, sort_keys=True, default=str)
        checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        
        return {
            "version": "1.0",
            "dimension": dimension,
            "count": len(records),
            "metadata_count": sum(1 for r in records if r.get("resume_metadata")),
            "checksum": checksum,
            "records": records,
            "saved_at": time.time()
        }
    
    def deserialize(self, data: dict[str, Any]) -> None:
        """Restore the in-memory store from a serialized dictionary."""
        if not data or "records" not in data:
            raise VectorStoreError("Invalid memory vector store data: missing 'records'", "MemoryVectorStore")
        
        self._store.clear()
        for record in data["records"]:
            self._store[record["id"]] = {
                "vector": record["vector"],
                "chunk_id": record.get("chunk_id", ""),
                "section": record.get("section", ""),
                "resume_metadata": record.get("resume_metadata", {})
            }
    
    def save(self, path: str | None = None) -> dict[str, Any]:
        """Persist the in-memory store to a JSON file."""
        save_path = Path(path) if path else Path("data/vector_store/memory.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.serialize()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return {
            "success": True,
            "path": str(save_path),
            "vectors_saved": data["count"],
            "dimension": data["dimension"],
            "checksum": data["checksum"]
        }
    
    def load(self, path: str | None = None) -> dict[str, Any]:
        """Restore the in-memory store from a JSON file."""
        load_path = Path(path) if path else Path("data/vector_store/memory.json")
        
        if not load_path.exists():
            return {
                "success": False,
                "path": str(load_path),
                "vectors_restored": 0,
                "error": f"Persistence file not found: {load_path}"
            }
        
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.deserialize(data)
            
            return {
                "success": True,
                "path": str(load_path),
                "vectors_restored": data.get("count", 0),
                "dimension": data.get("dimension", 0),
                "checksum": data.get("checksum", "")
            }
        except Exception as e:
            return {
                "success": False,
                "path": str(load_path),
                "vectors_restored": 0,
                "error": str(e)
            }
    
    def integrity_check(self) -> dict[str, Any]:
        """Validate the in-memory store's integrity."""
        errors = []
        warnings = []
        
        expected_dimension = self.config.dimension if self.config else None
        count = len(self._store)
        metadata_count = 0
        dimension = None
        
        for record_id, data in self._store.items():
            vector = data.get("vector", [])
            if not vector:
                errors.append(f"Record {record_id}: empty vector")
                continue
            
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                errors.append(f"Record {record_id}: dimension mismatch ({len(vector)} != {dimension})")
            
            if expected_dimension is not None and len(vector) != expected_dimension:
                errors.append(f"Record {record_id}: dimension {len(vector)} does not match expected {expected_dimension}")
            
            if not data.get("resume_metadata"):
                warnings.append(f"Record {record_id}: missing metadata")
            else:
                metadata_count += 1
        
        return {
            "valid": len(errors) == 0,
            "dimension": dimension,
            "count": count,
            "metadata_count": metadata_count,
            "expected_dimension": expected_dimension,
            "errors": errors,
            "warnings": warnings
        }
