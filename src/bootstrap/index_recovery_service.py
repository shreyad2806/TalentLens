"""
Index Recovery Service for Bootstrap System.

This module is owned by BootstrapService and runs only during startup.  It
verifies that the persisted BM25, dense vector, and indexed-document state is
consistent after CSV / file ingestion and, when necessary, rebuilds the BM25
index from the cached chunks without re-embedding or re-parsing resumes.

Recovery is intentionally separated from CSVIngestionService and from
SearchService/retrieval code so that it cannot affect search or CSV parsing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..indexing.indexing_service import IndexingService
from ..chunks.schema import Chunk
from ..models import ResumeMetadata

logger = logging.getLogger(__name__)


class IndexRecoveryService:
    """
    Bootstrap-only recovery service for index consistency.

    Responsibilities:
    - Verify BM25, dense vector, and indexed-document counts.
    - If BM25 count is 0 but chunks are cached, rebuild BM25 from those chunks.
    - Persist the rebuilt BM25 index.
    - Re-verify and fail startup if the index remains empty.
    """

    def __init__(
        self,
        indexing_service: IndexingService,
        cache_dir: Optional[Path] = None,
        bm25_index_path: Optional[Path] = None,
    ):
        self.indexing_service = indexing_service
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.bm25_index_path = Path(bm25_index_path) if bm25_index_path else Path("data/indexes/bm25")

    def _load_cached_chunks(self) -> Optional[List[Chunk]]:
        """Load cached chunk objects from disk (not from CSV)."""
        chunks_path = self.cache_dir / "chunks.json"
        if not chunks_path.exists():
            logger.warning("No cached chunks found at %s", chunks_path)
            return None

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            chunks = []
            for data in chunks_data:
                try:
                    # Pydantic V2 model_validate can parse the stored dict back
                    # into a Chunk object. Legacy caches may be missing the canonical
                    # resume_metadata; supply a minimal ResumeMetadata object so the
                    # canonical field is always present.
                    if not data.get("resume_metadata"):
                        data["resume_metadata"] = ResumeMetadata(
                            resume_id=data.get("resume_id", ""),
                            candidate_name=data.get("candidate_name"),
                        ).model_dump(mode="json")
                    chunk = Chunk.model_validate(data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning("Failed to reconstruct chunk %s: %s", data.get("chunk_id"), e)

            logger.info("Loaded %d cached chunks for recovery", len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to load cached chunks: %s", e)
            return None

    def verify_indexes(self) -> Dict[str, Any]:
        """Return current index counts."""
        stats = self.indexing_service.get_statistics()
        return {
            "indexed_documents": stats.get("indexed_documents", 0),
            "vector_count": stats.get("vector_count", 0),
            "bm25_count": stats.get("bm25_count", 0),
            "bm25_stats": stats.get("bm25_stats"),
        }

    def recover_indexes(self) -> Dict[str, Any]:
        """
        Run index recovery.

        If BM25 is empty but cached chunks exist, rebuild BM25 from those
        chunks and persist the result.  If recovery fails, raise RuntimeError
        so BootstrapService can fail startup cleanly.
        """
        pre_recovery = self.verify_indexes()
        logger.info("Pre-recovery state: %s", pre_recovery)

        if pre_recovery["bm25_count"] > 0:
            return {
                "status": "ok",
                "action": "none_needed",
                "pre_recovery": pre_recovery,
                "post_recovery": pre_recovery,
            }

        chunks = self._load_cached_chunks()
        if not chunks:
            raise RuntimeError(
                "Index recovery failed: BM25 count is 0 and no cached chunks are available."
            )

        if self.indexing_service._bm25_index is None:
            raise RuntimeError("Index recovery failed: IndexingService has no BM25 index.")

        # Rebuild BM25 from cached chunks using the same builder/index instance
        # that the rest of the system uses.
        logger.info("Rebuilding BM25 index from %d cached chunks", len(chunks))
        start = time.perf_counter()

        try:
            self.indexing_service.index_builder.build_index_incremental(
                self.indexing_service._bm25_index, chunks
            )
        except Exception as e:
            raise RuntimeError(f"BM25 rebuild from cached chunks failed: {e}") from e

        # Persist the rebuilt index
        try:
            self.indexing_service._bm25_index.save_to_disk(self.bm25_index_path)
            logger.info("Rebuilt BM25 index persisted to %s", self.bm25_index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to persist rebuilt BM25 index: {e}") from e

        post_recovery = self.verify_indexes()
        post_recovery["rebuild_time_ms"] = (time.perf_counter() - start) * 1000
        logger.info("Post-recovery state: %s", post_recovery)

        if post_recovery["bm25_count"] == 0:
            raise RuntimeError(
                "Index recovery failed: BM25 count is still 0 after rebuild."
            )

        return {
            "status": "recovered",
            "action": "rebuilt_bm25_from_cache",
            "pre_recovery": pre_recovery,
            "post_recovery": post_recovery,
        }
