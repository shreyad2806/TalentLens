from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src.vector_store import VectorStoreService
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.sparse.bm25_index import BM25Index
from src.retrieval.dense.dense_retrieval_service import DenseRetrievalService
from src.retrieval.sparse.sparse_retrieval_service import SparseRetrievalService
from src.retrieval.hybrid.hybrid_retrieval_service import HybridRetrievalService

logger = logging.getLogger(__name__)

# Module-level singleton to ensure BootstrapService and app.py share the same retrieval instances
_retrieval_bundle_singleton: Optional[RetrievalBundle] = None


@dataclass(frozen=True)
class RetrievalBundle:
    vector_store_service: VectorStoreService
    embedding_service: EmbeddingService
    bm25_index: BM25Index
    dense_service: DenseRetrievalService
    sparse_service: SparseRetrievalService
    hybrid_service: HybridRetrievalService


def _maybe_load_bm25_index(bm25_index: BM25Index, bm25_index_path: Path) -> None:
    if (bm25_index_path / "metadata.json").exists():
        print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 metadata.json found at {bm25_index_path} - loading persisted index")
        logger.info(f"Loading BM25 index from {bm25_index_path}")
        bm25_index.load_from_disk(bm25_index_path)
        post_stats = bm25_index.get_statistics()
        if hasattr(post_stats, 'num_documents'):
            num_docs = post_stats.num_documents
        elif hasattr(post_stats, 'total_documents'):
            num_docs = post_stats.total_documents
        else:
            num_docs = post_stats.get('num_documents', 0) if isinstance(post_stats, dict) else 0
        print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 loaded: num_documents={num_docs}")
    else:
        print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 metadata.json NOT found at {bm25_index_path} - using EMPTY BM25 index")
        logger.info(f"BM25 index path not found ({bm25_index_path}); using empty BM25 index")


def create_retrieval_bundle(
    *,
    vector_store_service: Optional[VectorStoreService] = None,
    embedding_service: Optional[EmbeddingService] = None,
    bm25_index: Optional[BM25Index] = None,
    bm25_index_path: Optional[Path] = None,
    dense_service: Optional[DenseRetrievalService] = None,
    sparse_service: Optional[SparseRetrievalService] = None,
    hybrid_service: Optional[HybridRetrievalService] = None,
) -> RetrievalBundle:
    """Create (or reuse) retrieval services.

    If any dependencies are provided they are reused.
    Otherwise this function constructs them.
    """
    global _retrieval_bundle_singleton
    
    # Reuse the singleton if it already exists and no overrides are provided
    if _retrieval_bundle_singleton is not None and all(
        arg is None for arg in [
            vector_store_service, embedding_service, bm25_index,
            dense_service, sparse_service, hybrid_service
        ]
    ):
        print("[BOOTSTRAP-TRACE][composition_root.py] Returning existing retrieval bundle singleton")
        return _retrieval_bundle_singleton
    
    print("[BOOTSTRAP-TRACE][composition_root.py] create_retrieval_bundle() invoked")
    print("[BOOTSTRAP-TRACE][composition_root.py] NOTE: This function does NOT call BootstrapService - it only creates/loads retrieval services")

    # Defaults for persistence locations
    if bm25_index_path is None:
        bm25_index_path = Path("data/indexes/bm25")

    vector_store_service = vector_store_service or VectorStoreService()
    embedding_service = embedding_service or EmbeddingService()
    bm25_index = bm25_index or BM25Index()

    # Load persisted BM25 index if present
    _maybe_load_bm25_index(bm25_index, bm25_index_path)

    # Dense service: accept injected vector store
    dense_service = dense_service or DenseRetrievalService(vector_store_service=vector_store_service, embedding_service=embedding_service)

    # Sparse service: accept injected BM25 index
    sparse_service = sparse_service or SparseRetrievalService(index=bm25_index)

    # Hybrid service: accept injected dense/sparse
    hybrid_service = hybrid_service or HybridRetrievalService(
        dense_retrieval_service=dense_service,
        sparse_retrieval_service=sparse_service,
    )

    bundle = RetrievalBundle(
        vector_store_service=vector_store_service,
        embedding_service=embedding_service,
        bm25_index=bm25_index,
        dense_service=dense_service,
        sparse_service=sparse_service,
        hybrid_service=hybrid_service,
    )
    
    # Store as singleton for subsequent calls
    _retrieval_bundle_singleton = bundle
    
    return bundle

