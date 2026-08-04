from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.dense.dense_retrieval_service import DenseRetrievalService
from src.retrieval.hybrid.hybrid_retrieval_service import HybridRetrievalService
from src.retrieval.sparse.bm25_index import BM25Index, IncompatibleIndexError
from src.retrieval.sparse.sparse_retrieval_service import SparseRetrievalService
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# Module-level singleton to ensure BootstrapService and app.py share the same retrieval instances
_retrieval_bundle_singleton: RetrievalBundle | None = None


@dataclass(frozen=True)
class RetrievalBundle:
    vector_store_service: VectorStoreService
    embedding_service: EmbeddingService
    bm25_index: BM25Index
    dense_service: DenseRetrievalService
    sparse_service: SparseRetrievalService
    hybrid_service: HybridRetrievalService
    reranker: CrossEncoderReranker
    startup_metrics: dict[str, float]


def _maybe_load_bm25_index(bm25_index: BM25Index, bm25_index_path: Path) -> None:
    if (bm25_index_path / "metadata.json").exists():
        print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 metadata.json found at {bm25_index_path} - loading persisted index")
        logger.info(f"Loading BM25 index from {bm25_index_path}")
        try:
            bm25_index.load_from_disk(bm25_index_path)
            post_stats = bm25_index.get_statistics()
            if hasattr(post_stats, 'num_documents'):
                num_docs = post_stats.num_documents
            elif hasattr(post_stats, 'total_documents'):
                num_docs = post_stats.total_documents
            else:
                num_docs = post_stats.get('num_documents', 0) if isinstance(post_stats, dict) else 0
            print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 loaded: num_documents={num_docs}")
        except IncompatibleIndexError as e:
            logger.error(f"Incompatible persisted BM25 index at {bm25_index_path}: {e}")
            print("[BOOTSTRAP-TRACE][composition_root.py] Incompatible persisted index; discarding and starting fresh")
            shutil.rmtree(bm25_index_path, ignore_errors=True)
    else:
        print(f"[BOOTSTRAP-TRACE][composition_root.py] BM25 metadata.json NOT found at {bm25_index_path} - using EMPTY BM25 index")
        logger.info(f"BM25 index path not found ({bm25_index_path}); using empty BM25 index")


def create_retrieval_bundle(
    *,
    vector_store_service: VectorStoreService | None = None,
    embedding_service: EmbeddingService | None = None,
    bm25_index: BM25Index | None = None,
    bm25_index_path: Path | None = None,
    dense_service: DenseRetrievalService | None = None,
    sparse_service: SparseRetrievalService | None = None,
    hybrid_service: HybridRetrievalService | None = None,
    reranker: CrossEncoderReranker | None = None,
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
            dense_service, sparse_service, hybrid_service, reranker
        ]
    ):
        print("[BOOTSTRAP-TRACE][composition_root.py] Returning existing retrieval bundle singleton")
        return _retrieval_bundle_singleton
    
    print("[BOOTSTRAP-TRACE][composition_root.py] create_retrieval_bundle() invoked")
    print("[BOOTSTRAP-TRACE][composition_root.py] NOTE: This function does NOT call BootstrapService - it only creates/loads retrieval services")

    bundle_start = time.perf_counter()
    stage_start = bundle_start
    startup_metrics: dict[str, float] = {}

    # Defaults for persistence locations
    if bm25_index_path is None:
        bm25_index_path = Path("data/indexes/bm25")

    vector_store_service = vector_store_service or VectorStoreService()
    startup_metrics["vector_store_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    embedding_service = embedding_service or EmbeddingService()
    startup_metrics["embedding_service_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    bm25_index = bm25_index or BM25Index()
    startup_metrics["bm25_init_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Load persisted BM25 index if present
    _maybe_load_bm25_index(bm25_index, bm25_index_path)
    startup_metrics["bm25_load_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Dense service: accept injected vector store
    dense_service = dense_service or DenseRetrievalService(vector_store_service=vector_store_service, embedding_service=embedding_service)
    startup_metrics["dense_service_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Sparse service: accept injected BM25 index
    sparse_service = sparse_service or SparseRetrievalService(index=bm25_index)
    startup_metrics["sparse_service_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Hybrid service: accept injected dense/sparse
    hybrid_service = hybrid_service or HybridRetrievalService(
        dense_retrieval_service=dense_service,
        sparse_retrieval_service=sparse_service,
    )
    startup_metrics["hybrid_service_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Reranker: singleton, shared across searches
    reranker = reranker or CrossEncoderReranker()
    startup_metrics["reranker_init_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    # Preload the models once during bundle creation so every search stays fast.
    # Both services are singletons owned by the cached bundle, so the model is
    # never re-instantiated during a search.
    logger.info("Preloading embedding model...")
    embedding_service.warmup()
    startup_metrics["embedding_model_load_ms"] = (time.perf_counter() - stage_start) * 1000
    stage_start = time.perf_counter()

    logger.info("Preloading cross-encoder reranker...")
    reranker.load()
    startup_metrics["cross_encoder_load_ms"] = (time.perf_counter() - stage_start) * 1000

    bundle = RetrievalBundle(
        vector_store_service=vector_store_service,
        embedding_service=embedding_service,
        bm25_index=bm25_index,
        dense_service=dense_service,
        sparse_service=sparse_service,
        hybrid_service=hybrid_service,
        reranker=reranker,
        startup_metrics=startup_metrics,
    )

    startup_seconds = time.perf_counter() - bundle_start
    startup_metrics["total_ms"] = startup_seconds * 1000
    logger.info("Retrieval bundle ready in %.2fs", startup_seconds)
    print(f"[BOOTSTRAP-TRACE][composition_root.py] Bundle ready in {startup_seconds:.2f}s")

    # Store as singleton for subsequent calls
    _retrieval_bundle_singleton = bundle

    return bundle

