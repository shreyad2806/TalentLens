import time
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnostics")

def run_diagnostics():
    print("--- 1. Embedding Model Diagnostics ---")
    try:
        from src.embeddings.embedding_service import EmbeddingService
        start = time.time()
        embed_service = EmbeddingService()
        load_time = time.time() - start
        print(f"EmbeddingService initialization time: {load_time:.2f}s")
        print("Checking if model caching works... Initializing again:")
        start = time.time()
        embed_service2 = EmbeddingService()
        load_time2 = time.time() - start
        print(f"EmbeddingService second initialization time: {load_time2:.2f}s")
    except Exception as e:
        print(f"Error in embedding diagnostic: {e}")

    print("\n--- 3. BM25 Diagnostics ---")
    try:
        from src.retrieval.sparse.bm25_index import BM25Index
        from src.retrieval.sparse.sparse_retrieval_service import SparseRetrievalService
        
        # Check if there is a way to load index
        index = BM25Index()
        print(f"BM25Index initialized. Total documents: {index.total_documents}")
        print(f"BM25Index vocabulary size: {len(index.vocabulary)}")
        
        # We initialized a fresh BM25Index in app.py. Did we ever build it?
        print("If index is empty, BM25 returns 0 candidates.")
    except Exception as e:
        print(f"Error in BM25 diagnostic: {e}")

    print("\n--- 4. Vector Store Diagnostics ---")
    try:
        from src.vector_store import VectorStoreService
        vs_service = VectorStoreService()
        
        # Try to get stats
        if hasattr(vs_service, "get_stats"):
            stats = vs_service.get_stats()
            print(f"Vector Store Stats: {stats}")
        else:
            print("VectorStoreService has no get_stats method.")
            
        print("Checking for candidate chunks in vector store...")
        # Since we can't easily count, let's see if querying an empty vector returns anything
        dummy_vector = [0.1] * vs_service.config.dimension
        res = vs_service.query(dummy_vector, k=1)
        print(f"Querying dummy vector returned {len(res)} results.")
        if res:
            print(f"Sample result: {res[0]}")
    except Exception as e:
        print(f"Error in Vector Store diagnostic: {e}")

    print("\n--- 5. Metadata Filtering Diagnostics ---")
    try:
        from src.retrieval.metadata.schema import MetadataFilter
        from pydantic import ValidationError
        print("Checking filter schema...")
        m_filter = MetadataFilter(minimum_experience=2, maximum_experience=6, location="India", skills=["python"])
        print(f"Valid MetadataFilter: {m_filter.model_dump(exclude_none=True)}")
    except ValidationError as e:
        print(f"ValidationError in MetadataFilter: {e}")
    except Exception as e:
        print(f"Error in Metadata Filtering diagnostic: {e}")

    print("\n--- 6. Search Pipeline Diagnostics ---")
    try:
        from app import get_metadata_service, get_hybrid_service, get_reranker_service
        
        metadata_service = get_metadata_service()
        hybrid_service = get_hybrid_service()
        reranker_service = get_reranker_service()
        
        query = "Python developer in India"
        filters = {"minimum_experience": 2, "maximum_experience": 6, "location": "India", "skills": ["python"]}
        
        print("\nStage 1: Metadata Filter setup")
        print(f"Filters applied: {filters}")
        
        print("\nStage 2 & 3 & 4: Hybrid Retrieval (Dense + BM25)")
        start = time.time()
        retrieved = hybrid_service.search(query=query, top_k=10, filters=filters)
        hybrid_time = time.time() - start
        print(f"Input query: '{query}'")
        print(f"Output count: {len(retrieved)}")
        print(f"Latency: {hybrid_time:.2f}s")
        
        if retrieved:
            print("\nStage 5: Reranker")
            start = time.time()
            reranked = reranker_service.rerank(query=query, candidates=retrieved, top_k=5)
            rerank_time = time.time() - start
            print(f"Input count: {len(retrieved)}")
            print(f"Output count: {len(reranked)}")
            print(f"Latency: {rerank_time:.2f}s")
        else:
            print("\nStage 5: Reranker skipped because Hybrid returned 0.")
            
    except Exception as e:
        print(f"Error in Search Pipeline diagnostic: {e}")

if __name__ == "__main__":
    run_diagnostics()
