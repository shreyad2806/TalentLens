"""Test BM25 retrieval integration."""

from src.retrieval.sparse import SparseRetrievalService, BM25Index
from pathlib import Path

# Load BM25 index from persistent storage
bm25_index_path = Path("data/indexes/bm25")
if (bm25_index_path / "metadata.json").exists():
    print("Loading BM25 index from persistent storage...")
    bm25_index = BM25Index()
    bm25_index.load_from_disk(bm25_index_path)
    print(f"BM25 loaded: {bm25_index.num_documents} documents")
    print(f"BM25 vocabulary size: {len(bm25_index.vocabulary)}")
    print(f"BM25 avg doc length: {bm25_index.avg_doc_length:.2f}")
    
    # Create retrieval service
    sparse_service = SparseRetrievalService(index=bm25_index)
    
    # Test search
    print("\nTesting search...")
    results = sparse_service.search("Python developer", top_k=5)
    print(f"Search returned {len(results)} results")
    
    if results:
        print("\nTop result:")
        print(f"  Candidate: {results[0].candidate_name}")
        print(f"  Score: {results[0].bm25_score:.4f}")
        print(f"  Matched terms: {results[0].matched_terms}")
    else:
        print("No results returned - index may be empty")
else:
    print("BM25 index not found at data/indexes/bm25")
