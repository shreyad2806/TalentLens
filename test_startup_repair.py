"""Verify the repaired startup path builds indexes correctly."""
import sys
import os
sys.path.insert(0, '.')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

print("=" * 80)
print("TEST: BootstrapService builds indexes and populates BM25 + vector store")
print("=" * 80)

from src.bootstrap.bootstrap_service import BootstrapService
from src.bootstrap.composition_root import create_retrieval_bundle

# Create BootstrapService (this now uses the singleton retrieval bundle)
bs = BootstrapService(verbose=True)
print("\n[BootstrapService instantiated successfully]")

# Run bootstrap (should build indexes since we deleted stale files)
result = bs.bootstrap()
print(f"\n[Bootstrap result: bootstrapped={result.get('bootstrapped')}, reason={result.get('reason')}, success={result.get('success')}]")

# Check final stats
final_stats = bs.indexing_pipeline.get_statistics()
print(f"\nFinal indexing stats:")
print(f"  indexed_documents = {final_stats['indexed_documents']}")
print(f"  vector_count      = {final_stats['vector_count']}")
print(f"  bm25_count        = {final_stats['bm25_count']}")

# Verify the singleton retrieval bundle sees the same populated BM25 index
bundle = create_retrieval_bundle()
bm25_stats = bundle.bm25_index.get_statistics()
print(f"\nSingleton retrieval bundle BM25 docs: {bm25_stats.num_documents}")
print(f"Singleton retrieval bundle vector count: {bundle.vector_store_service.count()}")

# Assertions
assert final_stats['bm25_count'] > 0, "BM25 index is still empty"
assert final_stats['vector_count'] > 0, "Vector store is still empty"
assert final_stats['indexed_documents'] > 0, "No documents indexed"
assert bm25_stats.num_documents == final_stats['bm25_count'], "Singleton bundle sees different BM25 count"

print("\n" + "=" * 80)
print("ALL ASSERTIONS PASSED - Startup repair works correctly")
print("=" * 80)
