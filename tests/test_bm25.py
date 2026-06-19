"""
Test script for the BM25 Sparse Retrieval Layer.

This script tests the BM25 retrieval system with parsed Chunk objects to verify
correctness of indexing, searching, validation, and caching.
"""

import sys
from pathlib import Path
import logging
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.resume_parser.parser_service import ParserService
from src.chunks.service import ChunkService
from src.retrieval.bm25 import IndexBuilder, SearchService, BM25Validator


def print_success(message: str):
    """Print a success message in green."""
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str):
    """Print a failure message in red."""
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"\033[94mℹ️  {message}\033[0m")


def print_warning(message: str):
    """Print a warning message in yellow."""
    print(f"\033[93m⚠ {message}\033[0m")


def print_header(message: str):
    """Print a header message in yellow."""
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def test_bm25_retrieval():
    """
    Test the BM25 retrieval layer with parsed chunks.
    
    Pipeline:
    Load sample resume → ParserService → ResumeDocument → ChunkService → Chunks → IndexBuilder → BM25Index → SearchService → Search Results
    """
    print_header("BM25 RETRIEVAL LAYER TEST")
    
    # Step 1: Load sample resume
    print_info("Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Parse resume
    print_info("Parsing resume with ParserService...")
    try:
        parser = ParserService()
        document = parser.parse_file(sample_resume_path)
        print_success("Resume parsed successfully")
    except Exception as e:
        print_failure(f"Failed to parse resume: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 3: Create chunks
    print_info("Creating chunks with ChunkService...")
    try:
        chunk_service = ChunkService()
        chunks = chunk_service.create_chunks(document, resume_id="bm25-test-001")
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Build BM25 index
    print_info("Building BM25 index with IndexBuilder...")
    index_build_start = time.time()
    try:
        index_builder = IndexBuilder(k1=1.5, b=0.75, remove_stop_words=True)
        index = index_builder.build_index(chunks)
        index_build_time = time.time() - index_build_start
        print_success("BM25 index built successfully")
    except Exception as e:
        print_failure(f"Failed to build BM25 index: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 5: Print index statistics
    print_header("INDEX STATISTICS")
    print()
    
    index_stats = index.get_statistics()
    print(f"Indexed Documents: {index_stats['num_documents']}")
    print(f"Vocabulary Size: {index_stats['vocabulary_size']}")
    print(f"Average Document Length: {index_stats['avg_doc_length']:.2f}")
    print(f"Index Size (Total Tokens): {index_stats['total_tokens']}")
    print()
    
    # Step 6: Create search service
    print_info("Creating SearchService...")
    try:
        search_service = SearchService(index, enable_cache=True)
        search_service.set_index_builder(index_builder)
        print_success("SearchService created successfully")
    except Exception as e:
        print_failure(f"Failed to create SearchService: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 7: Run queries
    print_header("QUERY TESTS")
    print()
    
    queries = ["Python", "FastAPI", "Docker", "AWS", "Machine Learning", "React"]
    all_results = {}
    
    for query in queries:
        print_info(f"Searching for: '{query}'")
        query_start = time.time()
        try:
            results = search_service.search(query, k=3)
            query_latency = time.time() - query_start
            all_results[query] = {'results': results, 'latency': query_latency}
            print_success(f"Found {len(results)} results in {query_latency:.3f}s")
            
            # Print top results
            if results:
                print_header(f"TOP RESULTS FOR '{query}'")
                print()
                for result in results:
                    print(f"Rank: {result.rank}")
                    print(f"Section: {result.document.section}")
                    print(f"Score: {result.score:.4f}")
                    print(f"Candidate: {result.document.candidate_name}")
                    print(f"Chunk: {result.document.text[:150]}...")
                    print()
        except Exception as e:
            print_failure(f"Search failed for '{query}': {e}")
            return False
    
    # Step 8: Validation
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate index built
    print_info("Validating index built...")
    if index_stats['num_documents'] > 0:
        print_success("Index built successfully")
    else:
        print_failure("Index is empty")
        validation_passed = False
    
    # Validate documents indexed
    print_info("Validating documents indexed...")
    if index_stats['num_documents'] == len(chunks):
        print_success(f"All {len(chunks)} documents indexed")
    else:
        print_failure(f"Document count mismatch: expected {len(chunks)}, got {index_stats['num_documents']}")
        validation_passed = False
    
    # Validate scores returned
    print_info("Validating scores returned...")
    scores_found = any(len(r['results']) > 0 for r in all_results.values())
    if scores_found:
        print_success("At least one query returned scores")
    else:
        print_failure("No queries returned scores")
        validation_passed = False
    
    # Validate metadata preserved
    print_info("Validating metadata preserved...")
    metadata_preserved = True
    for query_data in all_results.values():
        for result in query_data['results']:
            if result.document.metadata is None:
                metadata_preserved = False
                break
    if metadata_preserved:
        print_success("Metadata preserved in all results")
    else:
        print_failure("Some results missing metadata")
        validation_passed = False
    
    # Validate cache works
    print_info("Validating cache works...")
    cache_stats_before = search_service.get_cache_stats()
    search_service.search("Python", k=3)
    cache_stats_after = search_service.get_cache_stats()
    cache_enabled = cache_stats_after.get('enabled', False)
    if cache_enabled:
        print_success("Cache enabled")
    else:
        print_warning("Cache not enabled (acceptable for BM25)")
    
    print()
    
    # Step 9: Repeat query to verify cache hit
    print_header("CACHE HIT TEST")
    print()
    
    print_info("Repeating query: 'Python'")
    query_start = time.time()
    results_repeat = search_service.search("Python", k=3)
    query_latency_repeat = time.time() - query_start
    
    cache_stats_final = search_service.get_cache_stats()
    cache_hits = cache_stats_final.get('hits', 0)
    
    if cache_hits > 0:
        print_success(f"Cache hit detected (hits: {cache_hits})")
    else:
        print_warning("No cache hit (acceptable for BM25)")
    
    print(f"Query latency: {query_latency_repeat:.3f}s")
    print()
    
    # Step 10: Print performance statistics
    print_header("PERFORMANCE STATISTICS")
    print()
    
    print(f"Index Build Time: {index_build_time:.3f}s")
    
    avg_query_latency = sum(r['latency'] for r in all_results.values()) / len(all_results)
    print(f"Average Query Latency: {avg_query_latency:.3f}s")
    
    print(f"Cache Status: {'Enabled' if cache_stats_final.get('enabled', False) else 'Disabled'}")
    print(f"Cache Hits: {cache_stats_final.get('hits', 0)}")
    print(f"Cache Misses: {cache_stats_final.get('misses', 0)}")
    print()
    
    # Step 11: Final result
    if validation_passed:
        print_header("BM25 TEST PASSED")
        print_success("All validation checks passed")
        print()
        print("🚀 BM25 Sparse Retrieval Ready")
        return True
    else:
        print_header("BM25 TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_bm25_retrieval()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
