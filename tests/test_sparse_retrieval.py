"""
BM25 Sparse Retrieval Service Test.

This script tests the complete BM25 sparse retrieval pipeline from resume parsing
to sparse retrieval service. It validates all components and prints detailed results.
"""

import sys
from pathlib import Path
import logging
import os
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
from src.retrieval.sparse import (
    SparseRetrievalService,
    IndexBuilder,
    Tokenizer,
    BM25Index
)
from src.vector_store.config import reset_config


def print_success(message: str):
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str):
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str):
    print(f"\033[94mℹ️  {message}\033[0m")


def print_header(message: str):
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def test_sparse_retrieval():
    """Test the complete BM25 sparse retrieval pipeline."""
    
    print_header("BM25 SPARSE RETRIEVAL SERVICE TEST")
    print()
    
    # Step 1: Load sample resume
    print_info("Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure("Sample resume not found")
        return False
    
    print_success(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Parse resume
    print_info("Parsing resume with ParserService...")
    try:
        parser = ParserService()
        parsed_resume = parser.parse_file(str(sample_resume_path))
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
        chunks = chunk_service.create_chunks(parsed_resume)
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Validate tokenization
    print_info("Validating tokenization...")
    try:
        tokenizer = Tokenizer()
        test_text = chunks[0].text if chunks else "Test text"
        tokens = tokenizer.tokenize_document(test_text)
        print_success(f"Tokenization validated: {len(tokens)} tokens")
    except Exception as e:
        print_failure(f"Tokenization validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 5: Build BM25 index
    print_info("Building BM25 index with IndexBuilder...")
    try:
        builder = IndexBuilder(tokenizer=tokenizer)
        index = builder.build_index(chunks)
        
        stats = index.get_statistics()
        print_success("BM25 index built successfully")
        print(f"  Vocabulary Size: {stats.vocabulary_size}")
        print(f"  Document Count: {stats.num_documents}")
        print(f"  Average Document Length: {stats.average_document_length:.2f}")
        print(f"  Total Tokens: {stats.total_tokens}")
    except Exception as e:
        print_failure(f"Failed to build BM25 index: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 6: Create SparseRetrievalService
    print_info("Creating SparseRetrievalService...")
    try:
        sparse_service = SparseRetrievalService(index=index, cache_enabled=True)
        print_success("SparseRetrievalService created successfully")
    except Exception as e:
        print_failure(f"Failed to create SparseRetrievalService: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 7: Run queries
    print_header("QUERY TESTS")
    print()
    
    queries = ["Python", "FastAPI", "Docker", "AWS", "Machine Learning", "React", "Kubernetes", "LangChain", "Pinecone"]
    
    for query in queries:
        print_info(f"Querying for: '{query}'")
        
        try:
            start_time = time.time()
            results = sparse_service.search(query, top_k=3)
            query_latency = time.time() - start_time
            
            print_success(f"Query completed in {query_latency:.3f}s")
            print(f"  Results: {len(results)}")
            
            for result in results:
                print(f"    - Candidate: {result.candidate_name}")
                print(f"      Chunk: {result.chunk_id}")
                print(f"      Section: {result.section}")
                print(f"      BM25 Score: {result.bm25_score:.4f}")
                print(f"      Matched Terms: {result.matched_terms}")
                print(f"      Rank: {result.rank}")
                print(f"      Matched Text: {result.matched_text[:100]}...")
            
        except Exception as e:
            print_failure(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
    
    # Step 8: Test cache
    print_header("CACHE TEST")
    print()
    
    print_info("Running identical query twice to test cache...")
    sparse_service.clear_cache()
    
    test_query = "Python"
    
    print_info(f"First query: '{test_query}'")
    start_time = time.time()
    results1 = sparse_service.search(test_query, top_k=3)
    first_latency = time.time() - start_time
    cache_stats1 = sparse_service.get_cache_stats()
    print_success(f"First query completed in {first_latency:.3f}s")
    print(f"  Cache hits: {cache_stats1['query_cache']['hits']}")
    print(f"  Cache misses: {cache_stats1['query_cache']['misses']}")
    print()
    
    print_info(f"Second query (cached): '{test_query}'")
    start_time = time.time()
    results2 = sparse_service.search(test_query, top_k=3)
    second_latency = time.time() - start_time
    cache_stats2 = sparse_service.get_cache_stats()
    print_success(f"Second query completed in {second_latency:.3f}s")
    print(f"  Cache hits: {cache_stats2['query_cache']['hits']}")
    print(f"  Cache misses: {cache_stats2['query_cache']['misses']}")
    
    if cache_stats2['query_cache']['hits'] > cache_stats1['query_cache']['hits']:
        print_success("Cache hit verified")
    else:
        print_failure("Cache hit not verified")
        return False
    
    print()
    
    # Step 9: Test incremental operations
    print_header("INCREMENTAL OPERATIONS TEST")
    print()
    
    # Test document add
    print_info("Testing document add...")
    try:
        if chunks:
            new_index = BM25Index()
            builder.add_document(new_index, chunks[0])
            stats = new_index.get_statistics()
            print_success(f"Document add successful: {stats.num_documents} documents")
        else:
            print_failure("No chunks available for add test")
    except Exception as e:
        print_failure(f"Document add failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test document delete
    print_info("Testing document delete...")
    try:
        if chunks:
            chunk_id = str(chunks[0].chunk_id)
            builder.remove_document(new_index, chunk_id)
            stats = new_index.get_statistics()
            print_success(f"Document delete successful: {stats.num_documents} documents")
        else:
            print_failure("No chunks available for delete test")
    except Exception as e:
        print_failure(f"Document delete failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test document update
    print_info("Testing document update...")
    try:
        if chunks:
            builder.add_document(new_index, chunks[0])
            builder.update_document(new_index, chunks[0])
            stats = new_index.get_statistics()
            print_success(f"Document update successful: {stats.num_documents} documents")
        else:
            print_failure("No chunks available for update test")
    except Exception as e:
        print_failure(f"Document update failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test index rebuild
    print_info("Testing index rebuild...")
    try:
        builder.rebuild_index(new_index, chunks)
        stats = new_index.get_statistics()
        print_success(f"Index rebuild successful: {stats.num_documents} documents")
    except Exception as e:
        print_failure(f"Index rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 10: Print final statistics
    print_header("FINAL STATISTICS")
    print()
    
    stats = index.get_statistics()
    print(f"Vocabulary Size: {stats.vocabulary_size}")
    print(f"Document Count: {stats.num_documents}")
    print(f"Average Document Length: {stats.average_document_length:.2f}")
    print(f"Total Tokens: {stats.total_tokens}")
    
    cache_stats = sparse_service.get_cache_stats()
    print(f"Cache Hits: {cache_stats['query_cache']['hits']}")
    print(f"Cache Misses: {cache_stats['query_cache']['misses']}")
    print(f"Cache Hit Rate: {cache_stats['query_cache']['hit_rate']:.2%}")
    print()
    
    # Step 11: Cleanup
    print_info("Cleaning up...")
    index.clear()
    print_success("Cleanup complete")
    print()
    
    print_header("TEST COMPLETE")
    print_success("✅ BM25 Sparse Retrieval Test Passed")
    print()
    print("🚀 Sparse Retrieval Layer Ready")
    
    return True


if __name__ == "__main__":
    success = test_sparse_retrieval()
    sys.exit(0 if success else 1)
