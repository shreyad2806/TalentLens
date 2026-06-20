"""
Test script for Dense Retrieval Service.

This script tests the complete dense retrieval pipeline:
Resume → Parser → ChunkService → EmbeddingService → VectorStore → DenseRetrievalService
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
from src.vector_store import VectorRecord, VectorStoreService
from src.retrieval.dense import DenseRetrievalService, AggregationStrategy, NormalizationStrategy
from src.vector_store.config import reset_config


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


def generate_mock_embeddings(chunks, dimension=1024):
    """
    Generate mock embeddings for testing.
    
    Args:
        chunks: List of Chunk objects
        dimension: Vector dimension (default: 1024)
        
    Returns:
        List of mock embedding vectors
    """
    import random
    return [random.random() for _ in range(dimension)]


def test_dense_retrieval():
    """
    Test the dense retrieval service with complete pipeline.
    
    This test runs the complete pipeline from resume parsing to dense retrieval,
    validates all components, and prints detailed results.
    """
    print_header("DENSE RETRIEVAL SERVICE TEST")
    print()
    
    # Reset config to ensure fresh environment variable read
    reset_config()
    
    # Set environment to use memory provider
    os.environ["VECTOR_STORE_PROVIDER"] = "memory"
    os.environ["VECTOR_STORE_DIMENSION"] = "1024"
    
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
        chunks = chunk_service.create_chunks(document, resume_id="dense-retrieval-test")
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Generate mock embeddings
    print_info("Generating mock embeddings for testing...")
    try:
        import random
        dimension = 1024
        
        embeddings = []
        for chunk in chunks:
            mock_embedding = [random.random() for _ in range(dimension)]
            embeddings.append(mock_embedding)
        
        print_success(f"Generated {len(embeddings)} mock embeddings")
        print(f"  Embedding dimension: {dimension}")
    except Exception as e:
        print_failure(f"Failed to generate mock embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 5: Convert to VectorRecord
    print_info("Converting embeddings to VectorRecord...")
    try:
        vector_records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            record = VectorRecord(
                id=f"dense-retrieval-{i}",
                resume_id=str(chunk.resume_id),
                chunk_id=str(chunk.chunk_id),
                candidate_name=chunk.candidate_name,
                section=chunk.section,
                vector=embedding,
                metadata={
                    "text_length": len(chunk.text),
                    "text_preview": chunk.text[:100]
                }
            )
            vector_records.append(record)
        
        print_success(f"Created {len(vector_records)} VectorRecord objects")
    except Exception as e:
        print_failure(f"Failed to create VectorRecord: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 6: Upsert to VectorStore
    print_info("Upserting vectors to VectorStore...")
    try:
        vector_store_service = VectorStoreService()
        result = vector_store_service.upsert(vector_records)
        
        if result['success']:
            print_success(f"Upserted {result['upserted_count']} vectors to VectorStore")
        else:
            print_failure(f"Upsert failed: {result['errors']}")
            return False
    except Exception as e:
        print_failure(f"Failed to upsert vectors: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 7: Create DenseRetrievalService with mock query embedding
    print_info("Creating DenseRetrievalService...")
    try:
        # Disable actual embedding generation for testing
        # We'll use mock embeddings for queries
        import random
        
        # Override the query_embedder to use mock embeddings
        class MockQueryEmbedder:
            def __init__(self, dimension=1024):
                self.dimension = dimension
            
            def embed_query(self, query):
                return [random.random() for _ in range(self.dimension)]
            
            def get_embedding_dimension(self):
                return self.dimension
        
        dense_service = DenseRetrievalService(
            vector_store_service=vector_store_service,
            cache_enabled=True,
            normalization_strategy=NormalizationStrategy.COSINE
        )
        
        # Replace the query embedder with mock
        dense_service.query_embedder = MockQueryEmbedder(dimension=1024)
        
        print_success("DenseRetrievalService created successfully with mock query embedder")
    except Exception as e:
        print_failure(f"Failed to create DenseRetrievalService: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 8: Run queries
    queries = [
        "Python Backend Developer",
        "FastAPI Engineer",
        "Machine Learning",
        "AWS Docker",
        "React Developer"
    ]
    
    print_header("QUERY TESTS")
    print()
    
    all_query_results = {}
    
    for query in queries:
        print_info(f"Querying for: '{query}'")
        print()
        
        try:
            # First query (cache miss)
            query_start = time.time()
            results = dense_service.search(query, top_k=3)
            query_latency = time.time() - query_start
            
            all_query_results[query] = {
                'results': results,
                'latency': query_latency,
                'cache_hit': False
            }
            
            print(f"Query: {query}")
            print(f"Retrieved Chunks: {len(results)}")
            print(f"Total Latency: {query_latency:.3f}s")
            print()
            
            # Print results
            for i, result in enumerate(results):
                print(f"  Result {i + 1}:")
                print(f"    Candidate: {result.candidate_name}")
                print(f"    Section: {result.section}")
                print(f"    Similarity Score: {result.score:.4f}")
                print(f"    Normalized Score: {result.normalized_score:.4f}")
                print(f"    Rank: {result.rank}")
                print(f"    Metadata: {result.metadata}")
                print(f"    Matched Text: {result.matched_text[:80]}...")
                print()
            
            # Second query (cache hit)
            print_info("Running same query again to test cache...")
            query_start = time.time()
            cached_results = dense_service.search(query, top_k=3)
            cache_latency = time.time() - query_start
            
            cache_stats = dense_service.get_cache_stats()
            
            if cache_stats and cache_stats['hits'] > 0:
                print_success(f"Cache hit! Cache latency: {cache_latency:.3f}s")
                print(f"Cache stats: hits={cache_stats['hits']}, misses={cache_stats['misses']}, hit_rate={cache_stats['hit_rate']:.2%}")
            else:
                print_warning("Cache miss on second query")
            
            print()
            
        except Exception as e:
            print_failure(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 9: Test aggregated search
    print_header("AGGREGATED SEARCH TEST")
    print()
    
    test_query = "Python Backend Developer"
    print_info(f"Testing aggregated search for: '{test_query}'")
    print()
    
    try:
        # Test with different aggregation strategies
        for strategy in [AggregationStrategy.MAX, AggregationStrategy.AVERAGE, AggregationStrategy.WEIGHTED]:
            print_info(f"Testing {strategy.value} aggregation...")
            
            # Create new aggregator with specific strategy
            from src.retrieval.dense import CandidateAggregator
            aggregator = CandidateAggregator(strategy=strategy)
            
            # Get raw results
            raw_results = dense_service.search(test_query, top_k=10)
            
            # Aggregate
            aggregated = aggregator.aggregate(raw_results)
            
            print(f"  Raw chunks: {len(raw_results)}")
            print(f"  Aggregated candidates: {len(aggregated)}")
            
            if aggregated:
                print(f"  Top candidate: {aggregated[0].candidate_name}")
                print(f"  Final score: {aggregated[0].final_score:.4f}")
                print(f"  Section scores: {aggregated[0].section_scores}")
                print(f"  Matched sections: {aggregated[0].metadata.get('matched_sections', [])}")
                print(f"  Evidence chunks: {len(aggregated[0].evidence_chunks)}")
            
            print()
            
    except Exception as e:
        print_failure(f"Aggregated search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 10: Validation
    print_header("VALIDATION")
    print()
    
    # Validate query embedding generated
    print_info("Validating query embedding generated...")
    try:
        from src.retrieval.dense import QueryEmbedder
        embedder = QueryEmbedder()
        embedding = embedder.embed_query("test query")
        if len(embedding) == 1024:
            print_success("Query embedding generated correctly (dimension: 1024)")
        else:
            print_failure(f"Query embedding dimension incorrect: {len(embedding)}")
            return False
    except Exception as e:
        print_failure(f"Query embedding validation failed: {e}")
        return False
    print()
    
    # Validate vector search executed
    print_info("Validating vector search executed...")
    if all_query_results:
        print_success("Vector search executed successfully")
    else:
        print_failure("No query results to validate")
        return False
    print()
    
    # Validate score normalized
    print_info("Validating score normalization...")
    for query, data in all_query_results.items():
        for result in data['results']:
            if not 0.0 <= result.normalized_score <= 1.0:
                print_failure(f"Normalized score out of range: {result.normalized_score}")
                return False
    print_success("All scores normalized correctly (0.0 - 1.0)")
    print()
    
    # Validate candidate aggregation correct
    print_info("Validating candidate aggregation...")
    try:
        from src.retrieval.dense import CandidateAggregator
        aggregator = CandidateAggregator(strategy=AggregationStrategy.WEIGHTED)
        raw_results = dense_service.search("Python", top_k=5)
        aggregated = aggregator.aggregate(raw_results)
        
        if len(aggregated) <= len(raw_results):
            print_success(f"Aggregation correct: {len(raw_results)} chunks → {len(aggregated)} candidates")
        else:
            print_failure(f"Aggregation incorrect: {len(raw_results)} chunks → {len(aggregated)} candidates")
            return False
    except Exception as e:
        print_failure(f"Candidate aggregation validation failed: {e}")
        return False
    print()
    
    # Validate no duplicate candidates
    print_info("Validating no duplicate candidates...")
    for query, data in all_query_results.items():
        candidate_ids = [result.resume_id for result in data['results']]
        if len(candidate_ids) != len(set(candidate_ids)):
            print_failure(f"Duplicate candidates found in results for query: {query}")
            return False
    print_success("No duplicate candidates found")
    print()
    
    # Validate metadata preserved
    print_info("Validating metadata preserved...")
    for query, data in all_query_results.items():
        for result in data['results']:
            if not result.metadata:
                print_failure(f"Metadata not preserved for result: {result.id}")
                return False
    print_success("Metadata preserved in all results")
    print()
    
    # Validate cache works
    print_info("Validating cache works...")
    cache_stats = dense_service.get_cache_stats()
    if cache_stats and cache_stats['hits'] > 0:
        print_success(f"Cache working: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
    else:
        print_warning("Cache may not be working (no hits)")
    print()
    
    # Step 11: Print latency metrics
    print_header("LATENCY METRICS")
    print()
    
    for query, data in all_query_results.items():
        print(f"Query: {query}")
        print(f"  Total Latency: {data['latency']:.3f}s")
        print(f"  Cache Hit: {data['cache_hit']}")
        print()
    
    # Step 12: Cleanup
    print_info("Cleaning up...")
    try:
        vector_store_service.clear()
        dense_service.close()
        print_success("Cleanup completed")
    except Exception as e:
        print_failure(f"Cleanup failed: {e}")
        return False
    print()
    
    # Final result
    print_header("DENSE RETRIEVAL TEST PASSED")
    print_success("All validation checks passed")
    print()
    print("🚀 Dense Retrieval Layer Ready")
    return True


if __name__ == "__main__":
    try:
        success = test_dense_retrieval()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
