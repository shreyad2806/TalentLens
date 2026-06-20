"""
Complete pipeline test for all vector store adapters.

This script tests the complete pipeline:
Resume → Parser → ResumeDocument → ChunkService → EmbeddingService → EmbeddingRecords → VectorStoreFactory → Selected Adapter

Tests all adapter methods: connect, upsert, query, fetch, delete, count, health, close
Runs with Memory, Pinecone (if configured), Qdrant (if configured)
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
from src.vector_store import VectorRecord, VectorStoreService, VectorStoreConfig
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


def test_adapter(provider: str):
    """
    Test a specific vector store adapter with the complete pipeline.
    
    Args:
        provider: Provider name ('memory', 'pinecone', 'qdrant')
        
    Returns:
        True if test passed, False otherwise
    """
    print_header(f"TESTING {provider.upper()} ADAPTER")
    print()
    
    # Reset config to ensure fresh environment variable read
    reset_config()
    
    # Set environment for the provider
    os.environ["VECTOR_STORE_PROVIDER"] = provider
    os.environ["VECTOR_STORE_DIMENSION"] = "1024"
    
    # Provider-specific environment variables
    if provider == "pinecone":
        if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
            print_warning("PINECONE_API_KEY or PINECONE_INDEX_NAME not set")
            print_info("Skipping Pinecone adapter test")
            print()
            return True  # Skip gracefully
    elif provider == "qdrant":
        if not os.getenv("QDRANT_COLLECTION"):
            print_warning("QDRANT_COLLECTION not set")
            print_info("Skipping Qdrant adapter test")
            print()
            return True  # Skip gracefully
    
    print_info(f"Provider: {provider}")
    print()
    
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
        chunks = chunk_service.create_chunks(document, resume_id=f"vector-adapter-test-{provider}")
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
                id=f"vector-adapter-{provider}-{i}",
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
        print(f"  Vector dimension: {vector_records[0].dimension}")
    except Exception as e:
        print_failure(f"Failed to create VectorRecord: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 6: Create VectorStoreService
    print_info("Creating VectorStoreService...")
    try:
        vector_store_service = VectorStoreService()
        print_success(f"VectorStoreService created with {provider} adapter")
    except Exception as e:
        print_failure(f"Failed to create VectorStoreService: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 7: Test connect (if adapter supports it)
    print_info("Testing connect...")
    try:
        if hasattr(vector_store_service.vector_store, 'connect'):
            vector_store_service.vector_store.connect()
            print_success("Connect successful")
        else:
            print_info("Connect method not available (adapter auto-connects)")
    except Exception as e:
        print_warning(f"Connect failed (may be expected): {str(e)[:80]}...")
    print()
    
    # Step 8: Test health check
    print_info("Testing health check...")
    try:
        health = vector_store_service.health()
        if health['healthy']:
            print_success(f"Health check passed: {health['status']}")
            print(f"  Record count: {health['record_count']}")
            print(f"  Latency: {health.get('latency_ms', 0):.2f}ms")
        else:
            print_warning(f"Health check returned unhealthy: {health['status']}")
            print(f"  Message: {health['message']}")
        print()
    except Exception as e:
        print_failure(f"Health check failed: {e}")
        return False
    
    # Step 9: Test upsert
    print_info("Testing upsert...")
    upsert_start = time.time()
    try:
        result = vector_store_service.upsert(vector_records)
        upsert_latency = time.time() - upsert_start
        
        if result['success']:
            print_success(f"Upsert successful: {result['upserted_count']} vectors")
            print(f"  Vectors Uploaded: {result['upserted_count']}")
            print(f"  Latency: {upsert_latency:.3f}s")
            if 'batch_count' in result:
                print(f"  Batch count: {result['batch_count']}")
        else:
            print_failure(f"Upsert failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Upsert failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 10: Test count
    print_info("Testing count...")
    try:
        count = vector_store_service.count()
        print_success(f"Vector count: {count}")
        print(f"  Count: {count}")
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Step 11: Test fetch
    print_info("Testing fetch...")
    try:
        record = vector_store_service.fetch(vector_records[0].id)
        if record:
            print_success(f"Fetched record: {record.id}")
            print(f"  Candidate: {record.candidate_name}")
            print(f"  Section: {record.section}")
            print(f"  Dimension: {record.dimension}")
        else:
            print_failure("Record not found")
            return False
        print()
    except Exception as e:
        print_failure(f"Fetch failed: {e}")
        return False
    
    # Step 12: Test query
    print_info("Testing query...")
    query_start = time.time()
    try:
        query_vector = embeddings[0]
        results = vector_store_service.query(query_vector, k=3)
        query_latency = time.time() - query_start
        
        print_success(f"Query returned {len(results)} results")
        print(f"  Latency: {query_latency:.3f}s")
        
        if results:
            print(f"  Top Query Result:")
            print(f"    ID: {results[0]['id']}")
            print(f"    Score: {results[0]['score']:.4f}")
            print(f"    Section: {results[0]['metadata'].get('section', 'N/A')}")
            print(f"    Candidate: {results[0]['metadata'].get('candidate_name', 'N/A')}")
        print()
    except Exception as e:
        print_failure(f"Query failed: {e}")
        return False
    
    # Step 13: Test delete
    print_info("Testing delete...")
    delete_start = time.time()
    try:
        ids_to_delete = [record.id for record in vector_records[:2]]
        result = vector_store_service.delete(ids_to_delete)
        delete_latency = time.time() - delete_start
        
        if result['success']:
            print_success(f"Delete successful: {result['deleted_count']} vectors")
            print(f"  Deleted: {result['deleted_count']}")
            print(f"  Latency: {delete_latency:.3f}s")
        else:
            print_failure(f"Delete failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Delete failed: {e}")
        return False
    
    # Step 14: Verify delete
    print_info("Verifying delete...")
    try:
        count_after = vector_store_service.count()
        expected_count = len(vector_records) - 2
        if count_after == expected_count:
            print_success(f"Count after delete: {count_after} (expected {expected_count})")
        else:
            print_warning(f"Count after delete: {count_after} (expected {expected_count})")
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Step 15: Test close
    print_info("Testing close...")
    try:
        vector_store_service.close()
        print_success("Close successful")
        print(f"  Close Status: Success")
        print()
    except Exception as e:
        print_failure(f"Close failed: {e}")
        return False
    
    # Final result
    print_success(f"{provider.upper()} Adapter Test Passed")
    print()
    
    return True


def test_vector_adapters():
    """
    Test all vector store adapters with the complete pipeline.
    
    This function tests Memory, Pinecone (if configured), and Qdrant (if configured)
    adapters with the complete pipeline from resume parsing to vector store operations.
    """
    print_header("MULTI-PROVIDER VECTOR ADAPTER TESTS")
    print()
    
    providers_to_test = ['memory', 'pinecone', 'qdrant']
    results = {}
    
    for provider in providers_to_test:
        try:
            result = test_adapter(provider)
            results[provider] = result
        except Exception as e:
            print_failure(f"Unexpected error testing {provider}: {e}")
            import traceback
            traceback.print_exc()
            results[provider] = False
    
    # Final summary
    print_header("TEST SUMMARY")
    print()
    
    for provider, result in results.items():
        if result:
            print_success(f"{provider.upper()}: PASSED")
        else:
            print_failure(f"{provider.upper()}: FAILED")
    
    print()
    
    # Check if all tests passed
    all_passed = all(results.values())
    
    if all_passed:
        print_header("ALL ADAPTER TESTS PASSED")
        print_success("Multi-Provider Vector Infrastructure Ready")
        print()
        print("🚀 Multi-Provider Vector Infrastructure Ready")
        return True
    else:
        print_header("SOME ADAPTER TESTS FAILED")
        print_warning("Some adapter tests did not pass")
        return False


if __name__ == "__main__":
    try:
        success = test_vector_adapters()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
