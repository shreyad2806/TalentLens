"""
End-to-end test for the Vector Store Abstraction Layer.

This script tests the complete pipeline:
Parser → ResumeDocument → ChunkService → EmbeddingService → VectorRecord → VectorStoreService → Memory Adapter
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


def print_success(message: str):
    """Print a success message in green."""
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str):
    """Print a failure message in red."""
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"\033[94mℹ️  {message}\033[0m")


def print_header(message: str):
    """Print a header message in yellow."""
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def test_vector_store():
    """
    End-to-end test for the vector store abstraction.
    
    Pipeline:
    Load resume → Parser → ChunkService → EmbeddingService → VectorRecord → VectorStoreService → Memory Adapter
    """
    print_header("VECTOR STORE END-TO-END TEST")
    print()
    
    # Set environment for memory adapter
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
        chunks = chunk_service.create_chunks(document, resume_id="vector-store-test-001")
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Generate embeddings (use mock embeddings for testing)
    print_info("Generating mock embeddings for testing...")
    try:
        # Use mock embeddings instead of actual embedding service
        # This avoids requiring the model to be downloaded
        import random
        dimension = 1024
        
        embeddings = []
        for chunk in chunks:
            # Generate random embedding vector
            mock_embedding = [random.random() for _ in range(dimension)]
            
            # Create a simple embedding record object
            class MockEmbedding:
                def __init__(self, embedding):
                    self.embedding = embedding
            
            embeddings.append(MockEmbedding(mock_embedding))
        
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
                id=f"vector-record-{i}",
                resume_id=str(chunk.resume_id),
                chunk_id=str(chunk.chunk_id),
                candidate_name=chunk.candidate_name,
                section=chunk.section,
                vector=embedding.embedding,
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
    print_info("Creating VectorStoreService with Memory adapter...")
    try:
        vector_store_service = VectorStoreService()
        print_success("VectorStoreService created successfully")
    except Exception as e:
        print_failure(f"Failed to create VectorStoreService: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 7: Test health check
    print_header("HEALTH CHECK")
    print()
    print_info("Checking vector store health...")
    try:
        health = vector_store_service.health()
        if health['healthy']:
            print_success(f"Health check passed: {health['status']}")
            print(f"  Record count: {health['record_count']}")
        else:
            print_failure(f"Health check failed: {health['status']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Health check failed: {e}")
        return False
    
    # Step 8: Test upsert
    print_header("UPSERT TEST")
    print()
    print_info("Upserting vector records...")
    upsert_start = time.time()
    try:
        result = vector_store_service.upsert(vector_records)
        upsert_latency = time.time() - upsert_start
        
        if result['success']:
            print_success(f"Upserted {result['upserted_count']} vectors")
            print(f"  Upsert latency: {upsert_latency:.3f}s")
        else:
            print_failure(f"Upsert failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Upsert failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 9: Test count
    print_header("COUNT TEST")
    print()
    print_info("Counting vectors...")
    try:
        count = vector_store_service.count()
        print_success(f"Vector count: {count}")
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Step 10: Test fetch
    print_header("FETCH TEST")
    print()
    print_info("Fetching vector record...")
    try:
        record = vector_store_service.fetch(vector_records[0].id)
        if record:
            print_success(f"Fetched record: {record.id}")
            print(f"  Candidate: {record.candidate_name}")
            print(f"  Section: {record.section}")
            print(f"  Dimension: {record.dimension}")
            print(f"  Metadata: {record.metadata}")
        else:
            print_failure("Record not found")
            return False
        print()
    except Exception as e:
        print_failure(f"Fetch failed: {e}")
        return False
    
    # Step 11: Test query
    print_header("QUERY TEST")
    print()
    
    queries = ["Python", "FastAPI", "AWS", "Docker"]
    all_query_results = {}
    
    for query in queries:
        print_info(f"Querying for: '{query}'")
        query_start = time.time()
        try:
            # Generate mock embedding for query
            import random
            query_embedding = [random.random() for _ in range(1024)]
            
            # Query vector store
            results = vector_store_service.query(query_embedding, k=3)
            query_latency = time.time() - query_start
            
            all_query_results[query] = {'results': results, 'latency': query_latency}
            
            print_success(f"Query returned {len(results)} results in {query_latency:.3f}s")
            
            if results:
                print(f"  Top 3 results:")
                for i, result in enumerate(results):
                    print(f"    {i+1}. ID: {result['id']}")
                    print(f"       Score: {result['score']:.4f}")
                    print(f"       Section: {result['metadata'].get('section', 'N/A')}")
                    print(f"       Candidate: {result['metadata'].get('candidate_name', 'N/A')}")
            print()
        except Exception as e:
            print_failure(f"Query failed for '{query}': {e}")
            return False
    
    # Step 12: Validate results
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate ranking
    print_info("Validating ranking...")
    ranking_valid = True
    for query, data in all_query_results.items():
        results = data['results']
        if results:
            # Check if results are sorted by score descending
            scores = [r['score'] for r in results]
            if scores == sorted(scores, reverse=True):
                pass  # Ranking is correct
            else:
                print_warning(f"Ranking may be incorrect for query: {query}")
                ranking_valid = False
    
    if ranking_valid:
        print_success("Ranking validation passed")
    else:
        print_warning("Some queries may have incorrect ranking")
    print()
    
    # Validate metadata preserved
    print_info("Validating metadata preservation...")
    metadata_valid = True
    for query, data in all_query_results.items():
        for result in data['results']:
            metadata = result['metadata']
            if not metadata.get('candidate_name') or not metadata.get('section'):
                metadata_valid = False
                break
    
    if metadata_valid:
        print_success("Metadata preserved in all results")
    else:
        print_failure("Some results missing metadata")
        validation_passed = False
    print()
    
    # Validate vector dimensions preserved
    print_info("Validating vector dimensions preserved...")
    dimension_valid = True
    original_dimension = vector_records[0].dimension
    
    for record in vector_records:
        if record.dimension != original_dimension:
            dimension_valid = False
            break
    
    if dimension_valid:
        print_success(f"Vector dimensions preserved: {original_dimension}")
    else:
        print_failure("Vector dimensions not consistent")
        validation_passed = False
    print()
    
    # Step 13: Test delete
    print_header("DELETE TEST")
    print()
    print_info("Deleting vector records...")
    try:
        ids_to_delete = [record.id for record in vector_records[:2]]
        result = vector_store_service.delete(ids_to_delete)
        if result['success']:
            print_success(f"Deleted {result['deleted_count']} vectors")
        else:
            print_failure(f"Delete failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Delete failed: {e}")
        return False
    
    # Verify delete
    print_info("Verifying delete...")
    try:
        count_after = vector_store_service.count()
        expected_count = len(vector_records) - 2
        if count_after == expected_count:
            print_success(f"Count after delete: {count_after} (expected {expected_count})")
        else:
            print_failure(f"Count after delete: {count_after} (expected {expected_count})")
            validation_passed = False
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Step 14: Test clear
    print_header("CLEAR TEST")
    print()
    print_info("Clearing all vectors...")
    try:
        result = vector_store_service.vector_store.clear()
        if result['success']:
            print_success(f"Cleared {result['cleared_count']} vectors")
        else:
            print_failure(f"Clear failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Clear failed: {e}")
        return False
    
    # Verify clear
    print_info("Verifying clear...")
    try:
        count_after = vector_store_service.count()
        if count_after == 0:
            print_success(f"Count after clear: {count_after} (correct)")
        else:
            print_failure(f"Count after clear: {count_after} (expected 0)")
            validation_passed = False
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Step 15: Test close
    print_header("CLOSE TEST")
    print()
    print_info("Closing vector store...")
    try:
        vector_store_service.close()
        print_success("Vector store closed successfully")
        print()
    except Exception as e:
        print_failure(f"Close failed: {e}")
        return False
    
    # Step 16: Final result
    if validation_passed:
        print_header("VECTOR STORE TEST PASSED")
        print_success("All validation checks passed")
        print()
        print("🚀 Vector Store Abstraction Ready")
        return True
    else:
        print_header("VECTOR STORE TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_vector_store()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
