"""
Test script for the MemoryVectorStore adapter.

This script tests the MemoryVectorStore implementation to verify that it
correctly implements the VectorStore interface and performs all operations
as expected.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.vector_store import (
    VectorRecord,
    VectorStoreService,
    VectorStoreConfig,
    VectorStoreProvider
)


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


def test_memory_vector_store():
    """
    Test the MemoryVectorStore adapter.
    
    This test verifies that the MemoryVectorStore correctly implements
    all VectorStore methods and performs operations as expected.
    """
    print_header("MEMORY VECTOR STORE TEST")
    print()
    
    # Set environment to use memory provider
    import os
    os.environ["VECTOR_STORE_PROVIDER"] = "memory"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    # Create service
    print_info("Creating VectorStoreService with Memory adapter...")
    try:
        service = VectorStoreService()
        print_success("VectorStoreService created successfully")
        print()
    except Exception as e:
        print_failure(f"Failed to create service: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test health check
    print_header("HEALTH CHECK")
    print()
    print_info("Checking vector store health...")
    try:
        health = service.health()
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
    
    # Test upsert
    print_header("UPSERT TEST")
    print()
    print_info("Creating test records...")
    try:
        records = [
            VectorRecord(
                id="record-001",
                resume_id="resume-001",
                chunk_id="chunk-001",
                candidate_name="Alice Smith",
                section="skills",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={"skill": "python"}
            ),
            VectorRecord(
                id="record-002",
                resume_id="resume-001",
                chunk_id="chunk-002",
                candidate_name="Alice Smith",
                section="experience",
                vector=[0.2, 0.3, 0.4, 0.5],
                metadata={"role": "developer"}
            ),
            VectorRecord(
                id="record-003",
                resume_id="resume-002",
                chunk_id="chunk-003",
                candidate_name="Bob Jones",
                section="skills",
                vector=[0.3, 0.4, 0.5, 0.6],
                metadata={"skill": "java"}
            )
        ]
        
        result = service.upsert(records)
        if result['success']:
            print_success(f"Upserted {result['upserted_count']} records")
        else:
            print_failure(f"Upsert failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Upsert failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test count
    print_header("COUNT TEST")
    print()
    print_info("Counting records...")
    try:
        count = service.count()
        print_success(f"Count: {count} records")
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Test query
    print_header("QUERY TEST")
    print()
    print_info("Querying for similar vectors...")
    try:
        query_vector = [0.15, 0.25, 0.35, 0.45]
        results = service.query(query_vector, k=2)
        print_success(f"Query returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    ID: {result['id']}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Section: {result['metadata'].get('section', 'N/A')}")
        print()
    except Exception as e:
        print_failure(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test query with filters
    print_header("QUERY WITH FILTERS TEST")
    print()
    print_info("Querying with section filter...")
    try:
        results = service.query(query_vector, k=10, filters={"section": "skills"})
        print_success(f"Filtered query returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    ID: {result['id']}")
            print(f"    Section: {result['metadata'].get('section', 'N/A')}")
        print()
    except Exception as e:
        print_failure(f"Filtered query failed: {e}")
        return False
    
    # Test fetch
    print_header("FETCH TEST")
    print()
    print_info("Fetching record by ID...")
    try:
        record = service.fetch("record-001")
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
    
    # Test fetch non-existent
    print_info("Fetching non-existent record...")
    try:
        record = service.fetch("non-existent")
        if record is None:
            print_success("Correctly returned None for non-existent record")
        else:
            print_failure("Should have returned None")
            return False
        print()
    except Exception as e:
        print_failure(f"Fetch failed: {e}")
        return False
    
    # Test delete
    print_header("DELETE TEST")
    print()
    print_info("Deleting record...")
    try:
        result = service.delete(["record-003"])
        if result['success']:
            print_success(f"Deleted {result['deleted_count']} record")
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
        count_after = service.count()
        if count_after == 2:
            print_success(f"Count after delete: {count_after} (correct)")
        else:
            print_failure(f"Count after delete: {count_after} (expected 2)")
            return False
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Test delete_resume
    print_header("DELETE RESUME TEST")
    print()
    print_info("Deleting all records for resume-001...")
    try:
        result = service.vector_store.delete_resume("resume-001")
        if result['success']:
            print_success(f"Deleted {result['deleted_count']} records for resume")
        else:
            print_failure(f"Delete resume failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Delete resume failed: {e}")
        return False
    
    # Verify delete_resume
    print_info("Verifying delete_resume...")
    try:
        count_after = service.count()
        if count_after == 0:
            print_success(f"Count after delete_resume: {count_after} (correct)")
        else:
            print_failure(f"Count after delete_resume: {count_after} (expected 0)")
            return False
        print()
    except Exception as e:
        print_failure(f"Count failed: {e}")
        return False
    
    # Test clear
    print_header("CLEAR TEST")
    print()
    print_info("Clearing all records...")
    try:
        # First add some records
        service.upsert([
            VectorRecord(
                id="temp-001",
                resume_id="temp-resume",
                chunk_id="temp-chunk",
                candidate_name="Temp User",
                section="temp",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={}
            )
        ])
        
        result = service.vector_store.clear()
        if result['success']:
            print_success(f"Cleared {result['cleared_count']} records")
        else:
            print_failure(f"Clear failed: {result['errors']}")
            return False
        
        count_after = service.count()
        if count_after == 0:
            print_success(f"Count after clear: {count_after} (correct)")
        else:
            print_failure(f"Count after clear: {count_after} (expected 0)")
            return False
        print()
    except Exception as e:
        print_failure(f"Clear failed: {e}")
        return False
    
    # Test close
    print_header("CLOSE TEST")
    print()
    print_info("Closing vector store...")
    try:
        service.close()
        print_success("Vector store closed successfully")
        print()
    except Exception as e:
        print_failure(f"Close failed: {e}")
        return False
    
    # Test operations after close
    print_info("Testing operations after close (should fail)...")
    try:
        service.count()
        print_failure("Count should have failed after close")
        return False
    except Exception as e:
        print_success(f"Operations correctly blocked after close: {type(e).__name__}")
        print()
    
    # Final result
    print_header("MEMORY VECTOR STORE TEST PASSED")
    print_success("All operations working correctly")
    print()
    print("🚀 MemoryVectorStore Adapter Ready")
    return True


if __name__ == "__main__":
    try:
        success = test_memory_vector_store()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
