"""
Test script for the PineconeVectorStore adapter.

This script tests the PineconeVectorStore implementation to verify that it
correctly implements the VectorStore interface and handles errors gracefully.
Note: This test requires Pinecone credentials to run actual operations.
"""

import sys
from pathlib import Path
import logging
import os

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
    VectorStoreProvider,
    VectorStoreError
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


def print_warning(message: str):
    """Print a warning message in yellow."""
    print(f"\033[93m⚠ {message}\033[0m")


def print_header(message: str):
    """Print a header message in yellow."""
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def test_pinecone_adapter_initialization():
    """
    Test Pinecone adapter initialization without credentials.
    
    This test verifies that the adapter fails gracefully when credentials
    are not provided, with clear error messages.
    """
    print_header("PINECONE ADAPTER INITIALIZATION TEST")
    print()
    
    # Clear environment variables to test error handling
    api_key_backup = os.environ.get("PINECONE_API_KEY")
    index_backup = os.environ.get("PINECONE_INDEX")
    
    if "PINECONE_API_KEY" in os.environ:
        del os.environ["PINECONE_API_KEY"]
    if "PINECONE_INDEX" in os.environ:
        del os.environ["PINECONE_INDEX"]
    
    # Set provider to pinecone
    os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"
    
    print_info("Testing Pinecone adapter without credentials...")
    try:
        service = VectorStoreService()
        print_failure("Service should have failed without credentials")
        return False
    except VectorStoreError as e:
        if "PINECONE_API_KEY" in str(e):
            print_success(f"Correctly failed without API key: {str(e)[:80]}...")
        else:
            print_failure(f"Unexpected error: {e}")
            return False
    except Exception as e:
        print_failure(f"Unexpected error type: {e}")
        return False
    print()
    
    # Restore environment variables if they existed
    if api_key_backup:
        os.environ["PINECONE_API_KEY"] = api_key_backup
    if index_backup:
        os.environ["PINECONE_INDEX"] = index_backup
    
    return True


def test_pinecone_adapter_with_credentials():
    """
    Test Pinecone adapter with credentials (if provided).
    
    This test attempts to initialize the Pinecone adapter with credentials
    and performs basic operations if credentials are available.
    """
    print_header("PINECONE ADAPTER WITH CREDENTIALS TEST")
    print()
    
    # Check if credentials are provided
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX")
    
    if not api_key or not index_name:
        print_warning("PINECONE_API_KEY or PINECONE_INDEX not set")
        print_info("Skipping integration test (requires Pinecone credentials)")
        print()
        print_info("To test with Pinecone, set environment variables:")
        print("  export PINECONE_API_KEY=your_api_key")
        print("  export PINECONE_INDEX=your_index_name")
        print()
        return True
    
    print_info("Credentials found, testing Pinecone adapter...")
    print(f"  API Key: {api_key[:10]}...")
    print(f"  Index: {index_name}")
    print()
    
    # Set provider to pinecone
    os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"
    
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
    print_info("Checking Pinecone health...")
    try:
        health = service.health()
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
    
    # Test upsert (if healthy)
    if health.get('healthy', False):
        print_info("Testing upsert operation...")
        try:
            # Create a test record with correct dimension
            config = service.config
            dimension = config.dimension
            
            # Create a test vector with correct dimension
            test_vector = [0.1] * dimension
            
            record = VectorRecord(
                id="test-pinecone-001",
                resume_id="test-resume-001",
                chunk_id="test-chunk-001",
                candidate_name="Test User",
                section="test",
                vector=test_vector,
                metadata={"test": True}
            )
            
            result = service.upsert([record])
            if result['success']:
                print_success(f"Upsert successful: {result['upserted_count']} records")
            else:
                print_failure(f"Upsert failed: {result['errors']}")
                return False
            print()
        except Exception as e:
            print_failure(f"Upsert failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test query
        print_info("Testing query operation...")
        try:
            results = service.query(test_vector, k=1)
            print_success(f"Query returned {len(results)} results")
            if results:
                print(f"  Top result ID: {results[0]['id']}")
                print(f"  Top result score: {results[0]['score']:.4f}")
            print()
        except Exception as e:
            print_failure(f"Query failed: {e}")
            return False
        
        # Test delete
        print_info("Testing delete operation...")
        try:
            result = service.delete(["test-pinecone-001"])
            if result['success']:
                print_success(f"Delete successful: {result['deleted_count']} records")
            else:
                print_failure(f"Delete failed: {result['errors']}")
            print()
        except Exception as e:
            print_failure(f"Delete failed: {e}")
            return False
    
    # Close service
    print_info("Closing Pinecone connection...")
    try:
        service.close()
        print_success("Pinecone connection closed")
        print()
    except Exception as e:
        print_failure(f"Close failed: {e}")
        return False
    
    return True


def test_pinecone_vector_store():
    """
    Test the PineconeVectorStore adapter.
    
    This test verifies that the PineconeVectorStore correctly implements
    the VectorStore interface and handles errors gracefully.
    """
    print_header("PINECONE VECTOR STORE TEST")
    print()
    
    # Test initialization without credentials
    if not test_pinecone_adapter_initialization():
        return False
    
    # Test with credentials if available
    if not test_pinecone_adapter_with_credentials():
        return False
    
    # Final result
    print_header("PINECONE VECTOR STORE TEST PASSED")
    print_success("All tests passed")
    print()
    print("🚀 PineconeVectorStore Adapter Ready")
    print()
    print("Note: Full integration test requires Pinecone credentials.")
    print("Set PINECONE_API_KEY and PINECONE_INDEX environment variables to test with Pinecone.")
    return True


if __name__ == "__main__":
    try:
        success = test_pinecone_vector_store()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
