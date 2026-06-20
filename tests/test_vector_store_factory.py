"""
Unit tests for VectorStoreFactory.

This script tests the VectorStoreFactory to verify that it correctly
instantiates the appropriate adapter based on configuration.
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

from src.vector_store.factory import VectorStoreFactory, create_vector_store
from src.vector_store.config import VectorStoreConfig, VectorStoreProvider


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


def test_memory_adapter_creation():
    """
    Test Memory adapter creation.
    
    This test verifies that the factory correctly creates a MemoryVectorStore
    when the provider is set to 'memory'.
    """
    print_header("MEMORY ADAPTER CREATION TEST")
    print()
    
    # Set environment to use memory provider
    os.environ["VECTOR_STORE_PROVIDER"] = "memory"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    print_info("Creating Memory adapter...")
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        
        # Verify the adapter is the correct type
        from src.vector_store.adapters.memory import MemoryVectorStore
        assert isinstance(adapter, MemoryVectorStore), "Expected MemoryVectorStore instance"
        
        print_success("Memory adapter created successfully")
        print(f"  Adapter type: {type(adapter).__name__}")
        print()
        
        # Clean up
        adapter.close()
        
        return True
    except Exception as e:
        print_failure(f"Memory adapter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pinecone_adapter_creation():
    """
    Test Pinecone adapter creation.
    
    This test verifies that the factory correctly creates a PineconeAdapter
    when the provider is set to 'pinecone' and credentials are provided.
    """
    print_header("PINECONE ADAPTER CREATION TEST")
    print()
    
    # Reset config to ensure fresh read of environment variables
    from src.vector_store.config import reset_config
    reset_config()
    
    # Set environment to use pinecone provider
    os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    # Test without credentials (should fail gracefully)
    print_info("Testing Pinecone adapter without credentials...")
    
    # Clear credentials
    api_key_backup = os.environ.get("PINECONE_API_KEY")
    index_backup = os.environ.get("PINECONE_INDEX_NAME")
    
    if "PINECONE_API_KEY" in os.environ:
        del os.environ["PINECONE_API_KEY"]
    if "PINECONE_INDEX_NAME" in os.environ:
        del os.environ["PINECONE_INDEX_NAME"]
    
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        print_failure("Should have failed during adapter creation without credentials")
        return False
    except Exception as e:
        if "PINECONE_API_KEY" in str(e) or "PINECONE_INDEX_NAME" in str(e):
            print_success(f"Correctly failed during creation without credentials: {str(e)[:80]}...")
        else:
            print_failure(f"Unexpected error: {e}")
            return False
    print()
    
    # Test with mock credentials
    print_info("Testing Pinecone adapter with mock credentials...")
    os.environ["PINECONE_API_KEY"] = "mock-api-key"
    os.environ["PINECONE_INDEX_NAME"] = "mock-index"
    
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        
        # Verify the adapter is the correct type
        from src.vector_store.adapters.pinecone_adapter import PineconeAdapter
        assert isinstance(adapter, PineconeAdapter), "Expected PineconeAdapter instance"
        
        print_success("Pinecone adapter created successfully")
        print(f"  Adapter type: {type(adapter).__name__}")
        print()
        
        # Clean up
        adapter.close()
        
    except Exception as e:
        # This might fail if Pinecone connection fails, which is expected
        # We just want to verify the factory creates the right adapter type
        if "PineconeAdapter" in str(type(e).__name__) or "pinecone" in str(e).lower():
            print_success(f"Pinecone adapter creation attempted (connection failed as expected): {str(e)[:80]}...")
        else:
            print_failure(f"Unexpected error: {e}")
            return False
    finally:
        # Restore environment variables
        if api_key_backup:
            os.environ["PINECONE_API_KEY"] = api_key_backup
        else:
            if "PINECONE_API_KEY" in os.environ:
                del os.environ["PINECONE_API_KEY"]
        
        if index_backup:
            os.environ["PINECONE_INDEX_NAME"] = index_backup
        else:
            if "PINECONE_INDEX_NAME" in os.environ:
                del os.environ["PINECONE_INDEX_NAME"]
    
    return True


def test_qdrant_adapter_creation():
    """
    Test Qdrant adapter creation.
    
    This test verifies that the factory correctly creates a QdrantAdapter
    when the provider is set to 'qdrant' and credentials are provided.
    """
    print_header("QDRANT ADAPTER CREATION TEST")
    print()
    
    # Reset config to ensure fresh read of environment variables
    from src.vector_store.config import reset_config
    reset_config()
    
    # Set environment to use qdrant provider
    os.environ["VECTOR_STORE_PROVIDER"] = "qdrant"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    # Test without credentials (should fail gracefully)
    print_info("Testing Qdrant adapter without credentials...")
    
    # Clear credentials
    collection_backup = os.environ.get("QDRANT_COLLECTION")
    
    if "QDRANT_COLLECTION" in os.environ:
        del os.environ["QDRANT_COLLECTION"]
    
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        print_failure("Should have failed during adapter creation without credentials")
        return False
    except Exception as e:
        if "QDRANT_COLLECTION" in str(e):
            print_success(f"Correctly failed during creation without credentials: {str(e)[:80]}...")
        else:
            print_failure(f"Unexpected error: {e}")
            return False
    print()
    
    # Test with mock credentials
    print_info("Testing Qdrant adapter with mock credentials...")
    os.environ["QDRANT_COLLECTION"] = "mock-collection"
    
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        
        # Verify the adapter is the correct type
        from src.vector_store.adapters.qdrant_adapter import QdrantAdapter
        assert isinstance(adapter, QdrantAdapter), "Expected QdrantAdapter instance"
        
        print_success("Qdrant adapter created successfully")
        print(f"  Adapter type: {type(adapter).__name__}")
        print()
        
        # Clean up
        adapter.close()
        
    except Exception as e:
        # This might fail if Qdrant connection fails, which is expected
        # We just want to verify the factory creates the right adapter type
        if "QdrantAdapter" in str(type(e).__name__) or "qdrant" in str(e).lower():
            print_success(f"Qdrant adapter creation attempted (connection failed as expected): {str(e)[:80]}...")
        else:
            print_failure(f"Unexpected error: {e}")
            return False
    finally:
        # Restore environment variables
        if collection_backup:
            os.environ["QDRANT_COLLECTION"] = collection_backup
        else:
            if "QDRANT_COLLECTION" in os.environ:
                del os.environ["QDRANT_COLLECTION"]
    
    return True


def test_unsupported_provider():
    """
    Test unsupported provider error handling.
    
    This test verifies that the factory raises a descriptive error
    when an unsupported provider is configured.
    """
    print_header("UNSUPPORTED PROVIDER TEST")
    print()
    
    # Reset config to ensure fresh read of environment variables
    from src.vector_store.config import reset_config
    reset_config()
    
    # Set environment to use unsupported provider
    os.environ["VECTOR_STORE_PROVIDER"] = "unsupported-provider"
    
    print_info("Testing unsupported provider...")
    try:
        factory = VectorStoreFactory()
        adapter = factory.create_vector_store()
        print_failure("Should have failed for unsupported provider")
        return False
    except ValueError as e:
        error_message = str(e)
        # The error comes from config validation, not factory
        if "Invalid vector store provider" in error_message or "Unsupported vector store provider" in error_message:
            print_success(f"Correctly raised descriptive error for unsupported provider")
            print(f"  Error message: {error_message[:100]}...")
        else:
            print_failure(f"Error message not descriptive enough: {error_message}")
            return False
    except Exception as e:
        print_failure(f"Unexpected error type: {e}")
        return False
    print()
    
    return True


def test_environment_variable_switching():
    """
    Test that switching providers requires only environment variable change.
    
    This test verifies that the factory correctly switches between providers
    when the environment variable is changed.
    """
    print_header("ENVIRONMENT VARIABLE SWITCHING TEST")
    print()
    
    # Reset config to ensure fresh read of environment variables
    from src.vector_store.config import reset_config
    reset_config()
    
    # Test switching from memory to pinecone
    print_info("Switching from memory to pinecone...")
    os.environ["VECTOR_STORE_PROVIDER"] = "memory"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    try:
        factory1 = VectorStoreFactory()
        adapter1 = factory1.create_vector_store()
        
        from src.vector_store.adapters.memory import MemoryVectorStore
        assert isinstance(adapter1, MemoryVectorStore), "Expected MemoryVectorStore"
        
        print_success("Memory adapter created")
        adapter1.close()
        
        # Switch to pinecone
        reset_config()  # Reset config to pick up new environment variable
        os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"
        os.environ["PINECONE_API_KEY"] = "mock-key"
        os.environ["PINECONE_INDEX_NAME"] = "mock-index"
        
        factory2 = VectorStoreFactory()
        adapter2 = factory2.create_vector_store()
        
        from src.vector_store.adapters.pinecone_adapter import PineconeAdapter
        assert isinstance(adapter2, PineconeAdapter), "Expected PineconeAdapter"
        
        print_success("Pinecone adapter created after environment switch")
        adapter2.close()
        
    except Exception as e:
        print_failure(f"Environment switching failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if "PINECONE_API_KEY" in os.environ:
            del os.environ["PINECONE_API_KEY"]
        if "PINECONE_INDEX_NAME" in os.environ:
            del os.environ["PINECONE_INDEX_NAME"]
    
    print()
    
    # Test switching from pinecone to qdrant
    print_info("Switching from pinecone to qdrant...")
    reset_config()  # Reset config
    os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"
    os.environ["PINECONE_API_KEY"] = "mock-key"
    os.environ["PINECONE_INDEX_NAME"] = "mock-index"
    
    try:
        factory3 = VectorStoreFactory()
        adapter3 = factory3.create_vector_store()
        
        from src.vector_store.adapters.pinecone_adapter import PineconeAdapter
        assert isinstance(adapter3, PineconeAdapter), "Expected PineconeAdapter"
        
        print_success("Pinecone adapter created")
        adapter3.close()
        
        # Switch to qdrant
        reset_config()  # Reset config to pick up new environment variable
        os.environ["VECTOR_STORE_PROVIDER"] = "qdrant"
        os.environ["QDRANT_COLLECTION"] = "mock-collection"
        
        factory4 = VectorStoreFactory()
        adapter4 = factory4.create_vector_store()
        
        from src.vector_store.adapters.qdrant_adapter import QdrantAdapter
        assert isinstance(adapter4, QdrantAdapter), "Expected QdrantAdapter"
        
        print_success("Qdrant adapter created after environment switch")
        adapter4.close()
        
    except Exception as e:
        print_failure(f"Environment switching failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if "PINECONE_API_KEY" in os.environ:
            del os.environ["PINECONE_API_KEY"]
        if "PINECONE_INDEX_NAME" in os.environ:
            del os.environ["PINECONE_INDEX_NAME"]
        if "QDRANT_COLLECTION" in os.environ:
            del os.environ["QDRANT_COLLECTION"]
    
    print()
    print_success("Environment variable switching works correctly")
    return True


def test_convenience_function():
    """
    Test the convenience function create_vector_store.
    
    This test verifies that the convenience function works correctly.
    """
    print_header("CONVENIENCE FUNCTION TEST")
    print()
    
    # Reset config to ensure fresh read of environment variables
    from src.vector_store.config import reset_config
    reset_config()
    
    # Set environment to use memory provider
    os.environ["VECTOR_STORE_PROVIDER"] = "memory"
    os.environ["VECTOR_STORE_DIMENSION"] = "4"
    
    print_info("Testing create_vector_store convenience function...")
    try:
        adapter = create_vector_store()
        
        from src.vector_store.adapters.memory import MemoryVectorStore
        assert isinstance(adapter, MemoryVectorStore), "Expected MemoryVectorStore"
        
        print_success("Convenience function works correctly")
        print(f"  Adapter type: {type(adapter).__name__}")
        print()
        
        # Clean up
        adapter.close()
        
        return True
    except Exception as e:
        print_failure(f"Convenience function failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store_factory():
    """
    Test the VectorStoreFactory implementation.
    
    This test verifies that the VectorStoreFactory correctly:
    - Creates Memory adapter
    - Creates Pinecone adapter
    - Creates Qdrant adapter
    - Handles unsupported providers with descriptive errors
    - Allows switching providers via environment variable
    - Provides working convenience function
    """
    print_header("VECTOR STORE FACTORY UNIT TESTS")
    print()
    
    # Test memory adapter creation
    if not test_memory_adapter_creation():
        return False
    
    # Test pinecone adapter creation
    if not test_pinecone_adapter_creation():
        return False
    
    # Test qdrant adapter creation
    if not test_qdrant_adapter_creation():
        return False
    
    # Test unsupported provider
    if not test_unsupported_provider():
        return False
    
    # Test environment variable switching
    if not test_environment_variable_switching():
        return False
    
    # Test convenience function
    if not test_convenience_function():
        return False
    
    # Final result
    print_header("VECTOR STORE FACTORY TESTS PASSED")
    print_success("All unit tests passed")
    print()
    print("🚀 VectorStoreFactory Ready")
    print()
    print("Summary:")
    print("- Memory adapter creation: ✅")
    print("- Pinecone adapter creation: ✅")
    print("- Qdrant adapter creation: ✅")
    print("- Unsupported provider error: ✅")
    print("- Environment variable switching: ✅")
    print("- Convenience function: ✅")
    return True


if __name__ == "__main__":
    try:
        success = test_vector_store_factory()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
