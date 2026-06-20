"""
Test script for the Vector Store Abstraction Layer.

This script tests the vector store abstraction layer to verify that the
interface, schema, configuration, validator, factory, and service work correctly.
Note: Adapters are not implemented yet, so this tests only the abstraction.
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
    VectorStoreConfig,
    VectorStoreProvider,
    get_config,
    VectorStoreValidator,
    ValidationError,
    VectorStoreFactory,
    VectorStoreService,
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


def print_header(message: str):
    """Print a header message in yellow."""
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def test_vector_record_schema():
    """Test VectorRecord schema validation."""
    print_header("VECTOR RECORD SCHEMA TEST")
    print()
    
    # Test valid record
    print_info("Testing valid VectorRecord...")
    try:
        record = VectorRecord(
            id="test-001",
            resume_id="resume-001",
            chunk_id="chunk-001",
            candidate_name="John Doe",
            section="skills",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"test": "data"}
        )
        print_success(f"Valid record created: {record.id}")
        print(f"  Dimension: {record.dimension}")
        print(f"  Section: {record.section}")
        print()
    except Exception as e:
        print_failure(f"Failed to create valid record: {e}")
        return False
    
    # Test empty vector
    print_info("Testing empty vector validation...")
    try:
        invalid_record = VectorRecord(
            id="test-002",
            resume_id="resume-001",
            chunk_id="chunk-002",
            candidate_name="John Doe",
            section="skills",
            vector=[],
            metadata={}
        )
        print_failure("Empty vector should have failed validation")
        return False
    except Exception as e:
        print_success(f"Empty vector correctly rejected: {e}")
        print()
    
    # Test NaN vector
    print_info("Testing NaN vector validation...")
    try:
        import math
        invalid_record = VectorRecord(
            id="test-003",
            resume_id="resume-001",
            chunk_id="chunk-003",
            candidate_name="John Doe",
            section="skills",
            vector=[0.1, math.nan, 0.3],
            metadata={}
        )
        print_failure("NaN vector should have failed validation")
        return False
    except Exception as e:
        print_success(f"NaN vector correctly rejected: {e}")
        print()
    
    return True


def test_config():
    """Test configuration loading."""
    print_header("CONFIG TEST")
    print()
    
    # Test default configuration
    print_info("Testing default configuration...")
    try:
        config = get_config()
        print_success(f"Configuration loaded successfully")
        print(f"  Provider: {config.provider.value}")
        print(f"  Dimension: {config.dimension}")
        print(f"  Metric: {config.metric}")
        print(f"  Is Memory: {config.is_memory()}")
        print()
    except Exception as e:
        print_failure(f"Failed to load configuration: {e}")
        return False
    
    # Test provider enum
    print_info("Testing VectorStoreProvider enum...")
    try:
        assert VectorStoreProvider.MEMORY.value == "memory"
        assert VectorStoreProvider.PINECONE.value == "pinecone"
        assert VectorStoreProvider.QDRANT.value == "qdrant"
        print_success("VectorStoreProvider enum works correctly")
        print()
    except Exception as e:
        print_failure(f"VectorStoreProvider enum failed: {e}")
        return False
    
    return True


def test_validator():
    """Test vector record validator."""
    print_header("VALIDATOR TEST")
    print()
    
    validator = VectorStoreValidator(expected_dimension=4)
    
    # Test valid record
    print_info("Testing valid record validation...")
    try:
        record = VectorRecord(
            id="test-001",
            resume_id="resume-001",
            chunk_id="chunk-001",
            candidate_name="John Doe",
            section="skills",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"test": "data"}
        )
        validator.validate_record(record)
        print_success("Valid record passed validation")
        print()
    except Exception as e:
        print_failure(f"Valid record failed validation: {e}")
        return False
    
    # Test dimension mismatch
    print_info("Testing dimension mismatch validation...")
    try:
        invalid_record = VectorRecord(
            id="test-002",
            resume_id="resume-001",
            chunk_id="chunk-002",
            candidate_name="John Doe",
            section="skills",
            vector=[0.1, 0.2, 0.3],  # Wrong dimension
            metadata={}
        )
        validator.validate_record(invalid_record)
        print_failure("Dimension mismatch should have failed validation")
        return False
    except ValidationError as e:
        print_success(f"Dimension mismatch correctly rejected: {e.message}")
        print()
    except Exception as e:
        print_failure(f"Unexpected error: {e}")
        return False
    
    # Test batch validation
    print_info("Testing batch validation...")
    try:
        records = [
            VectorRecord(
                id=f"test-{i}",
                resume_id="resume-001",
                chunk_id=f"chunk-{i}",
                candidate_name="John Doe",
                section="skills",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={"index": i}
            )
            for i in range(5)
        ]
        result = validator.validate_records(records)
        if result['valid']:
            print_success(f"Batch validation passed: {result['valid_count']} valid records")
        else:
            print_failure(f"Batch validation failed: {result['errors']}")
            return False
        print()
    except Exception as e:
        print_failure(f"Batch validation failed: {e}")
        return False
    
    # Test duplicate ID detection
    print_info("Testing duplicate ID detection...")
    try:
        duplicate_records = [
            VectorRecord(
                id="duplicate-id",
                resume_id="resume-001",
                chunk_id=f"chunk-{i}",
                candidate_name="John Doe",
                section="skills",
                vector=[0.1, 0.2, 0.3, 0.4],
                metadata={}
            )
            for i in range(3)
        ]
        result = validator.validate_records(duplicate_records)
        if not result['valid'] and len(result['duplicate_ids']) > 0:
            print_success(f"Duplicate IDs correctly detected: {result['duplicate_ids']}")
        else:
            print_failure("Duplicate IDs not detected")
            return False
        print()
    except Exception as e:
        print_failure(f"Duplicate ID detection failed: {e}")
        return False
    
    return True


def test_factory():
    """Test vector store factory."""
    print_header("FACTORY TEST")
    print()
    
    # Test factory initialization
    print_info("Testing factory initialization...")
    try:
        factory = VectorStoreFactory()
        print_success("Factory initialized successfully")
        print(f"  Config: {factory.config}")
        print()
    except Exception as e:
        print_failure(f"Factory initialization failed: {e}")
        return False
    
    # Test adapter creation (should raise NotImplementedError)
    print_info("Testing adapter creation (should raise NotImplementedError)...")
    try:
        vector_store = factory.create_vector_store()
        print_failure("Adapter creation should have raised NotImplementedError")
        return False
    except NotImplementedError as e:
        print_success(f"NotImplementedError correctly raised: {str(e)[:80]}...")
        print()
    except Exception as e:
        print_failure(f"Unexpected error: {e}")
        return False
    
    return True


def test_service():
    """Test vector store service initialization."""
    print_header("SERVICE TEST")
    print()
    
    # Test service initialization (should raise NotImplementedError)
    print_info("Testing service initialization (should raise NotImplementedError)...")
    try:
        service = VectorStoreService()
        print_failure("Service initialization should have raised NotImplementedError")
        return False
    except NotImplementedError as e:
        print_success(f"NotImplementedError correctly raised: {str(e)[:80]}...")
        print()
    except Exception as e:
        print_failure(f"Unexpected error: {e}")
        return False
    
    return True


def test_vector_store_abstraction():
    """
    Test the vector store abstraction layer.
    
    This test verifies that the abstraction layer components work correctly.
    Note: Adapters are not implemented yet, so this tests only the abstraction.
    """
    print_header("VECTOR STORE ABSTRACTION TEST")
    print()
    
    # Test VectorRecord schema
    if not test_vector_record_schema():
        return False
    
    # Test Config
    if not test_config():
        return False
    
    # Test Validator
    if not test_validator():
        return False
    
    # Test Factory
    if not test_factory():
        return False
    
    # Test Service
    if not test_service():
        return False
    
    # Final result
    print_header("VECTOR STORE ABSTRACTION TEST PASSED")
    print_success("All abstraction components working correctly")
    print()
    print("🚀 Vector Store Abstraction Layer Ready")
    print()
    print("Note: Adapters (Pinecone, Qdrant, Memory) are not implemented yet.")
    print("This is the abstraction layer only. Implement adapters in adapters/ directory.")
    return True


if __name__ == "__main__":
    try:
        success = test_vector_store_abstraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
