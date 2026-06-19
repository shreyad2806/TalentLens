"""
Test script for the Embedding Layer.

This script tests the EmbeddingService with parsed Chunk objects to verify
correctness of the embedding generation, validation, and caching.
"""

import sys
from pathlib import Path
from typing import List
import math
import psutil
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser.parser_service import ParserService
from src.chunks.service import ChunkService
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.model_loader import get_model_loader


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


def test_embedding_layer():
    """
    Test the embedding layer with parsed chunks.
    
    Pipeline:
    Load sample resume → ParserService → ResumeDocument → ChunkService → Chunks → EmbeddingService → EmbeddingRecords
    """
    print_header("EMBEDDING LAYER TEST")
    
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
        chunks = chunk_service.create_chunks(document, resume_id="embedding-test-001")
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Embed chunks
    print_info("Embedding chunks with EmbeddingService...")
    try:
        embedding_service = EmbeddingService()
        result = embedding_service.embed_chunks_with_stats(chunks)
        embeddings = result['embeddings']
        stats = result['stats']
        print_success(f"Generated {len(embeddings)} EmbeddingRecord objects")
    except Exception as e:
        print_failure(f"Failed to embed chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 5: Print embedding details
    print_header("EMBEDDING RECORD DETAILS")
    print()
    
    for i, embedding in enumerate(embeddings, 1):
        print(f"\033[96m--- Embedding {i} ---\033[0m")
        print(f"Chunk Section: {embedding.section}")
        print(f"Vector Dimension: {embedding.vector_dimension}")
        print(f"Model Name: {embedding.model_name}")
        print(f"Vector Length: {len(embedding.vector)}")
        print(f"Embedding ID: {embedding.embedding_id}")
        print(f"Resume ID: {embedding.resume_id}")
        print(f"Candidate: {embedding.candidate_name}")
        print(f"Metadata: {embedding.metadata}")
        print()
    
    # Step 6: Validate embeddings
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate vector exists
    print_info("Validating vector exists...")
    if all(embedding.vector for embedding in embeddings):
        print_success("All embeddings have vectors")
    else:
        print_failure("Some embeddings have missing vectors")
        validation_passed = False
    
    # Validate vector dimension correct
    print_info("Validating vector dimension correct...")
    expected_dimension = 1024
    if all(embedding.vector_dimension == expected_dimension for embedding in embeddings):
        print_success(f"All embeddings have correct dimension: {expected_dimension}")
    else:
        print_failure("Some embeddings have incorrect dimension")
        validation_passed = False
    
    # Validate vector contains no NaN
    print_info("Validating vector contains no NaN...")
    has_nan = any(any(math.isnan(val) for val in embedding.vector) for embedding in embeddings)
    if not has_nan:
        print_success("All embeddings have no NaN values")
    else:
        print_failure("Some embeddings have NaN values")
        validation_passed = False
    
    # Validate vector not empty
    print_info("Validating vector not empty...")
    if all(embedding.vector for embedding in embeddings):
        print_success("All embeddings have non-empty vectors")
    else:
        print_failure("Some embeddings have empty vectors")
        validation_passed = False
    
    # Validate embedding id unique
    print_info("Validating embedding id unique...")
    embedding_ids = [str(embedding.embedding_id) for embedding in embeddings]
    if len(embedding_ids) == len(set(embedding_ids)):
        print_success("All embeddings have unique IDs")
    else:
        print_failure("Some embeddings have duplicate IDs")
        validation_passed = False
    
    # Validate chunk id preserved
    print_info("Validating chunk id preserved...")
    chunk_ids_from_embeddings = [str(embedding.chunk_id) for embedding in embeddings]
    chunk_ids_from_chunks = [str(chunk.chunk_id) for chunk in chunks]
    if set(chunk_ids_from_embeddings) == set(chunk_ids_from_chunks):
        print_success("All chunk IDs are preserved")
    else:
        print_failure("Some chunk IDs are not preserved")
        validation_passed = False
    
    # Validate metadata preserved
    print_info("Validating metadata preserved...")
    metadata_preserved = True
    for embedding in embeddings:
        if embedding.metadata is None:
            metadata_preserved = False
            break
    if metadata_preserved:
        print_success("All embeddings have metadata")
    else:
        print_failure("Some embeddings are missing metadata")
        validation_passed = False
    
    print()
    
    # Step 7: Test cache
    print_header("CACHE TEST")
    print()
    
    print_info("Embedding same chunk twice to test cache...")
    try:
        # Clear cache first
        embedding_service.clear_cache()
        
        # Get first chunk
        test_chunk = chunks[0]
        
        # First embedding (should be cache miss)
        embedding_service.embed_chunk(test_chunk)
        cache_stats_after_first = embedding_service.get_cache_stats()
        
        # Second embedding (should be cache hit)
        embedding_service.embed_chunk(test_chunk)
        cache_stats_after_second = embedding_service.get_cache_stats()
        
        # Check if cache was used
        hits_after_first = cache_stats_after_first['hits']
        hits_after_second = cache_stats_after_second['hits']
        
        if hits_after_second > hits_after_first:
            print_success("Cache Hit - Second call used cache")
        else:
            print_failure("Cache Miss - Cache not working as expected")
            validation_passed = False
    except Exception as e:
        print_failure(f"Cache test failed: {e}")
        validation_passed = False
    
    print()
    
    # Step 8: Print statistics
    print_header("STATISTICS")
    print()
    
    # Total embeddings
    print(f"Total embeddings: {len(embeddings)}")
    
    # Average vector length
    avg_vector_length = sum(len(embedding.vector) for embedding in embeddings) / len(embeddings)
    print(f"Average vector length: {avg_vector_length:.2f}")
    
    # Embedding dimension
    print(f"Embedding dimension: {embeddings[0].vector_dimension if embeddings else 0}")
    
    # Model loaded
    model_loader = get_model_loader()
    print(f"Model loaded: {model_loader.is_loaded()}")
    print(f"Model name: {model_loader.get_model_name()}")
    
    # Memory usage estimate
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory usage estimate: {memory_mb:.2f} MB")
    
    print()
    
    # Step 9: Print final result
    if validation_passed:
        print_header("EMBEDDING TEST PASSED")
        print_success("All validation checks passed")
        print()
        print("🚀 Embedding Layer Ready For Vector Database")
        return True
    else:
        print_header("EMBEDDING TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_embedding_layer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
