"""
Test script for the Semantic Chunking module.

This script tests the ChunkService with a parsed ResumeDocument to verify
correctness of the chunking pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser.parser_service import ParserService
from src.chunking.chunk_service import ChunkService


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


def test_chunking():
    """
    Test the ChunkService with a parsed ResumeDocument.
    
    This function:
    1. Parses a sample resume using ParserService
    2. Passes ResumeDocument into ChunkService.generate_chunks()
    3. Prints every chunk with detailed information
    4. Validates chunk quality and structure
    """
    print_header("SEMANTIC CHUNKING TEST")
    
    # Step 1: Load sample resume path
    print_info("Step 1: Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Parse the resume
    print_info("Step 2: Parsing resume with ParserService...")
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
    
    # Step 3: Generate chunks
    print_info("Step 3: Generating chunks with ChunkService...")
    try:
        chunk_service = ChunkService()
        chunks = chunk_service.generate_chunks(document, resume_id="test-resume-001")
        print_success(f"Generated {len(chunks)} chunks")
    except Exception as e:
        print_failure(f"Failed to generate chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Print every chunk
    print_header("GENERATED CHUNKS")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\033[96m--- Chunk {i} ---\033[0m")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section}")
        print(f"Chunk Order: {chunk.chunk_order}")
        print(f"Resume ID: {chunk.resume_id}")
        print(f"Candidate Name: {chunk.candidate_name}")
        print(f"Metadata:")
        print(f"  Experience: {chunk.metadata.experience}")
        print(f"  Location: {chunk.metadata.location}")
        print(f"  Role: {chunk.metadata.role}")
        print(f"  Education: {chunk.metadata.education}")
        print(f"  Source Section: {chunk.metadata.source_section}")
        print(f"Text: {chunk.text[:200]}..." if len(chunk.text) > 200 else f"Text: {chunk.text}")
        print()
    
    # Step 5: Print total chunk count
    print_header("CHUNK STATISTICS")
    print()
    print(f"Total Chunks: {len(chunks)}")
    print()
    
    # Step 6: Validate chunks
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate no empty chunk
    empty_chunks = [chunk for chunk in chunks if not chunk.text or not chunk.text.strip()]
    if not empty_chunks:
        print_success("No empty chunks")
    else:
        print_failure(f"Found {len(empty_chunks)} empty chunks")
        validation_passed = False
    
    # Validate every chunk has section
    chunks_without_section = [chunk for chunk in chunks if not chunk.section]
    if not chunks_without_section:
        print_success("Every chunk has section")
    else:
        print_failure(f"Found {len(chunks_without_section)} chunks without section")
        validation_passed = False
    
    # Validate every chunk has metadata
    chunks_without_metadata = [chunk for chunk in chunks if not chunk.metadata]
    if not chunks_without_metadata:
        print_success("Every chunk has metadata")
    else:
        print_failure(f"Found {len(chunks_without_metadata)} chunks without metadata")
        validation_passed = False
    
    # Validate every chunk belongs to one resume
    resume_ids = set(chunk.resume_id for chunk in chunks)
    if len(resume_ids) == 1:
        print_success("All chunks belong to one resume")
    else:
        print_failure(f"Chunks belong to {len(resume_ids)} different resumes")
        validation_passed = False
    
    # Validate chunk order is increasing
    chunk_orders = [chunk.chunk_order for chunk in chunks]
    if chunk_orders == sorted(chunk_orders):
        print_success("Chunk order is increasing")
    else:
        print_failure("Chunk order is not increasing")
        validation_passed = False
    
    print()
    
    # Step 7: Print final result
    if validation_passed:
        print_header("CHUNKING TEST PASSED")
        print_success("All validation checks passed")
        return True
    else:
        print_header("CHUNKING TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_chunking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
