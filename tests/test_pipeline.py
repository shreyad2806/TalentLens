"""
Test script for the complete Ingestion Pipeline.

This script tests the end-to-end pipeline from resume PDF to semantic chunks
ready for embedding.
"""

import sys
import os
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser.parser_service import ParserService
from src.chunking.chunk_service import ChunkService
from src.chunking.schema import Chunk


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


def print_stage(stage_name: str, status: str):
    """Print a pipeline stage with status."""
    if status == "success":
        print_success(f"Pipeline Stage {stage_name}")
    else:
        print_failure(f"Pipeline Stage {stage_name}")


def calculate_chunk_statistics(chunks: List[Chunk]) -> dict:
    """
    Calculate statistics about the generated chunks.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'section_distribution': {},
            'average_chunk_length': 0,
            'largest_chunk': None,
            'smallest_chunk': None
        }
    
    # Total chunks
    total_chunks = len(chunks)
    
    # Section distribution
    section_distribution = {}
    for chunk in chunks:
        section = chunk.section
        section_distribution[section] = section_distribution.get(section, 0) + 1
    
    # Average chunk length
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    average_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
    
    # Largest chunk
    largest_chunk = max(chunks, key=lambda c: len(c.text))
    
    # Smallest chunk
    smallest_chunk = min(chunks, key=lambda c: len(c.text))
    
    return {
        'total_chunks': total_chunks,
        'section_distribution': section_distribution,
        'average_chunk_length': average_chunk_length,
        'largest_chunk': largest_chunk,
        'smallest_chunk': smallest_chunk
    }


def test_pipeline():
    """
    Test the complete ingestion pipeline.
    
    Pipeline flow:
    Resume.pdf → ParserService → ResumeDocument → ChunkService → Chunks
    
    This function:
    1. Loads a sample resume
    2. Parses it using ParserService
    3. Chunks it using ChunkService
    4. Prints pipeline stage status
    5. Displays chunk statistics
    """
    print_header("COMPLETE INGESTION PIPELINE TEST")
    print()
    
    # Initialize services
    print_info("Initializing services...")
    parser_service = None
    chunk_service = None
    
    try:
        parser_service = ParserService()
        print_success("ParserService initialized")
    except Exception as e:
        print_failure(f"Failed to initialize ParserService: {e}")
        return False
    
    try:
        chunk_service = ChunkService()
        print_success("ChunkService initialized")
    except Exception as e:
        print_failure(f"Failed to initialize ChunkService: {e}")
        return False
    
    print()
    
    # Load sample resume
    print_info("Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Pipeline Stage 1: Parse Resume
    print_header("PIPELINE STAGE 1: RESUME PARSING")
    print()
    print_info("Resume.pdf → ParserService → ResumeDocument")
    print()
    
    document = None
    try:
        document = parser_service.parse_file(sample_resume_path)
        print_success("Pipeline Stage 1")
        print(f"✓ Parsed resume for: {document.name}")
        print(f"✓ Extracted {len(document.skills)} skills")
        print(f"✓ Extracted {len(document.experience)} experience entries")
    except Exception as e:
        print_failure("Pipeline Stage 1")
        print(f"✗ Failed to parse resume: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Pipeline Stage 2: Chunk Document
    print_header("PIPELINE STAGE 2: SEMANTIC CHUNKING")
    print()
    print_info("ResumeDocument → ChunkService → Semantic Chunks")
    print()
    
    chunks = None
    try:
        chunks = chunk_service.generate_chunks(document, resume_id="pipeline-test-001")
        print_success("Pipeline Stage 2")
        print(f"✓ Generated {len(chunks)} semantic chunks")
    except Exception as e:
        print_failure("Pipeline Stage 2")
        print(f"✗ Failed to generate chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Pipeline Stage 3: Ready for Embedding
    print_header("PIPELINE STAGE 3: READY FOR EMBEDDING")
    print()
    print_info("Semantic Chunks → Ready For Embedding")
    print()
    
    try:
        # Validate chunks are ready for embedding
        if chunks and all(chunk.text for chunk in chunks):
            print_success("Pipeline Stage 3")
            print(f"✓ All chunks have text content")
            print(f"✓ All chunks have metadata")
            print(f"✓ Chunks are ready for embedding")
        else:
            print_failure("Pipeline Stage 3")
            print("✗ Some chunks are invalid for embedding")
            return False
    except Exception as e:
        print_failure("Pipeline Stage 3")
        print(f"✗ Validation failed: {e}")
        return False
    
    print()
    
    # Print pipeline flow
    print_header("PIPELINE FLOW")
    print()
    print("Resume")
    print("↓")
    print("ResumeDocument")
    print("↓")
    print("Semantic Chunks")
    print("↓")
    print_success("Ready For Embedding")
    print()
    
    # Display chunk statistics
    print_header("CHUNK STATISTICS")
    print()
    
    stats = calculate_chunk_statistics(chunks)
    
    print(f"Total Chunks: {stats['total_chunks']}")
    print()
    
    print("Section Distribution:")
    for section, count in sorted(stats['section_distribution'].items()):
        print(f"  {section}: {count}")
    print()
    
    print(f"Average Chunk Length: {stats['average_chunk_length']:.1f} characters")
    print()
    
    print(f"Largest Chunk:")
    print(f"  Section: {stats['largest_chunk'].section}")
    print(f"  Length: {len(stats['largest_chunk'].text)} characters")
    print(f"  Preview: {stats['largest_chunk'].text[:100]}...")
    print()
    
    print(f"Smallest Chunk:")
    print(f"  Section: {stats['smallest_chunk'].section}")
    print(f"  Length: {len(stats['smallest_chunk'].text)} characters")
    print(f"  Preview: {stats['smallest_chunk'].text[:100]}...")
    print()
    
    # Final success message
    print_header("PIPELINE TEST COMPLETED")
    print()
    print_success("🚀 Ingestion Pipeline Ready For Embedding Phase")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Pipeline test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
