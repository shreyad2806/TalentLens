"""
Test script for the integrated ingestion pipeline.

This script tests the complete ingestion pipeline that integrates
Parser, Chunk, and Embedding layers with timing and logging.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import IngestionPipeline


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


def test_integrated_pipeline():
    """
    Test the integrated ingestion pipeline.
    
    Pipeline:
    Resume.pdf → ParserService → ResumeDocument → ChunkService → List[Chunk] → EmbeddingService → List[EmbeddingRecord]
    """
    print_header("INTEGRATED INGESTION PIPELINE TEST")
    
    # Step 1: Load sample resume
    print_info("Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Initialize pipeline
    print_info("Initializing ingestion pipeline...")
    try:
        pipeline = IngestionPipeline()
        print_success("Pipeline initialized successfully")
    except Exception as e:
        print_failure(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 3: Process resume through pipeline
    print_info("Processing resume through complete pipeline...")
    try:
        result = pipeline.process_resume(
            file_path=sample_resume_path,
            resume_id="pipeline-test-001",
            enable_embeddings=True
        )
        print_success("Pipeline processing completed successfully")
    except Exception as e:
        print_failure(f"Pipeline processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Print timing summary
    pipeline.print_timing_summary(result['timing'])
    print()
    
    # Step 5: Print statistics summary
    pipeline.print_stats_summary(result['stats'])
    print()
    
    # Step 6: Validate results
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate resume document
    print_info("Validating resume document...")
    if result['resume_document'] is not None:
        print_success("Resume document exists")
    else:
        print_failure("Resume document is missing")
        validation_passed = False
    
    # Validate chunks
    print_info("Validating chunks...")
    if result['chunks'] and len(result['chunks']) > 0:
        print_success(f"Chunks generated: {len(result['chunks'])}")
    else:
        print_failure("No chunks generated")
        validation_passed = False
    
    # Validate embeddings
    print_info("Validating embeddings...")
    if result['embeddings'] and len(result['embeddings']) > 0:
        print_success(f"Embeddings generated: {len(result['embeddings'])}")
    else:
        print_failure("No embeddings generated")
        validation_passed = False
    
    # Validate timing
    print_info("Validating timing information...")
    if 'parser_time' in result['timing'] and result['timing']['parser_time'] > 0:
        print_success("Parser timing recorded")
    else:
        print_failure("Parser timing missing")
        validation_passed = False
    
    if 'chunking_time' in result['timing'] and result['timing']['chunking_time'] > 0:
        print_success("Chunking timing recorded")
    else:
        print_failure("Chunking timing missing")
        validation_passed = False
    
    if 'embedding_time' in result['timing'] and result['timing']['embedding_time'] > 0:
        print_success("Embedding timing recorded")
    else:
        print_failure("Embedding timing missing")
        validation_passed = False
    
    if 'total_time' in result['timing'] and result['timing']['total_time'] > 0:
        print_success("Total timing recorded")
    else:
        print_failure("Total timing missing")
        validation_passed = False
    
    # Validate chunk and embedding count match
    print_info("Validating chunk and embedding count match...")
    if len(result['chunks']) == len(result['embeddings']):
        print_success("Chunk and embedding counts match")
    else:
        print_failure(f"Chunk count ({len(result['chunks'])}) does not match embedding count ({len(result['embeddings'])})")
        validation_passed = False
    
    print()
    
    # Step 7: Print final result
    if validation_passed:
        print_header("INTEGRATED PIPELINE TEST PASSED")
        print_success("All validation checks passed")
        print_success("Pipeline is production-ready")
        print()
        print("🚀 Ready For Vector Store Integration")
        return True
    else:
        print_header("INTEGRATED PIPELINE TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_integrated_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
