"""
Comprehensive test suite for the Chunk Object layer.

This script tests the ChunkService with a parsed ResumeDocument to verify
correctness of the Chunk object creation, serialization, and validation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser.parser_service import ParserService
from src.chunks.service import ChunkService
from src.chunks.schema import Chunk


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


def test_chunk_objects():
    """
    Comprehensive test for the Chunk Object layer.
    
    This function:
    1. Loads sample resume
    2. Runs ParserService → ResumeDocument → ChunkService.create_chunks()
    3. Prints total chunks and detailed info for each chunk
    4. Validates various aspects of the chunks
    5. Prints summary statistics
    """
    print_header("COMPREHENSIVE CHUNK OBJECT LAYER TEST")
    print()
    
    # Step 1: Load sample resume
    print_info("Step 1: Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Parse resume using ParserService
    print_info("Step 2: Parsing resume with ParserService...")
    document = None
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
    
    # Step 3: Create chunks using ChunkService
    print_info("Step 3: Creating chunks with ChunkService.create_chunks()...")
    chunks = None
    try:
        chunk_service = ChunkService()
        chunks = chunk_service.create_chunks(document, resume_id="chunk-object-test-001")
        print_success(f"Created {len(chunks)} Chunk objects")
    except Exception as e:
        print_failure(f"Failed to create chunks: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Print total chunks
    print_header("CHUNK INFORMATION")
    print()
    print(f"Total Chunks: {len(chunks)}")
    print()
    
    # Step 5: Print detailed information for every chunk
    print_header("DETAILED CHUNK INFORMATION")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\033[96m--- Chunk {i} ---\033[0m")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Resume ID: {chunk.resume_id}")
        print(f"Candidate Name: {chunk.candidate_name}")
        print(f"Section: {chunk.section}")
        print(f"Chunk Order: {chunk.chunk_order}")
        print(f"Metadata:")
        print(f"  Role: {chunk.metadata.role}")
        print(f"  Experience: {chunk.metadata.experience}")
        print(f"  Location: {chunk.metadata.location}")
        print(f"  Education: {chunk.metadata.education}")
        print(f"  Source Section: {chunk.metadata.source_section}")
        print(f"Text Length: {len(chunk.text)}")
        print(f"Embedding Status: {chunk.embedding_status}")
        print()
    
    # Step 6: Validate chunks
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate Chunk ID exists
    print_info("Validating Chunk ID exists...")
    try:
        if all(chunk.chunk_id for chunk in chunks):
            print_success("All chunks have Chunk ID")
        else:
            print_failure("Some chunks missing Chunk ID")
            validation_passed = False
    except Exception as e:
        print_failure(f"Chunk ID validation failed: {e}")
        validation_passed = False
    
    # Validate Resume ID exists
    print_info("Validating Resume ID exists...")
    try:
        if all(chunk.resume_id for chunk in chunks):
            print_success("All chunks have Resume ID")
        else:
            print_failure("Some chunks missing Resume ID")
            validation_passed = False
    except Exception as e:
        print_failure(f"Resume ID validation failed: {e}")
        validation_passed = False
    
    # Validate Candidate Name exists
    print_info("Validating Candidate Name exists...")
    try:
        if all(chunk.candidate_name for chunk in chunks):
            print_success("All chunks have Candidate Name")
        else:
            print_failure("Some chunks missing Candidate Name")
            validation_passed = False
    except Exception as e:
        print_failure(f"Candidate Name validation failed: {e}")
        validation_passed = False
    
    # Validate Section exists
    print_info("Validating Section exists...")
    try:
        if all(chunk.section for chunk in chunks):
            print_success("All chunks have Section")
        else:
            print_failure("Some chunks missing Section")
            validation_passed = False
    except Exception as e:
        print_failure(f"Section validation failed: {e}")
        validation_passed = False
    
    # Validate Metadata exists
    print_info("Validating Metadata exists...")
    try:
        if all(chunk.metadata for chunk in chunks):
            print_success("All chunks have Metadata")
        else:
            print_failure("Some chunks missing Metadata")
            validation_passed = False
    except Exception as e:
        print_failure(f"Metadata validation failed: {e}")
        validation_passed = False
    
    # Validate Text exists
    print_info("Validating Text exists...")
    try:
        if all(chunk.text for chunk in chunks):
            print_success("All chunks have Text")
        else:
            print_failure("Some chunks missing Text")
            validation_passed = False
    except Exception as e:
        print_failure(f"Text validation failed: {e}")
        validation_passed = False
    
    # Validate Chunk Order exists
    print_info("Validating Chunk Order exists...")
    try:
        if all(chunk.chunk_order is not None for chunk in chunks):
            print_success("All chunks have Chunk Order")
        else:
            print_failure("Some chunks missing Chunk Order")
            validation_passed = False
    except Exception as e:
        print_failure(f"Chunk Order validation failed: {e}")
        validation_passed = False
    
    # Validate no duplicate IDs
    print_info("Validating no duplicate IDs...")
    try:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) == len(set(chunk_ids)):
            print_success("No duplicate chunk IDs")
        else:
            print_failure("Found duplicate chunk IDs")
            validation_passed = False
    except Exception as e:
        print_failure(f"Duplicate ID validation failed: {e}")
        validation_passed = False
    
    # Validate no duplicate chunk text
    print_info("Validating no duplicate chunk text...")
    try:
        chunk_texts = [chunk.text.strip().lower() for chunk in chunks]
        if len(chunk_texts) == len(set(chunk_texts)):
            print_success("No duplicate chunk text")
        else:
            print_failure("Found duplicate chunk text")
            validation_passed = False
    except Exception as e:
        print_failure(f"Duplicate text validation failed: {e}")
        validation_passed = False
    
    # Validate chunk order sequential
    print_info("Validating chunk order sequential...")
    try:
        chunk_orders = [chunk.chunk_order for chunk in chunks]
        if chunk_orders == sorted(chunk_orders):
            print_success("Chunk order is sequential")
        else:
            print_failure("Chunk order is not sequential")
            validation_passed = False
    except Exception as e:
        print_failure(f"Chunk order validation failed: {e}")
        validation_passed = False
    
    # Validate no empty chunk
    print_info("Validating no empty chunk...")
    try:
        empty_chunks = [chunk for chunk in chunks if not chunk.text or not chunk.text.strip()]
        if not empty_chunks:
            print_success("No empty chunks")
        else:
            print_failure(f"Found {len(empty_chunks)} empty chunks")
            validation_passed = False
    except Exception as e:
        print_failure(f"Empty chunk validation failed: {e}")
        validation_passed = False
    
    # Validate no empty metadata
    print_info("Validating no empty metadata...")
    try:
        empty_metadata = [chunk for chunk in chunks if not chunk.metadata]
        if not empty_metadata:
            print_success("No empty metadata")
        else:
            print_failure(f"Found {len(empty_metadata)} chunks with empty metadata")
            validation_passed = False
    except Exception as e:
        print_failure(f"Empty metadata validation failed: {e}")
        validation_passed = False
    
    # Validate every chunk serializes correctly
    print_info("Validating every chunk serializes correctly...")
    try:
        if chunks:
            test_chunk = chunks[0]
            
            # Test model_dump()
            dump_dict = test_chunk.model_dump()
            if not dump_dict:
                print_failure("model_dump() returned empty dict")
                validation_passed = False
            else:
                print_success("model_dump() works correctly")
            
            # Test model_dump_json()
            dump_json = test_chunk.model_dump_json()
            if not dump_json:
                print_failure("model_dump_json() returned empty string")
                validation_passed = False
            else:
                print_success("model_dump_json() works correctly")
            
            # Test to_dict()
            to_dict_result = test_chunk.to_dict()
            if not to_dict_result:
                print_failure("to_dict() returned empty dict")
                validation_passed = False
            else:
                print_success("to_dict() works correctly")
            
            # Test to_json()
            to_json_result = test_chunk.to_json()
            if not to_json_result:
                print_failure("to_json() returned empty string")
                validation_passed = False
            else:
                print_success("to_json() works correctly")
            
            # Test summary()
            summary_result = test_chunk.summary()
            if not summary_result:
                print_failure("summary() returned empty string")
                validation_passed = False
            else:
                print_success("summary() works correctly")
    except Exception as e:
        print_failure(f"Serialization validation failed: {e}")
        validation_passed = False
    
    print()
    
    # Step 7: Print section distribution summary
    print_header("SECTION DISTRIBUTION SUMMARY")
    print()
    
    section_counts: Dict[str, int] = {}
    for chunk in chunks:
        section = chunk.section
        if section.startswith('experience_'):
            section_counts['experience'] = section_counts.get('experience', 0) + 1
        elif section.startswith('project_'):
            section_counts['projects'] = section_counts.get('projects', 0) + 1
        elif section.startswith('education_'):
            section_counts['education'] = section_counts.get('education', 0) + 1
        else:
            section_counts[section] = section_counts.get(section, 0) + 1
    
    print(f"Skills: {section_counts.get('skills', 0)}")
    print(f"Experience: {section_counts.get('experience', 0)}")
    print(f"Projects: {section_counts.get('projects', 0)}")
    print(f"Education: {section_counts.get('education', 0)}")
    print(f"Certifications: {section_counts.get('certifications', 0)}")
    print(f"Languages: {section_counts.get('languages', 0)}")
    print(f"Summary: {section_counts.get('summary', 0)}")
    print()
    
    # Step 8: Print statistics
    print_header("STATISTICS")
    print()
    
    # Total chunks
    print(f"Total chunks: {len(chunks)}")
    
    # Average text length
    if chunks:
        text_lengths = [len(chunk.text) for chunk in chunks]
        avg_text_length = sum(text_lengths) / len(text_lengths)
        print(f"Average text length: {avg_text_length:.1f} characters")
        
        # Largest chunk
        largest_chunk = max(chunks, key=lambda c: len(c.text))
        print(f"Largest chunk:")
        print(f"  Section: {largest_chunk.section}")
        print(f"  Length: {len(largest_chunk.text)} characters")
        
        # Smallest chunk
        smallest_chunk = min(chunks, key=lambda c: len(c.text))
        print(f"Smallest chunk:")
        print(f"  Section: {smallest_chunk.section}")
        print(f"  Length: {len(smallest_chunk.text)} characters")
        
        # Metadata completeness
        metadata_fields = ['role', 'experience', 'location', 'education', 'source_section']
        completeness_scores = []
        for chunk in chunks:
            filled_fields = sum(1 for field in metadata_fields if getattr(chunk.metadata, field) is not None)
            completeness = (filled_fields / len(metadata_fields)) * 100
            completeness_scores.append(completeness)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        print(f"Metadata completeness: {avg_completeness:.1f}%")
    
    print()
    
    # Step 9: Print final result
    if validation_passed:
        print_header("CHUNK OBJECT TEST PASSED")
        print_success("All validation checks passed")
        print_success("🚀 Chunk Objects Ready For Embedding Layer")
        return True
    else:
        print_header("CHUNK OBJECT TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_chunk_objects()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
