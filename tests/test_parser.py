"""
Test script for the Production Parser module.

This script tests the ParserService with a sample resume to verify
correctness of the parsing pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resume_parser.parser_service import ParserService


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


def test_parser():
    """
    Test the ParserService with a sample resume.
    
    This function:
    1. Loads a sample resume from the project
    2. Parses it using ParserService.parse_file()
    3. Prints candidate information
    4. Prints complete ResumeDocument JSON
    5. Validates the parsed data
    """
    print_header("PRODUCTION PARSER TEST")
    
    # Step 1: Load sample resume path
    print_info("Step 1: Loading sample resume...")
    sample_resume_path = Path(__file__).parent / "sample_resume.pdf"
    
    if not sample_resume_path.exists():
        print_failure(f"Sample resume not found at: {sample_resume_path}")
        return False
    
    print(f"Found sample resume: {sample_resume_path}")
    print()
    
    # Step 2: Initialize ParserService
    print_info("Step 2: Initializing ParserService...")
    try:
        parser = ParserService()
        print_success("ParserService initialized successfully")
    except Exception as e:
        print_failure(f"Failed to initialize ParserService: {e}")
        return False
    print()
    
    # Step 3: Parse the resume
    print_info("Step 3: Parsing resume...")
    try:
        document = parser.parse_file(sample_resume_path)
        print_success("Resume parsed successfully")
    except Exception as e:
        print_failure(f"Failed to parse resume: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 4: Print candidate information
    print_header("PARSED RESUME INFORMATION")
    print()
    print(f"Candidate Name: {document.name}")
    print(f"Email: {document.email}")
    print(f"Phone: {document.phone}")
    print(f"Skills: {', '.join(document.skills[:10])}..." if len(document.skills) > 10 else f"Skills: {', '.join(document.skills)}")
    print(f"Experience Count: {len(document.experience)}")
    print(f"Education Count: {len(document.education)}")
    print(f"Project Count: {len(document.projects)}")
    print(f"Certification Count: {len(document.certifications)}")
    print(f"Languages: {', '.join(document.languages)}")
    print()
    
    # Step 5: Print complete ResumeDocument JSON
    print_header("COMPLETE RESUMEDOCUMENT JSON")
    print()
    try:
        print(document.to_json())
    except Exception as e:
        print_failure(f"Failed to serialize ResumeDocument to JSON: {e}")
    print()
    
    # Step 6: Validate parsed data
    print_header("VALIDATION")
    print()
    
    validation_passed = True
    
    # Validate name is not empty
    if document.name and document.name.strip():
        print_success("Name is not empty")
    else:
        print_failure("Name is empty")
        validation_passed = False
    
    # Validate skills list exists
    if isinstance(document.skills, list):
        print_success("Skills list exists")
    else:
        print_failure("Skills list does not exist")
        validation_passed = False
    
    # Validate raw_text exists
    if document.raw_text and document.raw_text.strip():
        print_success("Raw text exists")
    else:
        print_failure("Raw text is empty")
        validation_passed = False
    
    # Validate ResumeDocument object is returned
    if document is not None:
        print_success("ResumeDocument object is returned")
    else:
        print_failure("ResumeDocument object is None")
        validation_passed = False
    
    print()
    
    # Step 7: Print final result
    if validation_passed:
        print_header("PARSER TEST PASSED")
        print_success("All validation checks passed")
        return True
    else:
        print_header("PARSER TEST FAILED")
        print_failure("Some validation checks failed")
        return False


if __name__ == "__main__":
    try:
        success = test_parser()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
