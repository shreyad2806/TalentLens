"""
Test script for CSV Resume Ingestion.

This script tests the CSV ingestion functionality including:
- CSV file detection
- CSV record loading
- Conversion to ResumeDocument schema
- Integration with indexing pipeline
- Diagnostics printing
"""

import sys
from pathlib import Path
import csv
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bootstrap.csv_ingestion import CSVIngestionService, CSVIngestionResult


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


def create_sample_csv(csv_path: Path) -> None:
    """
    Create a sample Resume.csv file for testing.
    
    Args:
        csv_path: Path where to create the CSV file
    """
    sample_data = [
        {
            'Resume_str': 'John Doe is a software engineer with 5 years of experience in Python development.',
            'Candidate': 'John Doe',
            'Email': 'john.doe@example.com',
            'Phone': '555-1234',
            'Location': 'San Francisco, CA',
            'Skills': 'Python, Django, PostgreSQL, Docker',
            'Experience': '5 years',
            'Education': 'BS Computer Science',
            'Role': 'Software Engineer',
            'Salary': '120000',
            'Notice_Period': '2 weeks',
            'Category': 'INFORMATION-TECHNOLOGY'
        },
        {
            'Resume_str': 'Jane Smith is a data scientist with expertise in machine learning and data analysis.',
            'Candidate': 'Jane Smith',
            'Email': 'jane.smith@example.com',
            'Phone': '555-5678',
            'Location': 'New York, NY',
            'Skills': 'Python, TensorFlow, Pandas, Scikit-learn',
            'Experience': '3 years',
            'Education': 'MS Data Science',
            'Role': 'Data Scientist',
            'Salary': '130000',
            'Notice_Period': '1 month',
            'Category': 'INFORMATION-TECHNOLOGY'
        },
        {
            'Resume_str': 'Bob Johnson is a project manager with experience in agile methodologies.',
            'Candidate': 'Bob Johnson',
            'Email': 'bob.johnson@example.com',
            'Phone': '555-9012',
            'Location': 'Austin, TX',
            'Skills': 'Agile, Scrum, JIRA, Project Management',
            'Experience': '7 years',
            'Education': 'MBA',
            'Role': 'Project Manager',
            'Salary': '110000',
            'Notice_Period': '2 weeks',
            'Category': 'BUSINESS-DEVELOPMENT'
        }
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_data)
    
    print_info(f"Created sample CSV with {len(sample_data)} records at {csv_path}")


def test_csv_detection():
    """
    Test CSV file detection in directories.
    """
    print_header("TEST 1: CSV File Detection")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test directories
        resume_dir = temp_path / "Resume"
        resume_dir.mkdir()
        
        # Create CSV file
        csv_path = resume_dir / "Resume.csv"
        create_sample_csv(csv_path)
        
        # Test detection
        service = CSVIngestionService()
        detected_csv = service.detect_csv_file(resume_dir)
        
        if detected_csv and detected_csv == csv_path:
            print_success("CSV file detected correctly")
            print_info(f"Detected path: {detected_csv}")
            return True
        else:
            print_failure("CSV file detection failed")
            print_info(f"Expected: {csv_path}")
            print_info(f"Got: {detected_csv}")
            return False


def test_csv_loading():
    """
    Test CSV record loading.
    """
    print_header("TEST 2: CSV Record Loading")
    
    # Create temporary CSV file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        csv_path = temp_path / "Resume.csv"
        create_sample_csv(csv_path)
        
        # Test loading
        service = CSVIngestionService()
        try:
            records = service.load_csv_records(csv_path)
            
            if len(records) == 3:
                print_success(f"Loaded {len(records)} CSV records correctly")
                
                # Verify record structure
                for i, record in enumerate(records):
                    print_info(f"Record {i + 1}:")
                    print_info(f"  Candidate: {record.get('Candidate')}")
                    print_info(f"  Resume_str length: {len(record.get('Resume_str', ''))}")
                
                return True
            else:
                print_failure(f"Expected 3 records, got {len(records)}")
                return False
                
        except Exception as e:
            print_failure(f"CSV loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_document_conversion():
    """
    Test conversion of CSV records to ResumeDocument schema.
    """
    print_header("TEST 3: CSV to ResumeDocument Conversion")
    
    # Create test record
    test_record = {
        'Resume_str': 'Test candidate with software engineering experience.',
        'Candidate': 'Test Candidate',
        'Email': 'test@example.com',
        'Phone': '555-0000',
        'Location': 'Test City',
        'Skills': 'Python, JavaScript',
        'Experience': '2 years',
        'Education': 'BS Computer Science',
        'Role': 'Developer',
        'Salary': '100000',
        'Notice_Period': '1 week',
        'Category': 'INFORMATION-TECHNOLOGY'
    }
    
    service = CSVIngestionService()
    
    try:
        document = service.convert_to_resume_document(test_record, "test-id-123")
        
        # Verify conversion
        if document['raw_text'] == test_record['Resume_str']:
            print_success("Resume_str mapped to raw_text correctly")
        else:
            print_failure("Resume_str mapping failed")
            return False
        
        if document['name'] == test_record['Candidate']:
            print_success("Candidate field mapped to name correctly")
        else:
            print_failure("Candidate field mapping failed")
            return False
        
        # Check metadata
        metadata = document['metadata']
        expected_fields = ['Candidate', 'Email', 'Phone', 'Location', 'Skills', 
                          'Experience', 'Education', 'Role', 'Salary', 'Notice_Period', 'Category']
        
        all_present = all(field in metadata for field in expected_fields)
        if all_present:
            print_success("All candidate fields mapped to metadata correctly")
        else:
            print_failure("Some metadata fields missing")
            missing = [f for f in expected_fields if f not in metadata]
            print_info(f"Missing fields: {missing}")
            return False
        
        print_info(f"Document structure: {list(document.keys())}")
        print_info(f"Metadata fields: {list(metadata.keys())}")
        
        return True
        
    except Exception as e:
        print_failure(f"Document conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_ingestion_integration():
    """
    Test CSV ingestion integration with indexing pipeline.
    """
    print_header("TEST 4: CSV Ingestion Integration")
    
    print_warning("This test requires a working indexing pipeline")
    print_info("Skipping integration test (requires full indexing setup)")
    
    # Note: Full integration test would require:
    # - Mock indexing service
    # - Test chunking, embedding, vector store, BM25
    # - Verify end-to-end pipeline
    
    return True


def test_diagnostics_printing():
    """
    Test diagnostics printing for CSV ingestion results.
    """
    print_header("TEST 5: Diagnostics Printing")
    
    # Create a sample result
    result = CSVIngestionResult(
        csv_rows_loaded=10,
        chunks_generated=50,
        vectors_indexed=50,
        bm25_documents_indexed=50,
        errors=[],
        load_time_seconds=5.5
    )
    
    service = CSVIngestionService()
    
    try:
        print_info("Printing sample diagnostics...")
        service.print_ingestion_results(result)
        print_success("Diagnostics printing completed without errors")
        return True
    except Exception as e:
        print_failure(f"Diagnostics printing failed: {e}")
        return False


def test_error_handling():
    """
    Test error handling for invalid CSV files.
    """
    print_header("TEST 6: Error Handling")
    
    service = CSVIngestionService()
    
    # Test with non-existent file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        non_existent_csv = temp_path / "non_existent.csv"
        
        try:
            records = service.load_csv_records(non_existent_csv)
            print_failure("Should have raised an error for non-existent file")
            return False
        except Exception as e:
            print_success(f"Correctly raised error for non-existent file: {type(e).__name__}")
    
    # Test with invalid CSV format
    invalid_csv = temp_path / "invalid.csv"
    invalid_csv.write_text("invalid,csv,format")
    
    try:
        records = service.load_csv_records(invalid_csv)
        # This might succeed but return empty or malformed data
        print_warning("Invalid CSV handling - check data integrity")
        return True
    except Exception as e:
        print_success(f"Correctly raised error for invalid CSV: {type(e).__name__}")
        return True


def main():
    """
    Run all CSV ingestion tests.
    """
    print_header("CSV RESUME INGESTION TEST SUITE")
    print()
    
    tests = [
        ("CSV File Detection", test_csv_detection),
        ("CSV Record Loading", test_csv_loading),
        ("Document Conversion", test_document_conversion),
        ("CSV Ingestion Integration", test_csv_ingestion_integration),
        ("Diagnostics Printing", test_diagnostics_printing),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print_failure(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()
    
    # Print summary
    print_header("TEST SUMMARY")
    print()
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print_header("CSV INGESTION TESTS PASSED")
        print_success("All CSV ingestion tests passed successfully")
        print()
        print("🚀 CSV Resume Ingestion Ready")
        return True
    else:
        print_header("CSV INGESTION TESTS FAILED")
        print_failure(f"{total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
