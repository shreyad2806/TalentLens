"""
Tests for Bootstrap System.

This module contains tests for the bootstrap system components:
- ResumeLoader
- BootstrapService
- StartupValidator
- StartupReport
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from src.bootstrap.resume_loader import ResumeLoader, LoadResult
from src.bootstrap.startup_validator import StartupValidator
from src.bootstrap.startup_report import StartupReport


class TestResumeLoader:
    """Tests for ResumeLoader class."""
    
    def test_init_default_paths(self):
        """Test initialization with default resume paths."""
        loader = ResumeLoader()
        assert loader.resume_paths == ['Resume', 'resumes', 'data/resumes']
    
    def test_init_custom_paths(self):
        """Test initialization with custom resume paths."""
        custom_paths = ['custom/path1', 'custom/path2']
        loader = ResumeLoader(resume_paths=custom_paths)
        assert loader.resume_paths == custom_paths
    
    def test_discover_resumes_empty_directory(self, tmp_path):
        """Test discovering resumes in empty directory."""
        loader = ResumeLoader(resume_paths=['test_resumes'])
        test_dir = tmp_path / 'test_resumes'
        test_dir.mkdir()
        
        files = loader.discover_resumes(base_path=str(tmp_path))
        assert files == []
    
    def test_discover_resumes_with_files(self, tmp_path):
        """Test discovering resumes with valid files."""
        loader = ResumeLoader(resume_paths=['test_resumes'])
        test_dir = tmp_path / 'test_resumes'
        test_dir.mkdir()
        
        # Create test files
        (test_dir / 'resume1.pdf').touch()
        (test_dir / 'resume2.docx').touch()
        (test_dir / 'resume3.txt').touch()
        (test_dir / 'readme.md').touch()  # Should be ignored
        
        files = loader.discover_resumes(base_path=str(tmp_path))
        assert len(files) == 3  # Only pdf, docx, txt
    
    def test_load_resumes_success(self, tmp_path):
        """Test loading resumes successfully."""
        loader = ResumeLoader(resume_paths=['test_resumes'])
        test_dir = tmp_path / 'test_resumes'
        test_dir.mkdir()
        
        # Create test files with content
        (test_dir / 'resume1.pdf').write_text('test content')
        (test_dir / 'resume2.docx').write_text('test content')
        
        result = loader.load_resumes(base_path=str(tmp_path))
        
        assert isinstance(result, LoadResult)
        assert result.total_files_found == 2
        assert result.valid_files == 2
        assert result.invalid_files == 0
        assert len(result.file_paths) == 2
    
    def test_load_resumes_empty_files(self, tmp_path):
        """Test loading resumes with empty files."""
        loader = ResumeLoader(resume_paths=['test_resumes'])
        test_dir = tmp_path / 'test_resumes'
        test_dir.mkdir()
        
        # Create empty file
        (test_dir / 'empty.pdf').touch()
        
        result = loader.load_resumes(base_path=str(tmp_path))
        
        assert result.valid_files == 0
        assert result.skipped_files == 1
    
    def test_load_resumes_nonexistent_path(self):
        """Test loading resumes from nonexistent path."""
        loader = ResumeLoader(resume_paths=['nonexistent'])
        result = loader.load_resumes()
        
        assert result.total_files_found == 0
        assert result.valid_files == 0
    
    def test_get_resume_count(self, tmp_path):
        """Test getting resume count."""
        loader = ResumeLoader(resume_paths=['test_resumes'])
        test_dir = tmp_path / 'test_resumes'
        test_dir.mkdir()
        
        (test_dir / 'resume1.pdf').write_text('test')
        (test_dir / 'resume2.docx').write_text('test')
        
        count = loader.get_resume_count(base_path=str(tmp_path))
        assert count == 2
    
    def test_validate_resume_paths(self, tmp_path):
        """Test validating resume paths."""
        loader = ResumeLoader(resume_paths=['valid_path', 'invalid_path'])
        
        valid_dir = tmp_path / 'valid_path'
        valid_dir.mkdir()
        
        result = loader.validate_resume_paths(base_path=str(tmp_path))
        
        assert 'valid_paths' in result
        assert 'invalid_paths' in result
        assert 'missing_paths' in result
        assert len(result['valid_paths']) == 1
        assert len(result['missing_paths']) == 1


class TestStartupValidator:
    """Tests for StartupValidator class."""
    
    def test_init(self):
        """Test initialization."""
        # Mock indexing pipeline
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 0,
                    'vector_count': 0,
                    'bm25_count': 0
                }
        
        validator = StartupValidator(MockPipeline())
        assert validator.indexing_pipeline is not None
    
    def test_validate_empty_system(self):
        """Test validation of empty system."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 0,
                    'vector_count': 0,
                    'bm25_count': 0
                }
        
        validator = StartupValidator(MockPipeline())
        result = validator.validate()
        
        assert result['is_valid'] == False
        assert len(result['errors']) > 0
    
    def test_validate_healthy_system(self):
        """Test validation of healthy system."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 10,
                    'vector_count': 50,
                    'bm25_count': 50
                }
        
        validator = StartupValidator(MockPipeline())
        result = validator.validate()
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
    
    def test_is_healthy_true(self):
        """Test is_healthy returns True for healthy system."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 10,
                    'vector_count': 50,
                    'bm25_count': 50
                }
        
        validator = StartupValidator(MockPipeline())
        assert validator.is_healthy() == True
    
    def test_is_healthy_false(self):
        """Test is_healthy returns False for unhealthy system."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 0,
                    'vector_count': 0,
                    'bm25_count': 0
                }
        
        validator = StartupValidator(MockPipeline())
        assert validator.is_healthy() == False
    
    def test_check_consistency_all_zero(self):
        """Test consistency check with all zero counts."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 0,
                    'vector_count': 0,
                    'bm25_count': 0
                }
        
        validator = StartupValidator(MockPipeline())
        stats = MockPipeline().get_statistics()
        result = validator._check_consistency(stats)
        
        assert result['passed'] == True
    
    def test_check_consistency_all_nonzero(self):
        """Test consistency check with all non-zero counts."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 10,
                    'vector_count': 50,
                    'bm25_count': 50
                }
        
        validator = StartupValidator(MockPipeline())
        stats = MockPipeline().get_statistics()
        result = validator._check_consistency(stats)
        
        assert result['passed'] == True
    
    def test_check_consistency_mixed(self):
        """Test consistency check with mixed counts."""
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 10,
                    'vector_count': 0,
                    'bm25_count': 50
                }
        
        validator = StartupValidator(MockPipeline())
        stats = MockPipeline().get_statistics()
        result = validator._check_consistency(stats)
        
        assert result['passed'] == False


class TestStartupReport:
    """Tests for StartupReport class."""
    
    def test_init(self):
        """Test initialization."""
        reporter = StartupReport()
        assert reporter is not None
    
    def test_print_report_success(self, capsys):
        """Test printing success report."""
        reporter = StartupReport()
        
        bootstrap_result = {
            'success': True,
            'load_result': LoadResult(
                total_files_found=10,
                valid_files=8,
                invalid_files=1,
                skipped_files=1,
                file_paths=['file1.pdf', 'file2.pdf'],
                errors=[],
                load_time_seconds=1.5
            ),
            'indexing_result': {
                'ingestion': {'valid_files': 8},
                'indexing': {
                    'successful': 8,
                    'failed': 0,
                    'total_chunks': 40,
                    'total_embeddings': 40
                }
            },
            'validation_result': {
                'is_valid': True,
                'statistics': {
                    'indexed_documents': 8,
                    'vector_count': 40,
                    'bm25_count': 40
                }
            },
            'workflow_time_seconds': 10.0
        }
        
        reporter.print_report(bootstrap_result)
        captured = capsys.readouterr()
        assert "Bootstrap Complete" in captured.out
    
    def test_print_report_failure(self, capsys):
        """Test printing failure report."""
        reporter = StartupReport()
        
        bootstrap_result = {
            'success': False,
            'reason': 'no_valid_resumes',
            'load_result': LoadResult(
                total_files_found=0,
                valid_files=0,
                invalid_files=0,
                skipped_files=0,
                file_paths=[],
                errors=[],
                load_time_seconds=0.5
            ),
            'indexing_result': None,
            'validation_result': None
        }
        
        reporter.print_report(bootstrap_result)
        captured = capsys.readouterr()
        assert "Bootstrap Failed" in captured.out
    
    def test_print_validation_pass(self, capsys):
        """Test printing validation report with pass."""
        reporter = StartupReport()
        
        validation_result = {
            'is_valid': True,
            'checks': {
                'documents_indexed': {'passed': True, 'message': 'Documents indexed: 10'},
                'vectors_indexed': {'passed': True, 'message': 'Vectors indexed: 50'},
                'bm25_indexed': {'passed': True, 'message': 'BM25 documents indexed: 50'},
                'services_healthy': {'passed': True, 'message': 'Retrieval services healthy'}
            },
            'statistics': {
                'indexed_documents': 10,
                'vector_count': 50,
                'bm25_count': 50
            },
            'errors': [],
            'warnings': []
        }
        
        reporter.print_validation(validation_result)
        captured = capsys.readouterr()
        assert "VALID" in captured.out
    
    def test_print_validation_fail(self, capsys):
        """Test printing validation report with failure."""
        reporter = StartupReport()
        
        validation_result = {
            'is_valid': False,
            'checks': {
                'documents_indexed': {'passed': False, 'message': 'No documents indexed'},
                'vectors_indexed': {'passed': False, 'message': 'No vectors indexed'},
                'bm25_indexed': {'passed': False, 'message': 'No BM25 documents indexed'},
                'services_healthy': {'passed': True, 'message': 'Retrieval services healthy'}
            },
            'statistics': {
                'indexed_documents': 0,
                'vector_count': 0,
                'bm25_count': 0
            },
            'errors': ['No documents indexed', 'No vectors indexed'],
            'warnings': []
        }
        
        reporter.print_validation(validation_result)
        captured = capsys.readouterr()
        assert "INVALID" in captured.out
    
    def test_print_status(self, capsys):
        """Test printing status report."""
        reporter = StartupReport()
        
        status_info = {
            'statistics': {
                'indexed_documents': 10,
                'vector_count': 50,
                'bm25_count': 50
            },
            'is_bootstrapped': True,
            'is_healthy': True,
            'last_bootstrap_time': 100.0
        }
        
        reporter.print_status(status_info)
        captured = capsys.readouterr()
        assert "System Status" in captured.out
    
    def test_generate_summary_success(self):
        """Test generating summary for successful bootstrap."""
        reporter = StartupReport()
        
        bootstrap_result = {
            'success': True,
            'validation_result': {
                'statistics': {
                    'indexed_documents': 10,
                    'vector_count': 50,
                    'bm25_count': 50
                }
            }
        }
        
        summary = reporter.generate_summary(bootstrap_result)
        assert "Bootstrap Complete" in summary
        assert "Documents: 10" in summary
        assert "Vectors: 50" in summary
        assert "BM25 Docs: 50" in summary
    
    def test_generate_summary_failure(self):
        """Test generating summary for failed bootstrap."""
        reporter = StartupReport()
        
        bootstrap_result = {
            'success': False,
            'reason': 'no_valid_resumes'
        }
        
        summary = reporter.generate_summary(bootstrap_result)
        assert "Bootstrap failed" in summary
        assert "no_valid_resumes" in summary


class TestBootstrapServiceIntegration:
    """Integration tests for BootstrapService."""
    
    def test_bootstrap_service_init(self):
        """Test BootstrapService initialization."""
        from src.bootstrap.bootstrap_service import BootstrapService
        
        service = BootstrapService(verbose=False)
        assert service.resume_loader is not None
        assert service.indexing_pipeline is not None
        assert service.validator is not None
        assert service.reporter is not None
    
    def test_bootstrap_service_status_empty(self):
        """Test BootstrapService status with empty system."""
        from src.bootstrap.bootstrap_service import BootstrapService
        
        # Mock indexing pipeline
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 0,
                    'vector_count': 0,
                    'bm25_count': 0
                }
        
        service = BootstrapService(verbose=False)
        service.indexing_pipeline = MockPipeline()
        
        status = service.status()
        assert status['is_bootstrapped'] == False
        assert status['is_healthy'] == False
    
    def test_bootstrap_service_status_healthy(self):
        """Test BootstrapService status with healthy system."""
        from src.bootstrap.bootstrap_service import BootstrapService
        
        # Mock indexing pipeline
        class MockPipeline:
            def get_statistics(self):
                return {
                    'indexed_documents': 10,
                    'vector_count': 50,
                    'bm25_count': 50
                }
        
        service = BootstrapService(verbose=False)
        service.indexing_pipeline = MockPipeline()
        
        status = service.status()
        assert status['is_bootstrapped'] == True
        assert status['is_healthy'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
