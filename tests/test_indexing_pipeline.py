"""
Tests for the Indexing Pipeline module.

This module contains tests for the complete indexing pipeline, including
ingestion, parsing, chunking, embedding, and storage in both vector store
and BM25 index.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.indexing.pipeline import IndexingPipeline
from src.indexing.resume_ingestor import ResumeIngestor, IngestionResult
from src.indexing.indexing_service import IndexingService
from src.retrieval.sparse.bm25_index import BM25Index as SparseBM25Index


class TestResumeIngestor:
    """Tests for ResumeIngestor class."""
    
    def test_init_default_extensions(self):
        """Test initialization with default extensions."""
        ingestor = ResumeIngestor()
        assert ingestor.supported_extensions == {'.pdf', '.docx', '.doc', '.txt'}
    
    def test_init_custom_extensions(self):
        """Test initialization with custom extensions."""
        custom_exts = {'.pdf', '.txt'}
        ingestor = ResumeIngestor(supported_extensions=custom_exts)
        assert ingestor.supported_extensions == custom_exts
    
    def test_ingest_from_nonexistent_directory(self):
        """Test ingestion from a non-existent directory."""
        ingestor = ResumeIngestor()
        result = ingestor.ingest_from_directory("/nonexistent/path")
        
        assert result.total_files == 0
        assert result.valid_files == 0
        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0]
    
    def test_ingest_from_file_instead_of_directory(self):
        """Test ingestion when path is a file, not a directory."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test content")
            temp_file = f.name
        
        try:
            ingestor = ResumeIngestor()
            result = ingestor.ingest_from_directory(temp_file)
            
            assert result.valid_files == 0
            assert len(result.errors) > 0
            assert "not a directory" in result.errors[0]
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_from_empty_directory(self):
        """Test ingestion from an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ingestor = ResumeIngestor()
            result = ingestor.ingest_from_directory(temp_dir, recursive=False)
            
            assert result.total_files == 0
            assert result.valid_files == 0
            assert result.invalid_files == 0
    
    def test_ingest_from_directory_with_mixed_files(self):
        """Test ingestion from directory with mixed file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "resume1.pdf").touch()
            (temp_path / "resume2.docx").touch()
            (temp_path / "resume3.txt").touch()
            (temp_path / "image.jpg").touch()
            (temp_path / "data.csv").touch()
            
            ingestor = ResumeIngestor()
            result = ingestor.ingest_from_directory(temp_dir, recursive=False)
            
            assert result.valid_files == 3  # pdf, docx, txt
            assert result.skipped_files == 2  # jpg, csv
            assert result.invalid_files == 0
    
    def test_ingest_from_list_with_valid_files(self):
        """Test ingestion from a list of valid file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            pdf_file = temp_path / "resume.pdf"
            txt_file = temp_path / "resume.txt"
            pdf_file.touch()
            txt_file.touch()
            
            ingestor = ResumeIngestor()
            result = ingestor.ingest_from_list([pdf_file, txt_file])
            
            assert result.valid_files == 2
            assert result.invalid_files == 0
            assert len(result.file_paths) == 2
    
    def test_ingest_from_list_with_invalid_files(self):
        """Test ingestion from a list with non-existent files."""
        ingestor = ResumeIngestor()
        result = ingestor.ingest_from_list(["/nonexistent1.pdf", "/nonexistent2.docx"])
        
        assert result.valid_files == 0
        assert result.invalid_files == 2
        assert len(result.errors) == 2
    
    def test_filter_by_extension(self):
        """Test filtering files by extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            files = [
                temp_path / "resume1.pdf",
                temp_path / "resume2.docx",
                temp_path / "resume3.txt",
                temp_path / "image.jpg"
            ]
            for f in files:
                f.touch()
            
            ingestor = ResumeIngestor()
            filtered = ingestor.filter_by_extension(files, {'.pdf', '.txt'})
            
            assert len(filtered) == 2
            assert all(f.suffix in {'.pdf', '.txt'} for f in filtered)
    
    def test_filter_by_size(self):
        """Test filtering files by size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files with different sizes
            small_file = temp_path / "small.txt"
            large_file = temp_path / "large.txt"
            small_file.write_text("x" * 10)
            large_file.write_text("x" * 1000)
            
            ingestor = ResumeIngestor()
            filtered = ingestor.filter_by_size([small_file, large_file], min_size_bytes=100)
            
            assert len(filtered) == 1
            assert filtered[0] == large_file


class TestIndexingService:
    """Tests for IndexingService class."""
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_init(self):
        """Test initialization of IndexingService."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        
        assert service.parser is not None
        assert service.chunk_service is not None
        assert service.embedding_service is not None
        assert service.index_builder is not None
        assert service.document_count() == 0
        assert service.vector_count() == 0
        assert service.bm25_count() == 0
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_document_count(self):
        """Test document count tracking."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        assert service.document_count() == 0
        
        # Simulate adding a document
        service._indexed_documents["test_id"] = {"file_path": "test.pdf"}
        assert service.document_count() == 1
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_vector_count(self):
        """Test vector count tracking."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        assert service.vector_count() == 0
        
        service._vector_count = 10
        assert service.vector_count() == 10
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_bm25_count(self):
        """Test BM25 count tracking."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        assert service.bm25_count() == 0
        
        # After assigning a built (empty) BM25 index, count is still 0
        service._bm25_index = service.index_builder.build_index([])
        assert service.bm25_count() == 0  # Empty index
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_get_statistics(self):
        """Test getting comprehensive statistics."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        service._indexed_documents = {"doc1": {}, "doc2": {}}
        service._vector_count = 20
        service._bm25_index = service.index_builder.build_index([])
        
        stats = service.get_statistics()
        
        assert stats['indexed_documents'] == 2
        assert stats['vector_count'] == 20
        assert stats['bm25_count'] == 0
        assert 'bm25_stats' in stats
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_rebuild_index(self):
        """Test rebuilding the index."""
        service = IndexingService(
            bm25_index=SparseBM25Index(),
            embedding_service=MagicMock(),
        )
        service._indexed_documents = {"doc1": {}}
        service._vector_count = 10
        service._bm25_index = service.index_builder.build_index([])
        
        result = service.rebuild_index()
        
        assert result['bm25_cleared'] == True
        assert service.document_count() == 0
        assert service.vector_count() == 0


class TestIndexingPipeline:
    """Tests for IndexingPipeline class."""
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_init(self):
        """Test initialization of IndexingPipeline."""
        pipeline = IndexingPipeline(embedding_dim=1024)
        
        assert pipeline.ingestor is not None
        assert pipeline.indexing_service is not None
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_get_statistics(self):
        """Test getting statistics from pipeline."""
        pipeline = IndexingPipeline()
        stats = pipeline.get_statistics()
        
        assert 'indexed_documents' in stats
        assert 'vector_count' in stats
        assert 'bm25_count' in stats
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_index_directory_nonexistent(self):
        """Test indexing a non-existent directory."""
        pipeline = IndexingPipeline()
        result = pipeline.index_directory("/nonexistent/path", verbose=False)
        
        assert result['ingestion'].valid_files == 0
        assert result['indexing'] is None
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_index_directory_empty(self):
        """Test indexing an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = IndexingPipeline()
            result = pipeline.index_directory(temp_dir, verbose=False)
            
            assert result['ingestion'].valid_files == 0
            assert result['indexing'] is None
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_index_directory_with_files(self):
        """Test indexing a directory with resume files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple text file (not a real resume, but valid for structure test)
            test_file = temp_path / "test_resume.txt"
            test_file.write_text("John Doe\nSoftware Engineer\nSkills: Python, Java")
            
            pipeline = IndexingPipeline()
            
            # Mock the indexing service to avoid actual parsing/embedding
            with patch.object(pipeline.indexing_service, 'index_resumes') as mock_index:
                mock_index.return_value = {
                    'total_files': 1,
                    'successful': 1,
                    'failed': 0,
                    'total_chunks': 5,
                    'total_embeddings': 5,
                    'results': []
                }
                
                result = pipeline.index_directory(temp_dir, verbose=False)
                
                assert result['ingestion'].valid_files == 1
                assert result['indexing'] is not None
                assert result['indexing']['total_files'] == 1
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_index_files_list(self):
        """Test indexing a list of files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            file1 = temp_path / "resume1.txt"
            file2 = temp_path / "resume2.txt"
            file1.write_text("Resume 1")
            file2.write_text("Resume 2")
            
            pipeline = IndexingPipeline()
            
            # Mock the indexing service
            with patch.object(pipeline.indexing_service, 'index_resumes') as mock_index:
                mock_index.return_value = {
                    'total_files': 2,
                    'successful': 2,
                    'failed': 0,
                    'total_chunks': 10,
                    'total_embeddings': 10,
                    'results': []
                }
                
                result = pipeline.index_files([file1, file2], verbose=False)
                
                assert result['ingestion'].valid_files == 2
                assert result['indexing'] is not None
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_rebuild_all(self):
        """Test rebuilding all indexes."""
        pipeline = IndexingPipeline()
        
        # Add some data
        pipeline.indexing_service._indexed_documents = {"doc1": {}}
        pipeline.indexing_service._vector_count = 10
        
        result = pipeline.rebuild_all(verbose=False)
        
        assert result['rebuild']['bm25_cleared'] == True
        assert pipeline.indexing_service.document_count() == 0


class TestIntegration:
    """Integration tests for the complete indexing pipeline."""
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_pipeline_end_to_end_structure(self):
        """Test that the pipeline structure is correct."""
        pipeline = IndexingPipeline()
        
        # Verify all components are present
        assert hasattr(pipeline, 'ingestor')
        assert hasattr(pipeline, 'indexing_service')
        assert hasattr(pipeline, 'index_directory')
        assert hasattr(pipeline, 'index_files')
        assert hasattr(pipeline, 'index_single_file')
        assert hasattr(pipeline, 'rebuild_all')
        assert hasattr(pipeline, 'get_statistics')
        assert hasattr(pipeline, 'print_startup_status')
    
    @patch.dict('sys.modules', {'pinecone': MagicMock()})
    def test_statistics_verification(self):
        """Test that statistics can be retrieved and verified."""
        pipeline = IndexingPipeline()
        
        # Initially should be empty
        stats = pipeline.get_statistics()
        assert stats['indexed_documents'] == 0
        assert stats['vector_count'] == 0
        assert stats['bm25_count'] == 0
        
        # Simulate having indexed data
        pipeline.indexing_service._indexed_documents = {
            "doc1": {"file_path": "resume1.pdf"},
            "doc2": {"file_path": "resume2.pdf"},
            "doc3": {"file_path": "resume3.pdf"}
        }
        pipeline.indexing_service._vector_count = 15
        pipeline.indexing_service._bm25_index = pipeline.indexing_service.index_builder.build_index([])
        
        # Verify counts are greater than 0
        stats = pipeline.get_statistics()
        assert stats['indexed_documents'] == 3
        assert stats['vector_count'] == 15
        # BM25 count might be 0 if no chunks were added, but the index exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
