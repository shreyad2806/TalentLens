"""
Indexing Pipeline module - End-to-end indexing workflow.

This module provides the IndexingPipeline class that orchestrates the entire
indexing workflow from resume discovery to final storage in both vector store
and BM25 index. It provides a high-level interface for batch indexing operations.
"""

import logging
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

from .indexing_service import IndexingService
from .resume_ingestor import ResumeIngestor, IngestionResult
from ..config import EMBEDDING_DIM
from ..embeddings.embedding_service import EmbeddingService
from ..retrieval.sparse.bm25_index import BM25Index as SparseBM25Index

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    End-to-end indexing pipeline for resume documents.
    
    This class orchestrates the complete indexing workflow:
    1. Discover resume files from a source (directory or list)
    2. Ingest and validate files
    3. Index each resume through the full pipeline
    4. Aggregate results and statistics
    
    The pipeline provides a simple interface for batch indexing operations
    while handling errors gracefully and providing detailed feedback.
    """
    
    def __init__(
        self,
        *,
        embedding_dim: Optional[int] = None,
        bm25_index=None,
        embedding_service=None,
        vector_store_service=None,
    ):
        """Initialize the indexing pipeline.

        Dependency injection note:
        - bm25_index / embedding_service / vector_store_service are optional.
        - If provided, they are passed through to IndexingService.
        - If not provided, the pipeline creates default BM25/sparse and
          EmbeddingService instances so the service still receives fully
          configured dependencies.
        """
        self.ingestor = ResumeIngestor()

        bm25_passed = bm25_index is not None
        emb_passed = embedding_service is not None
        vec_passed = vector_store_service is not None

        bm25_index = bm25_index or SparseBM25Index()
        embedding_service = embedding_service or EmbeddingService(expected_dimension=embedding_dim or EMBEDDING_DIM)

        self.indexing_service = IndexingService(
            bm25_index=bm25_index,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
        )
        logger.info(
            "IndexingPipeline initialized (bm25_injected=%s, embedding_injected=%s, vector_store_injected=%s)",
            bm25_passed,
            emb_passed,
            vec_passed,
        )

    
    def index_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Index all resume files in a directory.
        
        This is the main entry point for batch indexing. It discovers
        all resume files in the specified directory and indexes them
        through the complete pipeline.
        
        Args:
            directory: Path to the directory containing resume files
            recursive: Whether to search recursively (default: True)
            verbose: Whether to print detailed progress (default: True)
            
        Returns:
            Dictionary with pipeline results including:
                - ingestion: IngestionResult from file discovery
                - indexing: Aggregate indexing results
                - statistics: Final indexing statistics
        """
        directory = Path(directory)
        
        if verbose:
            print(f"\n🔍 Discovering resumes in: {directory}")
        
        # Step 1: Ingest files from directory
        ingestion_result = self.ingestor.ingest_from_directory(
            directory=directory,
            recursive=recursive
        )
        
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_directory: discovered {ingestion_result.valid_files} valid files from {directory}")
        
        if verbose:
            print(f"📄 Found {ingestion_result.valid_files} valid resume files")
            if ingestion_result.invalid_files > 0:
                print(f"⚠️  Skipped {ingestion_result.invalid_files} invalid files")
            if ingestion_result.skipped_files > 0:
                print(f"ℹ️  Skipped {ingestion_result.skipped_files} non-resume files")
        
        if ingestion_result.valid_files == 0:
            logger.warning("No valid resume files found")
            return {
                'ingestion': ingestion_result,
                'indexing': None,
                'statistics': self.indexing_service.get_statistics()
            }
        
        # Step 2: Index all discovered files
        if verbose:
            print(f"\n🚀 Starting indexing pipeline for {ingestion_result.valid_files} files...")
        
        indexing_result = self.indexing_service.index_resumes(
            file_paths=ingestion_result.file_paths
        )
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_directory: indexing complete - successful={indexing_result.get('successful')}, failed={indexing_result.get('failed')}, total_chunks={indexing_result.get('total_chunks')}, total_embeddings={indexing_result.get('total_embeddings')}")
        
        if verbose:
            print(f"✅ Indexing complete:")
            print(f"   - Successful: {indexing_result['successful']}")
            print(f"   - Failed: {indexing_result['failed']}")
            print(f"   - Total chunks: {indexing_result['total_chunks']}")
            print(f"   - Total embeddings: {indexing_result['total_embeddings']}")
        
        # Step 3: Get final statistics
        statistics = self.indexing_service.get_statistics()
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_directory final stats: indexed_documents={statistics.get('indexed_documents')}, vector_count={statistics.get('vector_count')}, bm25_count={statistics.get('bm25_count')}")
        
        if verbose:
            self._print_statistics(statistics)
        
        return {
            'ingestion': ingestion_result,
            'indexing': indexing_result,
            'statistics': statistics
        }
    
    def index_files(
        self,
        file_paths: List[Union[str, Path]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Index a specific list of resume files.
        
        Args:
            file_paths: List of file paths to index
            verbose: Whether to print detailed progress (default: True)
            
        Returns:
            Dictionary with pipeline results
        """
        if verbose:
            print(f"\n🔍 Ingesting {len(file_paths)} resume files...")
        
        # Step 1: Ingest files from list
        ingestion_result = self.ingestor.ingest_from_list(file_paths)
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_files: validated {ingestion_result.valid_files} files from list of {len(file_paths)}")
        
        if verbose:
            print(f"📄 Validated {ingestion_result.valid_files} files")
            if ingestion_result.invalid_files > 0:
                print(f"⚠️  Skipped {ingestion_result.invalid_files} invalid files")
        
        if ingestion_result.valid_files == 0:
            logger.warning("No valid resume files found")
            return {
                'ingestion': ingestion_result,
                'indexing': None,
                'statistics': self.indexing_service.get_statistics()
            }
        
        # Step 2: Index all files
        if verbose:
            print(f"\n🚀 Starting indexing pipeline...")
        
        indexing_result = self.indexing_service.index_resumes(
            file_paths=ingestion_result.file_paths
        )
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_files: indexing complete - successful={indexing_result.get('successful')}, failed={indexing_result.get('failed')}, total_chunks={indexing_result.get('total_chunks')}, total_embeddings={indexing_result.get('total_embeddings')}")
        
        if verbose:
            print(f"✅ Indexing complete:")
            print(f"   - Successful: {indexing_result['successful']}")
            print(f"   - Failed: {indexing_result['failed']}")
            print(f"   - Total chunks: {indexing_result['total_chunks']}")
            print(f"   - Total embeddings: {indexing_result['total_embeddings']}")
        
        # Step 3: Get final statistics
        statistics = self.indexing_service.get_statistics()
        print(f"[BOOTSTRAP-TRACE][indexing_pipeline.py] index_files final stats: indexed_documents={statistics.get('indexed_documents')}, vector_count={statistics.get('vector_count')}, bm25_count={statistics.get('bm25_count')}")
        
        if verbose:
            self._print_statistics(statistics)
        
        return {
            'ingestion': ingestion_result,
            'indexing': indexing_result,
            'statistics': statistics
        }
    
    def index_single_file(
        self,
        file_path: Union[str, Path],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Index a single resume file.
        
        Args:
            file_path: Path to the resume file
            verbose: Whether to print detailed progress (default: True)
            
        Returns:
            Dictionary with indexing result
        """
        if verbose:
            print(f"\n🔍 Ingesting resume: {file_path}")
        
        # Step 1: Ingest single file
        ingestion_result = self.ingestor.ingest_single_file(file_path)
        
        if ingestion_result.valid_files == 0:
            logger.warning("File validation failed")
            if verbose:
                print(f"❌ File validation failed")
                for error in ingestion_result.errors:
                    print(f"   Error: {error}")
            return {
                'ingestion': ingestion_result,
                'indexing': None,
                'statistics': self.indexing_service.get_statistics()
            }
        
        # Step 2: Index the file
        if verbose:
            print(f"\n🚀 Starting indexing pipeline...")
        
        indexing_result = self.indexing_service.index_resume(
            file_path=ingestion_result.file_paths[0]
        )
        
        if verbose:
            if indexing_result['errors']:
                print(f"⚠️  Indexing completed with errors:")
                for error in indexing_result['errors']:
                    print(f"   Error: {error}")
            else:
                print(f"✅ Indexing successful:")
                print(f"   - Chunks: {indexing_result['chunks_count']}")
                print(f"   - Embeddings: {indexing_result['embeddings_count']}")
                print(f"   - Vector store: {'✓' if indexing_result['vector_store_success'] else '✗'}")
                print(f"   - BM25: {'✓' if indexing_result['bm25_success'] else '✗'}")
        
        # Step 3: Get final statistics
        statistics = self.indexing_service.get_statistics()
        
        if verbose:
            self._print_statistics(statistics)
        
        return {
            'ingestion': ingestion_result,
            'indexing': indexing_result,
            'statistics': statistics
        }
    
    def rebuild_all(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Rebuild the entire index from scratch.
        
        Args:
            verbose: Whether to print detailed progress (default: True)
            
        Returns:
            Dictionary with rebuild results
        """
        if verbose:
            print(f"\n🔄 Rebuilding index from scratch...")
        
        rebuild_result = self.indexing_service.rebuild_index()
        
        if verbose:
            print(f"✅ Rebuild complete:")
            print(f"   - BM25 cleared: {'✓' if rebuild_result['bm25_cleared'] else '✗'}")
            print(f"   - Vector store cleared: {'✓' if rebuild_result['vector_store_cleared'] else '✗'}")
        
        statistics = self.indexing_service.get_statistics()
        
        if verbose:
            self._print_statistics(statistics)
        
        return {
            'rebuild': rebuild_result,
            'statistics': statistics
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current indexing statistics.
        
        Returns:
            Dictionary with current statistics
        """
        return self.indexing_service.get_statistics()
    
    def print_startup_status(self) -> None:
        """
        Print startup status with indexed counts.
        
        This method prints the current state of the indexing system,
        including the number of indexed documents, vectors, and BM25 documents.
        """
        stats = self.indexing_service.get_statistics()
        
        print("\n" + "="*50)
        print("📊 Indexing Status")
        print("="*50)
        print(f"Indexed Documents: {stats['indexed_documents']}")
        print(f"Vector Count: {stats['vector_count']}")
        print(f"BM25 Count: {stats['bm25_count']}")
        
        if 'bm25_stats' in stats:
            bm25_stats = stats['bm25_stats']
            print(f"\nBM25 Statistics:")
            print(f"  Vocabulary Size: {bm25_stats['vocabulary_size']}")
            print(f"  Average Doc Length: {bm25_stats['avg_doc_length']:.2f}")
            print(f"  Total Tokens: {bm25_stats['total_tokens']}")
        
        print("="*50)
        
        # Check if we have indexed data
        if stats['indexed_documents'] > 0 and stats['vector_count'] > 0 and stats['bm25_count'] > 0:
            print("🚀 Indexing Ready")
        else:
            print("⚠️  Indexing incomplete - run indexing pipeline")
        print("="*50 + "\n")
    
    def _print_statistics(self, statistics: Dict[str, Any]) -> None:
        """
        Print indexing statistics.
        
        Args:
            statistics: Statistics dictionary
        """
        print(f"\n📊 Current Index Statistics:")
        print(f"   - Indexed Documents: {statistics['indexed_documents']}")
        print(f"   - Vector Count: {statistics['vector_count']}")
        print(f"   - BM25 Count: {statistics['bm25_count']}")
        
        if 'bm25_stats' in statistics:
            bm25_stats = statistics['bm25_stats']
            print(f"\n   BM25 Details:")
            print(f"   - Vocabulary Size: {bm25_stats['vocabulary_size']}")
            print(f"   - Average Doc Length: {bm25_stats['avg_doc_length']:.2f}")
            print(f"   - Total Tokens: {bm25_stats['total_tokens']}")
