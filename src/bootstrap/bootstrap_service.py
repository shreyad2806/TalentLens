"""
Bootstrap Service for Production System Initialization.

This module provides the main BootstrapService that orchestrates the complete
bootstrap workflow: loading resumes, parsing, chunking, embedding, and indexing
into vector store and BM25 index.

Workflow:
Application Startup → Load Resumes → CSV Ingestion (if detected) →
Production Parser → Semantic Chunking → Embedding Generation →
Vector Store Indexing → BM25 Index Building → Validation → Startup Report
"""

import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

from .resume_loader import ResumeLoader, LoadResult
from .startup_validator import StartupValidator
from .startup_report import StartupReport
from .csv_ingestion import CSVIngestionService, CSVIngestionResult

# Import existing indexing pipeline
from ..indexing.pipeline import IndexingPipeline

logger = logging.getLogger(__name__)


class BootstrapService:
    """
    Main service for bootstrapping the production retrieval system.
    
    This service orchestrates the complete bootstrap workflow to ensure
    the system has indexed data on startup. It only runs if the index
    is empty to avoid unnecessary re-indexing on every restart.
    
    Public Methods:
        - bootstrap(): Run bootstrap if index is empty
        - rebuild(): Force rebuild the entire index
        - validate(): Validate system state
        - status(): Get current system status
    """
    
    def __init__(
        self,
        resume_paths: Optional[list] = None,
        base_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the bootstrap service.
        
        Args:
            resume_paths: List of directory paths to search for resumes.
                         If None, uses default paths.
            base_path: Base directory to resolve relative paths from.
                      If None, uses current working directory.
            verbose: Whether to print detailed progress information
        """
        self.resume_loader = ResumeLoader(resume_paths=resume_paths)
        self.base_path = base_path
        self.verbose = verbose
        
        # Initialize indexing pipeline
        self.indexing_pipeline = IndexingPipeline()
        
        # Initialize CSV ingestion service
        self.csv_ingestion_service = CSVIngestionService()
        
        # Initialize validator and reporter
        self.validator = StartupValidator(self.indexing_pipeline)
        self.reporter = StartupReport()
        
        # Track bootstrap state
        self._last_bootstrap_time: Optional[float] = None
        self._last_bootstrap_result: Optional[Dict[str, Any]] = None
        
        logger.info("BootstrapService initialized")
    
    def bootstrap(self) -> Dict[str, Any]:
        """
        Run bootstrap if index is empty.
        
        This method checks if the system has indexed data. If the index is empty,
        it runs the complete bootstrap workflow. If the index already has data,
        it skips bootstrapping to avoid unnecessary re-indexing.
        
        Returns:
            Dictionary with bootstrap results and statistics
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("🔄 Production Bootstrap System")
            print("="*70)
        
        # Check if index is empty
        stats = self.indexing_pipeline.get_statistics()
        is_empty = (
            stats['indexed_documents'] == 0 and
            stats['vector_count'] == 0 and
            stats['bm25_count'] == 0
        )
        
        if not is_empty:
            if self.verbose:
                print(f"✅ Index already contains data")
                print(f"   Documents: {stats['indexed_documents']}")
                print(f"   Vectors: {stats['vector_count']}")
                print(f"   BM25 Docs: {stats['bm25_count']}")
                print("⏭️  Skipping bootstrap - system ready")
                print("="*70 + "\n")
            
            logger.info("Bootstrap skipped - index already contains data")
            
            return {
                'bootstrapped': False,
                'reason': 'index_not_empty',
                'statistics': stats,
                'bootstrap_time_seconds': 0.0
            }
        
        if self.verbose:
            print("📋 Index is empty - starting bootstrap workflow")
            print()
        
        # Run bootstrap workflow
        result = self._run_bootstrap_workflow()
        
        bootstrap_time = time.time() - start_time
        result['bootstrap_time_seconds'] = bootstrap_time
        self._last_bootstrap_time = bootstrap_time
        self._last_bootstrap_result = result
        
        # Print startup report
        if self.verbose:
            self.reporter.print_report(result)
        
        logger.info(f"Bootstrap complete in {bootstrap_time:.2f}s")
        
        return result
    
    def rebuild(self) -> Dict[str, Any]:
        """
        Force rebuild the entire index.
        
        This method clears existing indexes and rebuilds them from scratch.
        Use this when you want to refresh the entire index with new data.
        
        Returns:
            Dictionary with rebuild results and statistics
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("🔄 Production Bootstrap System - FORCE REBUILD")
            print("="*70)
            print("⚠️  Clearing existing indexes...")
        
        # Clear existing indexes
        rebuild_result = self.indexing_pipeline.rebuild_all(verbose=self.verbose)
        
        if self.verbose:
            print()
            print("📋 Rebuilding index from scratch...")
            print()
        
        # Run bootstrap workflow
        result = self._run_bootstrap_workflow()
        result['rebuild'] = True
        result['rebuild_result'] = rebuild_result
        
        bootstrap_time = time.time() - start_time
        result['bootstrap_time_seconds'] = bootstrap_time
        self._last_bootstrap_time = bootstrap_time
        self._last_bootstrap_result = result
        
        # Print startup report
        if self.verbose:
            self.reporter.print_report(result)
        
        logger.info(f"Rebuild complete in {bootstrap_time:.2f}s")
        
        return result
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the current system state.
        
        This method checks that the system is in a healthy state with
        indexed data and functional retrieval services.
        
        Returns:
            Dictionary with validation results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🔍 System Validation")
            print("="*70)
        
        validation_result = self.validator.validate()
        
        if self.verbose:
            self.reporter.print_validation(validation_result)
        
        return validation_result
    
    def status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        This method returns the current state of the system including
        indexed document counts, bootstrap status, and system health.
        
        Returns:
            Dictionary with system status information
        """
        stats = self.indexing_pipeline.get_statistics()
        
        status_info = {
            'statistics': stats,
            'last_bootstrap_time': self._last_bootstrap_time,
            'last_bootstrap_result': self._last_bootstrap_result,
            'is_bootstrapped': stats['indexed_documents'] > 0,
            'is_healthy': self.validator.is_healthy()
        }
        
        return status_info
    
    def _run_bootstrap_workflow(self) -> Dict[str, Any]:
        """
        Run the complete bootstrap workflow.
        
        This internal method executes the full bootstrap pipeline:
        1. Load resumes (detect CSV)
        2. Process CSV if detected
        3. Index individual files via IndexingPipeline
        4. Validate results
        
        Returns:
            Dictionary with workflow results
        """
        workflow_start = time.time()
        
        # Step 1: Load resumes
        if self.verbose:
            print("📂 Step 1: Loading Resumes")
        
        load_result = self.resume_loader.load_resumes(self.base_path)
        
        if self.verbose:
            print(f"   Found: {load_result.total_files_found} files")
            print(f"   Valid: {load_result.valid_files} files")
            print(f"   Invalid: {load_result.invalid_files} files")
            print(f"   Skipped: {load_result.skipped_files} files")
            print(f"   CSV Detected: {'Yes' if load_result.csv_detected else 'No'}")
            if load_result.csv_detected:
                print(f"   CSV Path: {load_result.csv_path}")
            print(f"   Load time: {load_result.load_time_seconds:.2f}s")
            print()
        
        # Step 2: Process CSV if detected
        csv_ingestion_result: Optional[CSVIngestionResult] = None
        if load_result.csv_detected and load_result.csv_path:
            if self.verbose:
                print("📊 Step 2: CSV Resume Ingestion")
                print("   Loading CSV records → Chunking → Embedding → Vector Store → BM25")
                print()
            
            try:
                csv_path = Path(load_result.csv_path)
                csv_ingestion_result = self.csv_ingestion_service.process_csv_for_indexing(
                    csv_path=csv_path,
                    indexing_service=self.indexing_pipeline.indexing_service
                )
                
                if self.verbose:
                    self.csv_ingestion_service.print_ingestion_results(csv_ingestion_result)
                    print()
                
            except Exception as e:
                logger.error(f"CSV ingestion failed: {e}")
                if self.verbose:
                    print(f"⚠️  CSV ingestion failed: {str(e)}")
                    print()
        else:
            if self.verbose:
                print("⏭️  Step 2: CSV Ingestion Skipped (No CSV detected)")
                print()
        
        # Step 3: Index individual resume files
        if self.verbose:
            print("📝 Step 3: Indexing Individual Resume Files")
            print("   Parsing → Chunking → Embedding → Vector Store → BM25")
            print()
        
        indexing_result = None
        if load_result.valid_files > 0:
            indexing_result = self.indexing_pipeline.index_files(
                load_result.file_paths,
                verbose=self.verbose
            )
            
            if self.verbose:
                print()
        else:
            if self.verbose:
                print("⏭️  Step 3: Individual File Indexing Skipped (No valid files)")
                print()
        
        # Check if we have any data to validate
        has_csv_data = csv_ingestion_result and csv_ingestion_result.csv_rows_loaded > 0
        has_file_data = load_result.valid_files > 0
        
        if not has_csv_data and not has_file_data:
            if self.verbose:
                print("⚠️  No resume data found (CSV or individual files)")
                print("❌ Bootstrap failed - no resumes to index")
                print("="*70 + "\n")
            
            return {
                'success': False,
                'reason': 'no_resume_data',
                'load_result': load_result,
                'csv_ingestion_result': csv_ingestion_result,
                'indexing_result': indexing_result,
                'validation_result': None
            }
        
        # Step 4: Validate
        if self.verbose:
            print("✅ Step 4: Validation")
        
        validation_result = self.validator.validate()
        
        workflow_time = time.time() - workflow_start
        
        result = {
            'success': validation_result['is_valid'],
            'load_result': load_result,
            'csv_ingestion_result': csv_ingestion_result,
            'indexing_result': indexing_result,
            'validation_result': validation_result,
            'workflow_time_seconds': workflow_time
        }
        
        return result
