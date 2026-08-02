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
from pathlib import Path
from typing import Any

from ..debug_logger import log_stage_end, log_stage_start

# Import existing indexing pipeline
from ..indexing.pipeline import IndexingPipeline
from ..retrieval.sparse.bm25_index import IncompatibleIndexError
from .bootstrap_status import BootstrapStatus
from .csv_ingestion import CSVIngestionResult, CSVIngestionService
from .dataset_hash import DatasetHashManager
from .index_recovery_service import IndexRecoveryService
from .resume_loader import LoadResult, ResumeLoader
from .startup_report import StartupReport
from .startup_validator import StartupValidator

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
        resume_paths: list | None = None,
        base_path: str | None = None,
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
        
        # Initialize indexing pipeline (dependency injected from composition root)
        # NOTE: composition_root is the only place allowed to construct BM25Index/EmbeddingService/VectorStoreService.
        from .composition_root import create_retrieval_bundle
        bundle = create_retrieval_bundle()

        self.indexing_pipeline = IndexingPipeline(
            bm25_index=bundle.bm25_index,
            embedding_service=bundle.embedding_service,
            vector_store_service=bundle.vector_store_service,
        )

        
        # Initialize CSV ingestion service
        self.csv_ingestion_service = CSVIngestionService()
        
        # Initialize dataset hash manager for cache-invalidation decisions
        self.dataset_hash_manager = DatasetHashManager()

        # Store bundle references for temporary identity logging
        self._bm25_id = id(bundle.bm25_index)
        self._emb_id = id(bundle.embedding_service)
        self._vec_id = id(bundle.vector_store_service)
        if self.verbose:
            print(f"[IDENTITY] BootstrapService bm25_id={self._bm25_id} embedding_id={self._emb_id} vector_store_id={self._vec_id}")

        
        # Initialize index recovery service (bootstrap-only)
        self.index_recovery_service = IndexRecoveryService(self.indexing_pipeline.indexing_service)

        # Initialize validator and reporter
        self.validator = StartupValidator(self.indexing_pipeline)
        self.reporter = StartupReport()
        
        # Track bootstrap state
        self._last_bootstrap_time: float | None = None
        self._last_bootstrap_result: dict[str, Any] | None = None
        
        logger.info("BootstrapService initialized")
    
    def _indexes_exist_on_disk(self) -> bool:
        """Check whether persisted BM25 index files exist on disk."""
        bm25_metadata_path = Path("data/indexes/bm25/metadata.json")
        return bm25_metadata_path.exists()
    
    def _load_indexes(self) -> dict[str, Any]:
        """Load persisted indexes from disk into the in-memory services."""
        result = {
            'loaded': True,
            'bm25_loaded': False,
            'vector_count': 0,
            'indexed_documents': 0,
            'errors': []
        }
        
        try:
            bm25_index_path = Path("data/indexes/bm25")
            bm25_index = self.indexing_pipeline.indexing_service.get_bm25_index()
            if bm25_index is not None:
                bm25_index.load_from_disk(bm25_index_path)
                result['bm25_loaded'] = True
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] BM25 index loaded from {bm25_index_path}")
            else:
                result['errors'].append("BM25 index instance not available")
        except IncompatibleIndexError as e:
            logger.error(f"Incompatible persisted BM25 index detected: {e}")
            result['errors'].append(f"Incompatible persisted index: {e!s}")
            self._delete_persisted_indexes()
        except Exception as e:
            result['errors'].append(f"BM25 load failed: {e!s}")
            logger.error(f"BM25 load failed: {e}")
        
        # Try to restore indexed_documents map from cache
        indexed_docs_path = Path("data/cache/indexed_documents.json")
        if indexed_docs_path.exists():
            try:
                import json
                with open(indexed_docs_path, 'r', encoding='utf-8') as f:
                    self.indexing_pipeline.indexing_service._indexed_documents = json.load(f)
                result['indexed_documents'] = len(self.indexing_pipeline.indexing_service._indexed_documents)
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Restored {result['indexed_documents']} indexed documents from cache")
            except Exception as e:
                result['errors'].append(f"Indexed documents cache load failed: {e!s}")
        
        # Load the persisted vector store (no-op for Qdrant, file load for Memory)
        try:
            vector_store = self.indexing_pipeline.indexing_service._vector_store_service
            if vector_store is not None:
                load_result = vector_store.load()
                if not load_result.get('success'):
                    result['errors'].append(f"Vector store load: {load_result.get('error')}")
            else:
                result['errors'].append("Vector store service not available")
        except Exception as e:
            result['errors'].append(f"Vector store load failed: {e!s}")
        
        # Refresh vector count from statistics
        pre_count = self.indexing_pipeline.get_statistics().get('vector_count', 0)
        print(f"[DIAGNOSTIC][BootstrapService] vector count before load: {pre_count}")
        stats = self.indexing_pipeline.get_statistics()
        result['vector_count'] = stats.get('vector_count', 0)
        print(f"[DIAGNOSTIC][BootstrapService] vectors restored: {result['vector_count']}")
        
        # Run vector store integrity validation
        try:
            if vector_store is not None:
                integrity = vector_store.integrity_check()
                result['integrity'] = integrity
                if not integrity.get('valid'):
                    result['errors'].append(f"Vector store integrity check failed: {integrity.get('errors', [])}")
        except Exception as e:
            result['errors'].append(f"Vector store integrity check failed: {e!s}")
        
        return result
    
    def bootstrap(self) -> dict[str, Any]:
        """
        Run bootstrap: validate startup, load existing indexes, or build from scratch.
        
        Returns:
            Dictionary with bootstrap results and statistics
        """
        start_time = time.perf_counter()
        
        print("[BOOTSTRAP] Bootstrap started")
        
        # ── STAGE 2 — BOOTSTRAP ───────────────────────────────────────────────
        log_stage_start(2, "BOOTSTRAP", Verbose=self.verbose)
        
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] BootstrapService.bootstrap() invoked")
        
        if self.verbose:
            print("\n" + "="*70)
            print("🔄 Production Bootstrap System")
            print("="*70)
        
        # Discover the current dataset and compute its content hash
        load_result = self.resume_loader.load_resumes(self.base_path)
        current_hash = self.dataset_hash_manager.compute_hash(load_result.file_paths, load_result.csv_path)
        cached_hash = self.dataset_hash_manager.load_hash()
        hash_unchanged = cached_hash is not None and cached_hash == current_hash
        
        print(f"[BOOTSTRAP] Dataset hash: current={current_hash[:16]}... cached={'(none)' if cached_hash is None else cached_hash[:16] + '...'}")
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] hash_unchanged={hash_unchanged}")
        
        # Validate current state before deciding load vs build
        pre_validation = self.validator.validate()
        pre_stats = self.indexing_pipeline.get_statistics()
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Pre-bootstrap stats: indexed_documents={pre_stats['indexed_documents']}, vector_count={pre_stats['vector_count']}, bm25_count={pre_stats['bm25_count']}")
        
        # Detect whether persisted indexes exist
        indexes_exist = self._indexes_exist_on_disk()
        print(f"[BOOTSTRAP] Indexes exist? {indexes_exist}")
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] _indexes_exist_on_disk()={indexes_exist}")
        
        # Reuse persisted indexes only if they exist AND the dataset hash is unchanged
        if indexes_exist and hash_unchanged:
            print("[BOOTSTRAP] Loading persisted indexes")
            load_result_dict = self._load_indexes()
            post_stats = self.indexing_pipeline.get_statistics()
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Post-load stats: indexed_documents={post_stats['indexed_documents']}, vector_count={post_stats['vector_count']}, bm25_count={post_stats['bm25_count']}")
            
            # Persisted indexes are only usable when BOTH sparse (BM25) and
            # dense (vector) stores contain data and the vector store passes
            # its integrity check.  Otherwise, we must rebuild.
            integrity = load_result_dict.get('integrity', {})
            integrity_valid = integrity.get('valid', True)
            if post_stats['bm25_count'] > 0 and post_stats['vector_count'] > 0 and integrity_valid:
                result = {
                    'bootstrapped': True,
                    'success': True,
                    'reason': 'loaded_existing_indexes',
                    'status': BootstrapStatus.LOADED_EXISTING_INDEXES,
                    'hash_unchanged': True,
                    'loaded': load_result_dict,
                    'statistics': post_stats,
                }
                self.dataset_hash_manager.save_hash(current_hash)
            else:
                print(f"[BOOTSTRAP] Persisted indexes incomplete (BM25={post_stats['bm25_count']}, Vectors={post_stats['vector_count']}), rebuilding")
                print("[BOOTSTRAP-TRACE][bootstrap_service.py] DECISION: REBUILD - persisted indexes contained no data")
                
                if self.verbose:
                    print("📋 Persisted indexes empty - starting bootstrap workflow")
                    print()
                
                result = self._run_bootstrap_workflow(
                    load_result=load_result,
                    status=BootstrapStatus.REBUILT_CORRUPTED_INDEXES
                )
                self.dataset_hash_manager.save_hash(current_hash)
        else:
            if not indexes_exist:
                print("[BOOTSTRAP] Building indexes - no persisted indexes found")
            else:
                print("[BOOTSTRAP] Building indexes - dataset changed")
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] DECISION: RUN bootstrap workflow")
            
            if self.verbose:
                print("📋 Starting bootstrap workflow")
                print()
            
            # Run bootstrap workflow
            result = self._run_bootstrap_workflow(
                load_result=load_result,
                status=BootstrapStatus.BUILT_NEW_INDEXES
            )
            self.dataset_hash_manager.save_hash(current_hash)
        
        bootstrap_time = time.perf_counter() - start_time
        result['bootstrap_time_seconds'] = bootstrap_time
        self._last_bootstrap_time = bootstrap_time
        self._last_bootstrap_result = result

        # Final validation after load/build
        post_validation = self.validator.validate()
        post_stats = self.indexing_pipeline.get_statistics()
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Post-bootstrap stats: indexed_documents={post_stats['indexed_documents']}, vector_count={post_stats['vector_count']}, bm25_count={post_stats['bm25_count']}")

        result['validation_result'] = post_validation

        # Loaded indexes are a success even if optional validation has warnings.
        # Only mark as FAILED when the cause was a real exception or corruption.
        if result.get('status') == BootstrapStatus.LOADED_EXISTING_INDEXES:
            result['success'] = True
        elif not post_validation.get('is_valid', False):
            result['success'] = False
            result['status'] = BootstrapStatus.FAILED
            result['reason'] = 'validation_failure'

        if post_stats.get('bm25_count', 0) == 0:
            result['success'] = False
            result['status'] = BootstrapStatus.FAILED
            result['reason'] = 'missing_index'
            raise RuntimeError(
                f"Startup failed: BM25 Count is 0. "
                f"Indexed documents={post_stats.get('indexed_documents', 0)}, "
                f"vectors={post_stats.get('vector_count', 0)}. "
                f"Sparse indexing did not produce any BM25 documents."
            )
        
        # Print startup report
        if self.verbose:
            self.reporter.print_report(result)
            if result.get('validation_result'):
                self.reporter.print_validation(result['validation_result'])
        
        logger.info(f"Bootstrap complete in {bootstrap_time:.2f}s")
        
        # Stage 2 END banner
        final_stats = self.indexing_pipeline.get_statistics()
        log_stage_end(2, "BOOTSTRAP", status="SUCCESS",
                      time_ms=bootstrap_time * 1000,
                      sample={
                          "Indexed_Documents": final_stats.get('indexed_documents', 0),
                          "Vector_Count": final_stats.get('vector_count', 0),
                          "BM25_Count": final_stats.get('bm25_count', 0),
                          "Skipped": "No",
                          "Success": result.get('success', result.get('reason') == 'loaded_existing_indexes'),
                      })
        
        # Fatal validation/index issues stop the startup; warnings are already logged.
        if not result.get('success', False) and result.get('status') == BootstrapStatus.FAILED:
            raise RuntimeError(
                f"Startup failed: {result.get('reason', 'unknown')}. "
                f"Check the validation report above for fatal errors."
            )
        
        print("[BOOTSTRAP] Ready")
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] BootstrapService.bootstrap() returning")
        
        return result
    
    def _run_bootstrap_workflow(
        self,
        load_result: LoadResult | None = None,
        status: BootstrapStatus | None = None,
        use_cache: bool = True,
        clear_existing: bool = False
    ) -> dict[str, Any]:
        """
        Run the complete bootstrap workflow.
        
        This internal method executes the full bootstrap pipeline:
        1. Load resumes (detect CSV)
        2. Process CSV if detected
        3. Index individual files via IndexingPipeline
        4. Validate results
        
        Args:
            load_result: Optional pre-computed LoadResult (avoids double loading)
        
        Returns:
            Dictionary with workflow results
        """
        workflow_start = time.time()
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] _run_bootstrap_workflow() started")
        
        # Step 1: Load resumes (or reuse if provided)
        if load_result is None:
            if self.verbose:
                print("📂 Step 1: Loading Resumes")
            load_result = self.resume_loader.load_resumes(self.base_path)
        elif self.verbose:
            print("📂 Step 1: Resumes (already loaded)")
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] ResumeLoader found files={load_result.total_files_found}, valid={load_result.valid_files}, csv_detected={load_result.csv_detected}")
        
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
        csv_ingestion_result: CSVIngestionResult | None = None
        if load_result.csv_detected and load_result.csv_path:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] CSV detected - running CSV ingestion")
            if self.verbose:
                print("📊 Step 2: CSV Resume Ingestion")
                print("   Loading CSV records → Chunking → Embedding → Vector Store → BM25")
                print()
            
            try:
                csv_path = Path(load_result.csv_path)
                csv_ingestion_result = self.csv_ingestion_service.process_csv_for_indexing(
                    csv_path=csv_path,
                    indexing_service=self.indexing_pipeline.indexing_service,
                    use_cache=use_cache,
                    clear_existing=clear_existing
                )
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] CSV ingestion result: rows={csv_ingestion_result.csv_rows_loaded}, chunks={csv_ingestion_result.chunks_generated}, vectors={csv_ingestion_result.vectors_indexed}, bm25_docs={csv_ingestion_result.bm25_documents_indexed}")
                
                if self.verbose:
                    self.csv_ingestion_service.print_ingestion_results(csv_ingestion_result)
                    print()
                
            except Exception as e:
                logger.error(f"CSV ingestion failed: {e}")
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] CSV ingestion FAILED: {e}")
                if self.verbose:
                    print(f"⚠️  CSV ingestion failed: {e!s}")
                    print()
        else:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] CSV NOT detected - skipping CSV ingestion")
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
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Indexing {load_result.valid_files} individual resume files via IndexingPipeline.index_files()")
            indexing_result = self.indexing_pipeline.index_files(
                load_result.file_paths,
                verbose=self.verbose
            )
            
            if indexing_result:
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] IndexingPipeline result: successful={indexing_result.get('successful')}, failed={indexing_result.get('failed')}, total_chunks={indexing_result.get('total_chunks')}, total_embeddings={indexing_result.get('total_embeddings')}")
            
            if self.verbose:
                print()
        else:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] No valid individual files - skipping file indexing")
            if self.verbose:
                print("⏭️  Step 3: Individual File Indexing Skipped (No valid files)")
                print()
        
        # Check if we have any data to validate
        has_csv_data = csv_ingestion_result and csv_ingestion_result.csv_rows_loaded > 0
        has_file_data = load_result.valid_files > 0
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Data availability: has_csv_data={has_csv_data}, has_file_data={has_file_data}")
        
        if not has_csv_data and not has_file_data:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] Bootstrap FAILED - no resume data found")
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
        
        # Step 4: Index recovery (bootstrap-only)
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Running IndexRecoveryService.recover_indexes()")
        if self.verbose:
            print("🔄 Step 4: Index Recovery")
        
        recovery_result = self.index_recovery_service.recover_indexes()
        if recovery_result["status"] == "recovered":
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Index recovery rebuilt BM25: pre={recovery_result['pre_recovery']['bm25_count']} -> post={recovery_result['post_recovery']['bm25_count']}")
        else:
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Index recovery not needed: {recovery_result['pre_recovery']}")
        
        # Step 5: Persist the vector store and validate its integrity
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Persisting vector store and validating integrity")
        if self.verbose:
            print("💾 Step 5: Vector Store Persistence")
        
        try:
            vector_store = self.indexing_pipeline.indexing_service._vector_store_service
            if vector_store is not None:
                save_result = vector_store.save()
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store save result: {save_result}")
                integrity = vector_store.integrity_check()
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store integrity after save: valid={integrity.get('valid')}, count={integrity.get('count')}")
        except Exception as e:
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store save/integrity check failed: {e}")
        
        # Step 6: Validate
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Running StartupValidator.validate()")
        if self.verbose:
            print("✅ Step 6: Validation")
        
        validation_result = self.validator.validate()
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Validation result: is_valid={validation_result.get('is_valid')}, errors={len(validation_result.get('errors', []))}")
        
        workflow_time = time.time() - workflow_start
        
        # If the caller did not provide a status, label based on whether this
        # is a first-time build or a rebuild of corrupted/empty indexes.
        if status is None:
            status = BootstrapStatus.REBUILT_CORRUPTED_INDEXES if load_result and self._indexes_exist_on_disk() else BootstrapStatus.BUILT_NEW_INDEXES
        
        result = {
            'success': validation_result['is_valid'],
            'status': status,
            'reason': status.value.lower(),
            'load_result': load_result,
            'csv_ingestion_result': csv_ingestion_result,
            'indexing_result': indexing_result,
            'validation_result': validation_result,
            'workflow_time_seconds': workflow_time
        }
        
        return result
    
    def rebuild(self) -> dict[str, Any]:
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
        
        # Run bootstrap workflow without cache and with explicit index clearing
        result = self._run_bootstrap_workflow(
            status=BootstrapStatus.REBUILT_CORRUPTED_INDEXES,
            use_cache=False,
            clear_existing=True
        )
        result['rebuild'] = True
        result['rebuild_result'] = rebuild_result
        
        # Persist the new dataset hash
        if result.get('load_result'):
            lr = result['load_result']
            new_hash = self.dataset_hash_manager.compute_hash(lr.file_paths, lr.csv_path)
            self.dataset_hash_manager.save_hash(new_hash)
        
        bootstrap_time = time.time() - start_time
        result['bootstrap_time_seconds'] = bootstrap_time
        self._last_bootstrap_time = bootstrap_time
        self._last_bootstrap_result = result
        
        # Print startup report
        if self.verbose:
            self.reporter.print_report(result)
        
        logger.info(f"Rebuild complete in {bootstrap_time:.2f}s")
        
        return result
    
    def validate(self) -> dict[str, Any]:
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
    
    def status(self) -> dict[str, Any]:
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
    
    def _indexes_exist_on_disk(self) -> bool:
        """Check whether persisted BM25 index files exist on disk."""
        bm25_metadata_path = Path("data/indexes/bm25/metadata.json")
        return bm25_metadata_path.exists()
    
    def _load_indexes(self) -> dict[str, Any]:
        """Load persisted indexes from disk into the in-memory services."""
        result = {
            'loaded': True,
            'bm25_loaded': False,
            'vector_count': 0,
            'indexed_documents': 0,
            'errors': []
        }
        
        try:
            bm25_index_path = Path("data/indexes/bm25")
            bm25_index = self.indexing_pipeline.indexing_service.get_bm25_index()
            if bm25_index is not None:
                bm25_index.load_from_disk(bm25_index_path)
                result['bm25_loaded'] = True
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] BM25 index loaded from {bm25_index_path}")
            else:
                result['errors'].append("BM25 index instance not available")
        except IncompatibleIndexError as e:
            logger.error(f"Incompatible persisted BM25 index detected: {e}")
            result['errors'].append(f"Incompatible persisted index: {e!s}")
            self._delete_persisted_indexes()
        except Exception as e:
            result['errors'].append(f"BM25 load failed: {e!s}")
            logger.error(f"BM25 load failed: {e}")
        
        # Try to restore indexed_documents map from cache
        indexed_docs_path = Path("data/cache/indexed_documents.json")
        if indexed_docs_path.exists():
            try:
                import json
                with open(indexed_docs_path, 'r', encoding='utf-8') as f:
                    self.indexing_pipeline.indexing_service._indexed_documents = json.load(f)
                result['indexed_documents'] = len(self.indexing_pipeline.indexing_service._indexed_documents)
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Restored {result['indexed_documents']} indexed documents from cache")
            except Exception as e:
                result['errors'].append(f"Indexed documents cache load failed: {e!s}")
        
        # Refresh vector count from statistics
        stats = self.indexing_pipeline.get_statistics()
        result['vector_count'] = stats.get('vector_count', 0)

        return result

    def _delete_persisted_indexes(self) -> None:
        """Delete persisted BM25 index and vector cache to force a clean rebuild."""
        import shutil
        paths = [
            Path("data/indexes/bm25"),
            Path("data/cache"),
        ]
        for p in paths:
            if p.exists():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Deleted incompatible persisted data: {p}")
                except Exception as e:
                    logger.error(f"Failed to delete {p}: {e}")
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
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] _run_bootstrap_workflow() started")
        
        # Step 1: Load resumes
        if self.verbose:
            print("📂 Step 1: Loading Resumes")
        
        load_result = self.resume_loader.load_resumes(self.base_path)
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] ResumeLoader found files={load_result.total_files_found}, valid={load_result.valid_files}, csv_detected={load_result.csv_detected}")
        
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
        csv_ingestion_result: CSVIngestionResult | None = None
        if load_result.csv_detected and load_result.csv_path:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] CSV detected - running CSV ingestion")
            if self.verbose:
                print("📊 Step 2: CSV Resume Ingestion")
                print("   Loading CSV records → Chunking → Embedding → Vector Store → BM25")
                print()
            
            try:
                csv_path = Path(load_result.csv_path)
                csv_ingestion_result = self.csv_ingestion_service.process_csv_for_indexing(
                    csv_path=csv_path,
                    indexing_service=self.indexing_pipeline.indexing_service,
                    use_cache=use_cache,
                    clear_existing=clear_existing
                )
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] CSV ingestion result: rows={csv_ingestion_result.csv_rows_loaded}, chunks={csv_ingestion_result.chunks_generated}, vectors={csv_ingestion_result.vectors_indexed}, bm25_docs={csv_ingestion_result.bm25_documents_indexed}")
                
                if self.verbose:
                    self.csv_ingestion_service.print_ingestion_results(csv_ingestion_result)
                    print()
                
            except Exception as e:
                logger.error(f"CSV ingestion failed: {e}")
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] CSV ingestion FAILED: {e}")
                if self.verbose:
                    print(f"⚠️  CSV ingestion failed: {e!s}")
                    print()
        else:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] CSV NOT detected - skipping CSV ingestion")
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
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Indexing {load_result.valid_files} individual resume files via IndexingPipeline.index_files()")
            indexing_result = self.indexing_pipeline.index_files(
                load_result.file_paths,
                verbose=self.verbose
            )
            
            if indexing_result:
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] IndexingPipeline result: successful={indexing_result.get('successful')}, failed={indexing_result.get('failed')}, total_chunks={indexing_result.get('total_chunks')}, total_embeddings={indexing_result.get('total_embeddings')}")
            
            if self.verbose:
                print()
        else:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] No valid individual files - skipping file indexing")
            if self.verbose:
                print("⏭️  Step 3: Individual File Indexing Skipped (No valid files)")
                print()
        
        # Check if we have any data to validate
        has_csv_data = csv_ingestion_result and csv_ingestion_result.csv_rows_loaded > 0
        has_file_data = load_result.valid_files > 0
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Data availability: has_csv_data={has_csv_data}, has_file_data={has_file_data}")
        
        if not has_csv_data and not has_file_data:
            print("[BOOTSTRAP-TRACE][bootstrap_service.py] Bootstrap FAILED - no resume data found")
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
        
        # Step 4: Index recovery (bootstrap-only)
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Running IndexRecoveryService.recover_indexes()")
        if self.verbose:
            print("🔄 Step 4: Index Recovery")
        
        recovery_result = self.index_recovery_service.recover_indexes()
        if recovery_result["status"] == "recovered":
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Index recovery rebuilt BM25: pre={recovery_result['pre_recovery']['bm25_count']} -> post={recovery_result['post_recovery']['bm25_count']}")
        else:
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Index recovery not needed: {recovery_result['pre_recovery']}")
        
        # Step 5: Persist the vector store and validate its integrity
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Persisting vector store and validating integrity")
        if self.verbose:
            print("💾 Step 5: Vector Store Persistence")
        
        try:
            vector_store = self.indexing_pipeline.indexing_service._vector_store_service
            if vector_store is not None:
                save_result = vector_store.save()
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store save result: {save_result}")
                integrity = vector_store.integrity_check()
                print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store integrity after save: valid={integrity.get('valid')}, count={integrity.get('count')}")
        except Exception as e:
            print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Vector store save/integrity check failed: {e}")
        
        # Step 6: Validate
        print("[BOOTSTRAP-TRACE][bootstrap_service.py] Running StartupValidator.validate()")
        if self.verbose:
            print("✅ Step 6: Validation")
        
        validation_result = self.validator.validate()
        print(f"[BOOTSTRAP-TRACE][bootstrap_service.py] Validation result: is_valid={validation_result.get('is_valid')}, errors={len(validation_result.get('errors', []))}")
        
        workflow_time = time.time() - workflow_start
        
        # If the caller did not provide a status, label based on whether this
        # is a first-time build or a rebuild of corrupted/empty indexes.
        if status is None:
            status = BootstrapStatus.REBUILT_CORRUPTED_INDEXES if load_result and self._indexes_exist_on_disk() else BootstrapStatus.BUILT_NEW_INDEXES
        
        result = {
            'success': validation_result['is_valid'],
            'status': status,
            'reason': status.value.lower(),
            'load_result': load_result,
            'csv_ingestion_result': csv_ingestion_result,
            'indexing_result': indexing_result,
            'validation_result': validation_result,
            'workflow_time_seconds': workflow_time
        }
        
        return result
