"""
Startup Report Module for Bootstrap System.

This module provides reporting functionality for the bootstrap system,
generating and printing startup statistics and validation results.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StartupReport:
    """
    Reporter for bootstrap startup statistics.
    
    This class generates and prints formatted reports for the bootstrap
    system, including indexed document counts, chunk counts, vector counts,
    BM25 counts, embedding model information, index build time, and memory usage.
    """
    
    def print_report(self, bootstrap_result: Dict[str, Any]) -> None:
        """
        Print a formatted bootstrap startup report.
        
        Args:
            bootstrap_result: Result dictionary from bootstrap workflow
        """
        print("\n" + "="*70)
        print("📊 Bootstrap Startup Report")
        print("="*70)
        
        # Load results
        load_result = bootstrap_result.get('load_result')
        indexing_result = bootstrap_result.get('indexing_result')
        validation_result = bootstrap_result.get('validation_result')
        workflow_time = bootstrap_result.get('workflow_time_seconds', 0)
        
        # Print load statistics
        if load_result:
            print("\n📂 Resume Loading:")
            print(f"   Files Found: {load_result.total_files_found}")
            print(f"   Valid Files: {load_result.valid_files}")
            print(f"   Invalid Files: {load_result.invalid_files}")
            print(f"   Skipped Files: {load_result.skipped_files}")
            print(f"   Load Time: {load_result.load_time_seconds:.2f}s")
        
        # Print indexing statistics
        if indexing_result:
            print("\n📝 Indexing:")
            ingestion = indexing_result.get('ingestion')
            indexing = indexing_result.get('indexing')
            
            if ingestion:
                print(f"   Files Processed: {ingestion.valid_files}")
            
            if indexing:
                print(f"   Successful: {indexing.get('successful', 0)}")
                print(f"   Failed: {indexing.get('failed', 0)}")
                print(f"   Total Chunks: {indexing.get('total_chunks', 0)}")
                print(f"   Total Embeddings: {indexing.get('total_embeddings', 0)}")
        
        # Print final statistics
        if validation_result:
            stats = validation_result.get('statistics', {})
            print("\n📊 Final Statistics:")
            print(f"   Indexed Documents: {stats.get('indexed_documents', 0)}")
            print(f"   Vector Count: {stats.get('vector_count', 0)}")
            print(f"   BM25 Count: {stats.get('bm25_count', 0)}")
            
            # Add BM25 diagnostics from the already-computed statistics
            try:
                bm25_stats = stats.get('bm25_stats')
                if bm25_stats:
                    if hasattr(bm25_stats, 'num_documents'):
                        bm25_docs = bm25_stats.num_documents
                        bm25_vocab = bm25_stats.vocabulary_size
                        bm25_avg = bm25_stats.average_document_length
                    else:
                        bm25_docs = bm25_stats.get('num_documents', 0)
                        bm25_vocab = bm25_stats.get('vocabulary_size', len(bm25_stats.get('vocabulary', set())))
                        bm25_avg = bm25_stats.get('avg_doc_length', bm25_stats.get('average_document_length', 0.0))
                    print(f"   BM25 loaded documents: {bm25_docs}")
                    print(f"   BM25 vocabulary size: {bm25_vocab}")
                    print(f"   BM25 avg doc length: {bm25_avg:.2f}")
            except Exception as e:
                print(f"   BM25 diagnostics unavailable: {e}")
        
        # Print embedding model info
        print("\n🤖 Embedding Model:")
        try:
            from ..embeddings.model_loader import get_model_loader
            model_loader = get_model_loader()
            diagnostics = model_loader.get_diagnostics()
            print(f"   Model: {diagnostics.get('model_name', 'N/A')}")
            print(f"   Device: {diagnostics.get('device', 'N/A')}")
            print(f"   Load Time: {diagnostics.get('load_time', 0):.2f}s")
            print(f"   Memory Usage: {diagnostics.get('memory_usage_mb', 0):.2f} MB")
        except Exception as e:
            print(f"   Model info unavailable: {str(e)}")
        
        # Print workflow time
        print(f"\n⏱️  Total Workflow Time: {workflow_time:.2f}s")
        
        # Print validation result
        if validation_result:
            is_valid = validation_result.get('is_valid', False)
            print(f"\n✅ Validation: {'PASSED' if is_valid else 'FAILED'}")
            
            if not is_valid:
                errors = validation_result.get('errors', [])
                if errors:
                    print("\n❌ Errors:")
                    for error in errors:
                        print(f"   - {error}")
        
        # Print final status
        print("\n" + "="*70)
        if bootstrap_result.get('success', False):
            print("🚀 Bootstrap Complete")
            print(f"Indexed Documents: {validation_result.get('statistics', {}).get('indexed_documents', 0)}")
            print(f"Vectors: {validation_result.get('statistics', {}).get('vector_count', 0)}")
            print(f"BM25 Docs: {validation_result.get('statistics', {}).get('bm25_count', 0)}")
            print("System Ready")
        else:
            print("❌ Bootstrap Failed")
            reason = bootstrap_result.get('reason', 'Unknown')
            print(f"Reason: {reason}")
        print("="*70 + "\n")
    
    def print_validation(self, validation_result: Dict[str, Any]) -> None:
        """
        Print a formatted validation report.
        
        Args:
            validation_result: Result dictionary from validation
        """
        print("\n" + "="*70)
        print("🔍 System Validation Report")
        print("="*70)
        
        is_valid = validation_result.get('is_valid', False)
        checks = validation_result.get('checks', {})
        stats = validation_result.get('statistics', {})
        errors = validation_result.get('errors', [])
        warnings = validation_result.get('warnings', [])
        
        # Print overall status
        print(f"\nOverall Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        # Print individual checks
        print("\nValidation Checks:")
        for check_name, check_result in checks.items():
            status = "✅ PASS" if check_result.get('passed', False) else "❌ FAIL"
            print(f"   {status} - {check_name}")
            if 'message' in check_result:
                print(f"      {check_result['message']}")
        
        # Print statistics
        print("\nCurrent Statistics:")
        print(f"   Indexed Documents: {stats.get('indexed_documents', 0)}")
        print(f"   Vector Count: {stats.get('vector_count', 0)}")
        print(f"   BM25 Count: {stats.get('bm25_count', 0)}")
        
        # Print errors
        if errors:
            print("\n❌ Errors:")
            for error in errors:
                print(f"   - {error}")
        
        # Print warnings
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("="*70 + "\n")
    
    def print_status(self, status_info: Dict[str, Any]) -> None:
        """
        Print a formatted system status report.
        
        Args:
            status_info: Status information dictionary
        """
        print("\n" + "="*70)
        print("📊 System Status")
        print("="*70)
        
        stats = status_info.get('statistics', {})
        is_bootstrapped = status_info.get('is_bootstrapped', False)
        is_healthy = status_info.get('is_healthy', False)
        last_bootstrap = status_info.get('last_bootstrap_time')
        
        print(f"\nBootstrapped: {'✅ Yes' if is_bootstrapped else '❌ No'}")
        print(f"Healthy: {'✅ Yes' if is_healthy else '❌ No'}")
        
        print("\nIndex Statistics:")
        print(f"   Indexed Documents: {stats.get('indexed_documents', 0)}")
        print(f"   Vector Count: {stats.get('vector_count', 0)}")
        print(f"   BM25 Count: {stats.get('bm25_count', 0)}")
        
        if last_bootstrap:
            print(f"\nLast Bootstrap: {last_bootstrap:.2f}s ago")
        else:
            print("\nLast Bootstrap: Never")
        
        print("="*70 + "\n")
    
    def generate_summary(self, bootstrap_result: Dict[str, Any]) -> str:
        """
        Generate a summary string of the bootstrap result.
        
        Args:
            bootstrap_result: Result dictionary from bootstrap workflow
        
        Returns:
            Summary string
        """
        load_result = bootstrap_result.get('load_result')
        validation_result = bootstrap_result.get('validation_result')
        
        if not bootstrap_result.get('success', False):
            return f"Bootstrap failed: {bootstrap_result.get('reason', 'Unknown')}"
        
        doc_count = validation_result.get('statistics', {}).get('indexed_documents', 0)
        vector_count = validation_result.get('statistics', {}).get('vector_count', 0)
        bm25_count = validation_result.get('statistics', {}).get('bm25_count', 0)
        
        return (
            f"Bootstrap Complete - "
            f"Documents: {doc_count}, "
            f"Vectors: {vector_count}, "
            f"BM25 Docs: {bm25_count}"
        )
