"""
Startup Validator Module for Bootstrap System.

This module provides validation logic to verify the system state after
bootstrap. It checks that resumes are loaded, chunks are created, vectors
are indexed, BM25 documents are indexed, and retrieval services are healthy.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StartupValidator:
    """
    Validator for system state after bootstrap.
    
    This class provides validation methods to ensure the system is in a
    healthy state with indexed data and functional retrieval services.
    
    Validation Checks:
        - Resumes loaded (indexed_documents > 0)
        - Chunks created (implied by indexed_documents)
        - Vectors indexed (vector_count > 0)
        - BM25 documents indexed (bm25_count > 0)
        - Retrieval services healthy
    """
    
    def __init__(self, indexing_pipeline):
        """
        Initialize the startup validator.
        
        Args:
            indexing_pipeline: IndexingPipeline instance to validate
        """
        self.indexing_pipeline = indexing_pipeline
        logger.info("StartupValidator initialized")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the current system state.
        
        This method performs comprehensive validation of the system state
        after bootstrap to ensure everything is properly indexed and functional.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': False,
            'checks': {},
            'statistics': {},
            'errors': [],
            'warnings': []
        }
        
        # Get current statistics
        stats = self.indexing_pipeline.get_statistics()
        validation_result['statistics'] = stats
        
        # Check 1: Documents indexed
        docs_check = self._check_documents_indexed(stats)
        validation_result['checks']['documents_indexed'] = docs_check
        if not docs_check['passed']:
            validation_result['errors'].append(docs_check['message'])
        
        # Check 2: Vectors indexed
        vectors_check = self._check_vectors_indexed(stats)
        validation_result['checks']['vectors_indexed'] = vectors_check
        if not vectors_check['passed']:
            validation_result['errors'].append(vectors_check['message'])
        
        # Check 3: BM25 documents indexed
        bm25_check = self._check_bm25_indexed(stats)
        validation_result['checks']['bm25_indexed'] = bm25_check
        
        # Temporary identity logging
        try:
            bm25_id = stats.get('bm25_id')
            logger.info(f"[IDENTITY] StartupValidator bm25_id={bm25_id}")
        except Exception:
            pass

        if not bm25_check['passed']:
            validation_result['errors'].append(bm25_check['message'])
        
        # Check 4: Retrieval services healthy
        services_check = self._check_services_healthy()
        validation_result['checks']['services_healthy'] = services_check
        if not services_check['passed']:
            validation_result['errors'].append(services_check['message'])
        
        # Check 5: Consistency between counts
        consistency_check = self._check_consistency(stats)
        validation_result['checks']['consistency'] = consistency_check
        if not consistency_check['passed']:
            validation_result['warnings'].append(consistency_check['message'])
        
        # Overall validation result
        all_critical_checks_passed = all([
            docs_check['passed'],
            vectors_check['passed'],
            bm25_check['passed'],
            services_check['passed']
        ])
        
        validation_result['is_valid'] = all_critical_checks_passed
        
        if validation_result['is_valid']:
            logger.info("System validation passed - all checks successful")
        else:
            logger.warning(f"System validation failed - {len(validation_result['errors'])} errors")
        
        return validation_result
    
    def is_healthy(self) -> bool:
        """
        Quick health check of the system.
        
        This method performs a quick check to determine if the system
        is healthy without detailed validation.
        
        Returns:
            True if system is healthy, False otherwise
        """
        stats = self.indexing_pipeline.get_statistics()
        
        # Quick check: all counts should be > 0
        return (
            stats['indexed_documents'] > 0 and
            stats['vector_count'] > 0 and
            stats['bm25_count'] > 0
        )
    
    def _check_documents_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check that documents are indexed.
        
        Args:
            stats: System statistics from indexing pipeline
        
        Returns:
            Dictionary with check result
        """
        doc_count = stats.get('indexed_documents', 0)
        
        check_result = {
            'passed': doc_count > 0,
            'count': doc_count,
            'message': f"Documents indexed: {doc_count}"
        }
        
        if doc_count == 0:
            check_result['message'] = "No documents indexed - bootstrap may have failed"
            logger.warning(check_result['message'])
        else:
            logger.debug(check_result['message'])
        
        return check_result
    
    def _check_vectors_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check that vectors are indexed.
        
        Args:
            stats: System statistics from indexing pipeline
        
        Returns:
            Dictionary with check result
        """
        vector_count = stats.get('vector_count', 0)
        
        check_result = {
            'passed': vector_count > 0,
            'count': vector_count,
            'message': f"Vectors indexed: {vector_count}"
        }
        
        if vector_count == 0:
            check_result['message'] = "No vectors indexed - embedding generation may have failed"
            logger.warning(check_result['message'])
        else:
            logger.debug(check_result['message'])
        
        return check_result
    
    def _check_bm25_indexed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check that BM25 documents are indexed.
        
        Args:
            stats: System statistics from indexing pipeline
        
        Returns:
            Dictionary with check result
        """
        bm25_count = stats.get('bm25_count', 0)
        
        check_result = {
            'passed': bm25_count > 0,
            'count': bm25_count,
            'message': f"BM25 documents indexed: {bm25_count}"
        }
        
        if bm25_count == 0:
            check_result['message'] = "No BM25 documents indexed - BM25 indexing may have failed"
            logger.warning(check_result['message'])
        else:
            logger.debug(check_result['message'])
        
        return check_result
    
    def _check_services_healthy(self) -> Dict[str, Any]:
        """
        Check that retrieval services are healthy.
        
        Returns:
            Dictionary with check result
        """
        check_result = {
            'passed': True,
            'message': "Retrieval services healthy"
        }
        
        # Check if indexing service is available
        try:
            if self.indexing_pipeline.indexing_service is None:
                check_result['passed'] = False
                check_result['message'] = "Indexing service not available"
                logger.warning(check_result['message'])
            else:
                logger.debug("Indexing service available")
        except Exception as e:
            check_result['passed'] = False
            check_result['message'] = f"Indexing service check failed: {str(e)}"
            logger.warning(check_result['message'])
        
        return check_result
    
    def _check_consistency(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check consistency between different index counts.
        
        Args:
            stats: System statistics from indexing pipeline
        
        Returns:
            Dictionary with check result
        """
        doc_count = stats.get('indexed_documents', 0)
        vector_count = stats.get('vector_count', 0)
        bm25_count = stats.get('bm25_count', 0)
        
        check_result = {
            'passed': True,
            'message': "Index counts consistent"
        }
        
        # Check if all counts are zero or all are non-zero
        all_zero = (doc_count == 0 and vector_count == 0 and bm25_count == 0)
        all_non_zero = (doc_count > 0 and vector_count > 0 and bm25_count > 0)
        
        if not all_zero and not all_non_zero:
            check_result['passed'] = False
            check_result['message'] = (
                f"Inconsistent index counts: "
                f"docs={doc_count}, vectors={vector_count}, bm25={bm25_count}"
            )
            logger.warning(check_result['message'])
        else:
            logger.debug("Index counts consistent")
        
        return check_result
