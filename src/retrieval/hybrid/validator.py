"""
Validator for Hybrid Retrieval Service.

This module provides validation for hybrid retrieval results, ensuring
data integrity and consistency across the fusion process.

Architecture Notes:
- Validates duplicate candidates
- Validates ranks (dense, sparse, final)
- Validates metadata completeness
- Validates matched chunks
- Validates RRF scores

SOLID Principles Applied:
- Single Responsibility: Handles only validation
- Open/Closed: Open for new validation rules
- Dependency Inversion: Depends on abstract interfaces
"""

import logging
from typing import List, Dict, Any, Set

from .schema import HybridSearchResult

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class HybridRetrievalValidator:
    """
    Validator for hybrid retrieval results.
    
    This class provides validation methods for hybrid retrieval results,
    ensuring data integrity and consistency across the fusion process.
    
    Validation Rules:
        - No duplicate candidates by chunk_id
        - Ranks must be non-negative
        - Metadata must be present
        - Matched chunks must be valid
        - RRF scores must be non-negative
    """
    
    def __init__(self):
        """Initialize the validator."""
        logger.info("HybridRetrievalValidator initialized")
    
    def validate_results(
        self,
        results: List[HybridSearchResult],
        strict: bool = False
    ) -> bool:
        """
        Validate hybrid retrieval results.
        
        This method validates the complete set of hybrid retrieval results,
        checking for duplicate candidates, invalid ranks, missing metadata,
        invalid scores, and missing chunks.
        
        Args:
            results: List of HybridSearchResult objects
            strict: If True, raise exceptions on validation errors
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If strict mode is enabled and validation fails
        """
        errors = []
        
        # Check for duplicate candidates
        duplicate_error = self.validate_no_duplicates(results)
        if duplicate_error:
            errors.append(duplicate_error)
        
        # Check for invalid ranks
        rank_errors = self.validate_ranks(results)
        errors.extend(rank_errors)
        
        # Check for missing metadata
        metadata_errors = self.validate_metadata(results)
        errors.extend(metadata_errors)
        
        # Check for missing chunks
        chunk_errors = self.validate_chunks(results)
        errors.extend(chunk_errors)
        
        # Check for invalid scores
        score_errors = self.validate_scores(results)
        errors.extend(score_errors)
        
        if errors:
            error_message = "Validation failed:\n" + "\n".join(errors)
            logger.error(error_message)
            
            if strict:
                raise ValidationError(error_message)
            
            return False
        
        logger.info(f"Validation passed for {len(results)} results")
        return True
    
    def validate_no_duplicates(
        self,
        results: List[HybridSearchResult]
    ) -> str:
        """
        Validate that there are no duplicate candidates.
        
        Args:
            results: List of HybridSearchResult objects
            
        Returns:
            Error message if duplicates found, empty string otherwise
        """
        chunk_ids: Set[str] = set()
        duplicates: Set[str] = set()
        
        for result in results:
            if result.chunk_id in chunk_ids:
                duplicates.add(result.chunk_id)
            chunk_ids.add(result.chunk_id)
        
        if duplicates:
            error = f"Duplicate candidates found: {duplicates}"
            logger.error(error)
            return error
        
        return ""
    
    def validate_ranks(
        self,
        results: List[HybridSearchResult]
    ) -> List[str]:
        """
        Validate that ranks are non-negative.
        
        Args:
            results: List of HybridSearchResult objects
            
        Returns:
            List of error messages for invalid ranks
        """
        errors = []
        
        for result in results:
            # Validate dense rank
            if result.dense_rank is not None and result.dense_rank < 0:
                error = f"Invalid dense_rank for {result.chunk_id}: {result.dense_rank}"
                errors.append(error)
            
            # Validate sparse rank
            if result.sparse_rank is not None and result.sparse_rank < 0:
                error = f"Invalid sparse_rank for {result.chunk_id}: {result.sparse_rank}"
                errors.append(error)
            
            # Validate final rank
            if result.rank < 0:
                error = f"Invalid rank for {result.chunk_id}: {result.rank}"
                errors.append(error)
        
        if errors:
            logger.error(f"Rank validation failed: {len(errors)} errors")
        
        return errors
    
    def validate_metadata(
        self,
        results: List[HybridSearchResult]
    ) -> List[str]:
        """
        Validate that metadata is present.
        
        Args:
            results: List of HybridSearchResult objects
            
        Returns:
            List of error messages for missing metadata
        """
        errors = []
        
        for result in results:
            if not result.metadata:
                error = f"Missing metadata for {result.chunk_id}"
                errors.append(error)
        
        if errors:
            logger.error(f"Metadata validation failed: {len(errors)} errors")
        
        return errors
    
    def validate_chunks(
        self,
        results: List[HybridSearchResult]
    ) -> List[str]:
        """
        Validate that matched chunks are present and valid.
        
        Args:
            results: List of HybridSearchResult objects
            
        Returns:
            List of error messages for missing or invalid chunks
        """
        errors = []
        
        for result in results:
            if not result.matched_chunks:
                error = f"No matched chunks for {result.chunk_id}"
                errors.append(error)
                continue
            
            # Validate each matched chunk
            for chunk in result.matched_chunks:
                if not chunk.chunk_id:
                    error = f"Missing chunk_id in matched chunk for {result.chunk_id}"
                    errors.append(error)
                
                if not chunk.section:
                    error = f"Missing section in matched chunk for {result.chunk_id}"
                    errors.append(error)
                
                if not chunk.matched_text:
                    error = f"Missing matched_text in matched chunk for {result.chunk_id}"
                    errors.append(error)
                
                if chunk.score < 0:
                    error = f"Invalid score in matched chunk for {result.chunk_id}: {chunk.score}"
                    errors.append(error)
        
        if errors:
            logger.error(f"Chunk validation failed: {len(errors)} errors")
        
        return errors
    
    def validate_scores(
        self,
        results: List[HybridSearchResult]
    ) -> List[str]:
        """
        Validate that RRF scores are non-negative.
        
        Args:
            results: List of HybridSearchResult objects
            
        Returns:
            List of error messages for invalid scores
        """
        errors = []
        
        for result in results:
            if result.rrf_score < 0:
                error = f"Invalid rrf_score for {result.chunk_id}: {result.rrf_score}"
                errors.append(error)
        
        if errors:
            logger.error(f"Score validation failed: {len(errors)} errors")
        
        return errors
    
    def validate_fusion_inputs(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate fusion inputs.
        
        This method validates the inputs to the fusion process, ensuring
        that the dense and sparse results are properly formatted.
        
        Args:
            dense_results: List of dense retrieval results
            sparse_results: List of sparse retrieval results
            
        Returns:
            True if validation passes, False otherwise
        """
        errors = []
        
        # Validate dense results
        for idx, result in enumerate(dense_results):
            if "chunk_id" not in result:
                errors.append(f"Missing chunk_id in dense result {idx}")
            
            if "rank" not in result:
                errors.append(f"Missing rank in dense result {idx}")
        
        # Validate sparse results
        for idx, result in enumerate(sparse_results):
            if "chunk_id" not in result:
                errors.append(f"Missing chunk_id in sparse result {idx}")
            
            if "rank" not in result:
                errors.append(f"Missing rank in sparse result {idx}")
        
        if errors:
            logger.error(f"Fusion input validation failed: {len(errors)} errors")
            return False
        
        logger.info("Fusion input validation passed")
        return True
