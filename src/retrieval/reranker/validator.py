"""
Validator module for Reranker.

This module provides comprehensive validation for reranker inputs and outputs,
ensuring data integrity and preventing common errors.

Architecture Notes:
- Input Validation: Validates queries, candidates, and metadata
- Output Validation: Validates rerank scores and results
- Error Prevention: Catches common issues before processing
- Clear Error Messages: Provides actionable error messages

Validation Categories:
1. Query Validation: Empty queries, invalid characters
2. Candidate Validation: Empty candidates, duplicate candidates, missing metadata
3. Score Validation: NaN scores, infinite scores, invalid ranges
4. Model Output Validation: Invalid model outputs, malformed results

Cross-Encoder Validation Importance:
Cross-encoder models can produce unexpected outputs due to:
- Numerical instability in model inference
- Edge cases in input data
- Model architecture differences
- Tokenization issues

Comprehensive validation prevents these issues from propagating through
the system and causing downstream errors.

SOLID Principles Applied:
- Single Responsibility: Only handles validation
- Open/Closed: Can be extended with new validation rules
- Dependency Inversion: Depends on validation abstraction
- Interface Segregation: Focused validation interface
"""

import logging
import math
import re
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """
    Custom exception for validation errors.
    
    This exception provides detailed error messages for validation failures,
    making it easier to diagnose and fix issues.
    
    Attributes:
        message: Error message
        field: Field that failed validation (if applicable)
        value: Value that failed validation (if applicable)
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.field:
            return f"ValidationError: {self.field} - {self.message}"
        return f"ValidationError: {self.message}"


class RerankerValidator:
    """
    Validator for reranker inputs and outputs.
    
    This class provides comprehensive validation for reranker operations,
    including query validation, candidate validation, score validation,
    and model output validation.
    
    Architecture Pattern: Validator Pattern
    - Centralized validation logic
    - Clear error messages
    - Prevents invalid data from entering the system
    - Validates both inputs and outputs
    
    Validation Rules:
    1. Query must not be empty or whitespace-only
    2. Query must not contain only special characters
    3. Candidates list must not be empty
    4. Candidates must not have duplicate chunk_ids
    5. Candidates must have required metadata fields
    6. Scores must be valid numbers (not NaN or infinite)
    7. Scores must be within expected ranges
    8. Model outputs must be properly formatted
    
    Attributes:
        allow_empty_query: Whether to allow empty queries (default: False)
        allow_duplicates: Whether to allow duplicate candidates (default: False)
        min_score: Minimum valid score (default: -100.0)
        max_score: Maximum valid score (default: 100.0)
    """
    
    def __init__(
        self,
        allow_empty_query: bool = False,
        allow_duplicates: bool = False,
        min_score: float = -100.0,
        max_score: float = 100.0
    ):
        """
        Initialize the reranker validator.
        
        Args:
            allow_empty_query: Whether to allow empty queries (default: False)
            allow_duplicates: Whether to allow duplicate candidates (default: False)
            min_score: Minimum valid score (default: -100.0)
            max_score: Maximum valid score (default: 100.0)
        """
        self.allow_empty_query = allow_empty_query
        self.allow_duplicates = allow_duplicates
        self.min_score = min_score
        self.max_score = max_score
        
        logger.info(
            f"RerankerValidator initialized: allow_empty_query={allow_empty_query}, "
            f"allow_duplicates={allow_duplicates}, score_range=[{min_score}, {max_score}]"
        )
    
    def validate_query(self, query: str) -> None:
        """
        Validate a search query.
        
        This method ensures that the query is not empty, not whitespace-only,
        and contains at least some alphanumeric characters.
        
        Args:
            query: The search query to validate
            
        Raises:
            ValidationError: If query is invalid
        """
        if not isinstance(query, str):
            raise ValidationError(
                f"Query must be a string, got {type(query).__name__}",
                field="query",
                value=query
            )
        
        if not query:
            if not self.allow_empty_query:
                raise ValidationError(
                    "Query cannot be empty",
                    field="query",
                    value=query
                )
            return
        
        # Check if query is only whitespace
        if not query.strip():
            raise ValidationError(
                "Query cannot be whitespace-only",
                field="query",
                value=query
            )
        
        # Check if query contains at least one alphanumeric character
        if not re.search(r'[a-zA-Z0-9]', query):
            raise ValidationError(
                "Query must contain at least one alphanumeric character",
                field="query",
                value=query
            )
        
        logger.debug(f"Query validation passed: {query[:50]}...")
    
    def validate_candidates(self, candidates: List[Any]) -> None:
        """
        Validate a list of candidates.
        
        This method ensures that the candidates list is not empty, does not
        contain duplicates (if not allowed), and has valid metadata.
        
        Args:
            candidates: List of candidates to validate
            
        Raises:
            ValidationError: If candidates are invalid
        """
        if not isinstance(candidates, list):
            raise ValidationError(
                f"Candidates must be a list, got {type(candidates).__name__}",
                field="candidates"
            )
        
        if not candidates:
            raise ValidationError(
                "Candidates list cannot be empty",
                field="candidates"
            )
        
        # Check for duplicate chunk_ids
        if not self.allow_duplicates:
            chunk_ids = []
            for i, candidate in enumerate(candidates):
                if not hasattr(candidate, 'chunk_id'):
                    raise ValidationError(
                        f"Candidate at index {i} missing chunk_id attribute",
                        field="candidates"
                    )
                
                chunk_id = candidate.chunk_id
                if chunk_id in chunk_ids:
                    raise ValidationError(
                        f"Duplicate chunk_id found: {chunk_id}",
                        field="candidates",
                        value=chunk_id
                    )
                chunk_ids.append(chunk_id)
        
        # Validate each candidate has required fields
        for i, candidate in enumerate(candidates):
            self._validate_candidate(candidate, i)
        
        logger.debug(f"Candidates validation passed: {len(candidates)} candidates")
    
    def _validate_candidate(self, candidate: Any, index: int) -> None:
        """
        Validate a single candidate.
        
        This method ensures that a candidate has all required fields and
        valid metadata.
        
        Args:
            candidate: Candidate to validate
            index: Index of candidate in the list
            
        Raises:
            ValidationError: If candidate is invalid
        """
        required_fields = ['chunk_id', 'candidate_name', 'resume_id', 'section']
        
        for field in required_fields:
            if not hasattr(candidate, field):
                raise ValidationError(
                    f"Candidate at index {index} missing required field: {field}",
                    field="candidate",
                    value=candidate
                )
            
            value = getattr(candidate, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValidationError(
                    f"Candidate at index {index} has empty field: {field}",
                    field="candidate",
                    value=candidate
                )
        
        # Validate metadata if present
        if hasattr(candidate, 'metadata'):
            if not isinstance(candidate.metadata, dict):
                raise ValidationError(
                    f"Candidate at index {index} metadata must be a dictionary",
                    field="candidate.metadata",
                    value=candidate.metadata
                )
    
    def validate_score(self, score: float) -> None:
        """
        Validate a rerank score.
        
        This method ensures that the score is a valid number (not NaN or
        infinite) and within the expected range.
        
        Args:
            score: The score to validate
            
        Raises:
            ValidationError: If score is invalid
        """
        if not isinstance(score, (int, float)):
            raise ValidationError(
                f"Score must be a number, got {type(score).__name__}",
                field="score",
                value=score
            )
        
        # Check for NaN
        if math.isnan(score):
            raise ValidationError(
                "Score cannot be NaN",
                field="score",
                value=score
            )
        
        # Check for infinity
        if math.isinf(score):
            raise ValidationError(
                "Score cannot be infinite",
                field="score",
                value=score
            )
        
        # Check score range
        if score < self.min_score or score > self.max_score:
            raise ValidationError(
                f"Score must be between {self.min_score} and {self.max_score}, got {score}",
                field="score",
                value=score
            )
        
        logger.debug(f"Score validation passed: {score:.4f}")
    
    def validate_scores(self, scores: List[float]) -> None:
        """
        Validate a list of rerank scores.
        
        This method validates all scores in a list, ensuring they are all
        valid numbers and within the expected range.
        
        Args:
            scores: List of scores to validate
            
        Raises:
            ValidationError: If any score is invalid
        """
        if not isinstance(scores, list):
            raise ValidationError(
                f"Scores must be a list, got {type(scores).__name__}",
                field="scores"
            )
        
        if not scores:
            raise ValidationError(
                "Scores list cannot be empty",
                field="scores"
            )
        
        for i, score in enumerate(scores):
            try:
                self.validate_score(score)
            except ValidationError as e:
                raise ValidationError(
                    f"Score at index {i} invalid: {e.message}",
                    field=f"scores[{i}]",
                    value=score
                )
        
        logger.debug(f"Scores validation passed: {len(scores)} scores")
    
    def validate_model_output(self, output: Any) -> None:
        """
        Validate model output.
        
        This method ensures that the model output is properly formatted
        and contains valid scores.
        
        Args:
            output: Model output to validate
            
        Raises:
            ValidationError: If model output is invalid
        """
        if output is None:
            raise ValidationError(
                "Model output cannot be None",
                field="model_output"
            )
        
        # If output is a list of scores
        if isinstance(output, list):
            self.validate_scores(output)
        
        # If output is a numpy array
        elif hasattr(output, 'shape'):
            import numpy as np
            if isinstance(output, np.ndarray):
                # Convert to list and validate
                scores = output.tolist()
                self.validate_scores(scores)
        
        # If output is a single score
        elif isinstance(output, (int, float)):
            self.validate_score(output)
        
        else:
            raise ValidationError(
                f"Model output must be a list, array, or number, got {type(output).__name__}",
                field="model_output",
                value=output
            )
        
        logger.debug("Model output validation passed")
    
    def validate_reranked_results(self, results: List[Any]) -> None:
        """
        Validate reranked results.
        
        This method ensures that reranked results are properly formatted
        and contain valid data.
        
        Args:
            results: List of reranked results to validate
            
        Raises:
            ValidationError: If results are invalid
        """
        if not isinstance(results, list):
            raise ValidationError(
                f"Results must be a list, got {type(results).__name__}",
                field="results"
            )
        
        if not results:
            raise ValidationError(
                "Results list cannot be empty",
                field="results"
            )
        
        # Validate each result
        for i, result in enumerate(results):
            if not hasattr(result, 'rerank_score'):
                raise ValidationError(
                    f"Result at index {i} missing rerank_score",
                    field="results"
                )
            
            if not hasattr(result, 'final_rank'):
                raise ValidationError(
                    f"Result at index {i} missing final_rank",
                    field="results"
                )
            
            # Validate rerank score
            try:
                self.validate_score(result.rerank_score)
            except ValidationError as e:
                raise ValidationError(
                    f"Result at index {i} has invalid rerank_score: {e.message}",
                    field=f"results[{i}].rerank_score",
                    value=result.rerank_score
                )
        
        # Check that ranks are sequential
        ranks = [result.final_rank for result in results]
        expected_ranks = list(range(len(results)))
        if ranks != expected_ranks:
            raise ValidationError(
                f"Ranks must be sequential from 0 to {len(results)-1}, got {ranks}",
                field="results"
            )
        
        logger.debug(f"Reranked results validation passed: {len(results)} results")
    
    def validate_batch_size(self, batch_size: int, num_candidates: int) -> None:
        """
        Validate batch size.
        
        This method ensures that the batch size is valid for the number
        of candidates.
        
        Args:
            batch_size: Batch size to validate
            num_candidates: Number of candidates
            
        Raises:
            ValidationError: If batch size is invalid
        """
        if not isinstance(batch_size, int):
            raise ValidationError(
                f"Batch size must be an integer, got {type(batch_size).__name__}",
                field="batch_size",
                value=batch_size
            )
        
        if batch_size <= 0:
            raise ValidationError(
                f"Batch size must be positive, got {batch_size}",
                field="batch_size",
                value=batch_size
            )
        
        if batch_size > num_candidates:
            logger.warning(
                f"Batch size ({batch_size}) exceeds number of candidates ({num_candidates}), "
                "will use {num_candidates} instead"
            )
        
        logger.debug(f"Batch size validation passed: {batch_size}")
