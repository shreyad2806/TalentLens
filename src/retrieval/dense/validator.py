"""
Validator for Dense Retrieval Service.

This module provides validation logic for retrieval operations, ensuring
data integrity and correctness throughout the retrieval pipeline.

Architecture Notes:
- Single Responsibility: Handles only validation logic
- Dependency Inversion: Depends on abstractions (schemas)
- Open/Closed: Open for new validation rules
- Clear error messages for debugging

SOLID Principles Applied:
- Single Responsibility: Validates retrieval data
- Open/Closed: Extensible for new validation rules
- Liskov Substitution: Can be swapped with other validators
- Interface Segregation: Focused validation interface
"""

import logging
from typing import List, Dict, Any, Optional
from .schema import DenseSearchResult, AggregatedCandidateResult

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that caused the error (optional)
        """
        self.message = message
        self.field = field
        super().__init__(self.message)


class RetrievalValidator:
    """
    Validator for dense retrieval operations.
    
    This class provides comprehensive validation for retrieval operations,
    including query validation, vector validation, candidate validation,
    and metadata validation.
    
    Architecture Pattern: Validator Pattern
    - Validates input data before processing
    - Ensures data integrity
    - Provides clear error messages
    - Prevents invalid data from propagating
    
    Validation Rules:
    - Query: Non-empty, reasonable length
    - Vectors: Correct dimension, valid values
    - Candidates: No duplicates, required fields
    - Metadata: Required fields present
    - Scores: Valid range (0.0 - 1.0)
    """
    
    def __init__(self, vector_dimension: int = 1024):
        """
        Initialize the validator.
        
        Args:
            vector_dimension: Expected dimension of vectors (default: 1024)
        """
        self.vector_dimension = vector_dimension
        logger.info(f"RetrievalValidator initialized with vector dimension: {vector_dimension}")
    
    def validate_query(self, query: str) -> None:
        """
        Validate search query.
        
        Validates that the query is non-empty and has reasonable length.
        
        Args:
            query: Search query to validate
            
        Raises:
            ValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty", field="query")
        
        if len(query.strip()) < 2:
            raise ValidationError("Query must be at least 2 characters", field="query")
        
        if len(query) > 1000:
            raise ValidationError("Query cannot exceed 1000 characters", field="query")
        
        logger.debug(f"Query validation passed: {query[:50]}...")
    
    def validate_vector(self, vector: List[float]) -> None:
        """
        Validate vector.
        
        Validates that the vector has correct dimension and valid values.
        
        Args:
            vector: Vector to validate
            
        Raises:
            ValidationError: If vector is invalid
        """
        if not vector:
            raise ValidationError("Vector cannot be empty", field="vector")
        
        if len(vector) != self.vector_dimension:
            raise ValidationError(
                f"Vector dimension mismatch: expected {self.vector_dimension}, got {len(vector)}",
                field="vector"
            )
        
        # Validate vector values are valid floats
        for i, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"Vector contains non-numeric value at index {i}: {value}",
                    field="vector"
                )
            if not -1.0 <= value <= 1.0:
                raise ValidationError(
                    f"Vector value at index {i} outside valid range [-1.0, 1.0]: {value}",
                    field="vector"
                )
        
        logger.debug(f"Vector validation passed: dimension={len(vector)}")
    
    def validate_search_results(self, results: List[DenseSearchResult]) -> None:
        """
        Validate search results.
        
        Validates that search results have no duplicates, valid scores,
        and required metadata.
        
        Args:
            results: List of search results to validate
            
        Raises:
            ValidationError: If results are invalid
        """
        if not results:
            logger.debug("No results to validate")
            return
        
        # Check for duplicate candidates
        candidate_ids = set()
        for result in results:
            candidate_key = f"{result.resume_id}_{result.chunk_id}"
            if candidate_key in candidate_ids:
                raise ValidationError(
                    f"Duplicate candidate found: {result.resume_id} - {result.chunk_id}",
                    field="results"
                )
            candidate_ids.add(candidate_key)
        
        # Validate each result
        for i, result in enumerate(results):
            self._validate_search_result(result, i)
        
        logger.debug(f"Search results validation passed: {len(results)} results")
    
    def _validate_search_result(self, result: DenseSearchResult, index: int) -> None:
        """
        Validate a single search result.
        
        Args:
            result: Search result to validate
            index: Index of the result in the list
            
        Raises:
            ValidationError: If result is invalid
        """
        # Validate required fields
        if not result.query:
            raise ValidationError(f"Result {index}: Query cannot be empty", field="query")
        
        if not result.candidate_name:
            raise ValidationError(f"Result {index}: Candidate name cannot be empty", field="candidate_name")
        
        if not result.resume_id:
            raise ValidationError(f"Result {index}: Resume ID cannot be empty", field="resume_id")
        
        if not result.chunk_id:
            raise ValidationError(f"Result {index}: Chunk ID cannot be empty", field="chunk_id")
        
        if not result.section:
            raise ValidationError(f"Result {index}: Section cannot be empty", field="section")
        
        if not result.matched_text:
            raise ValidationError(f"Result {index}: Matched text cannot be empty", field="matched_text")
        
        # Validate scores
        if not 0.0 <= result.score <= 1.0:
            raise ValidationError(
                f"Result {index}: Score must be between 0.0 and 1.0, got {result.score}",
                field="score"
            )
        
        if not 0.0 <= result.normalized_score <= 1.0:
            raise ValidationError(
                f"Result {index}: Normalized score must be between 0.0 and 1.0, got {result.normalized_score}",
                field="normalized_score"
            )
        
        # Validate rank
        if result.rank < 0:
            raise ValidationError(
                f"Result {index}: Rank must be non-negative, got {result.rank}",
                field="rank"
            )
    
    def validate_aggregated_candidates(self, candidates: List[AggregatedCandidateResult]) -> None:
        """
        Validate aggregated candidate results.
        
        Validates that aggregated candidates have no duplicates and valid scores.
        
        Args:
            candidates: List of aggregated candidates to validate
            
        Raises:
            ValidationError: If candidates are invalid
        """
        if not candidates:
            logger.debug("No candidates to validate")
            return
        
        # Check for duplicate resume IDs
        resume_ids = set()
        for candidate in candidates:
            if candidate.resume_id in resume_ids:
                raise ValidationError(
                    f"Duplicate resume ID found: {candidate.resume_id}",
                    field="candidates"
                )
            resume_ids.add(candidate.resume_id)
        
        # Validate each candidate
        for i, candidate in enumerate(candidates):
            self._validate_aggregated_candidate(candidate, i)
        
        logger.debug(f"Aggregated candidates validation passed: {len(candidates)} candidates")
    
    def _validate_aggregated_candidate(self, candidate: AggregatedCandidateResult, index: int) -> None:
        """
        Validate a single aggregated candidate.
        
        Args:
            candidate: Aggregated candidate to validate
            index: Index of the candidate in the list
            
        Raises:
            ValidationError: If candidate is invalid
        """
        # Validate required fields
        if not candidate.candidate_name:
            raise ValidationError(
                f"Candidate {index}: Candidate name cannot be empty",
                field="candidate_name"
            )
        
        if not candidate.resume_id:
            raise ValidationError(
                f"Candidate {index}: Resume ID cannot be empty",
                field="resume_id"
            )
        
        # Validate final score
        if not 0.0 <= candidate.final_score <= 1.0:
            raise ValidationError(
                f"Candidate {index}: Final score must be between 0.0 and 1.0, got {candidate.final_score}",
                field="final_score"
            )
        
        # Validate section scores
        for section, score in candidate.section_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValidationError(
                    f"Candidate {index}: Section score for '{section}' must be between 0.0 and 1.0, got {score}",
                    field="section_scores"
                )
    
    def validate_filters(self, filters: Optional[Dict[str, Any]]) -> None:
        """
        Validate search filters.
        
        Validates that filters are properly structured and contain valid values.
        
        Args:
            filters: Filters dictionary to validate
            
        Raises:
            ValidationError: If filters are invalid
        """
        if filters is None:
            return
        
        if not isinstance(filters, dict):
            raise ValidationError("Filters must be a dictionary", field="filters")
        
        # Validate filter keys and values
        valid_filter_keys = {
            "resume_id", "candidate_name", "section", "experience", "location", "role", "education"
        }
        
        for key, value in filters.items():
            if key not in valid_filter_keys:
                logger.warning(f"Unknown filter key: {key}")
            
            if value is None:
                raise ValidationError(f"Filter value for '{key}' cannot be None", field="filters")
            
            if isinstance(value, str) and not value.strip():
                raise ValidationError(f"Filter value for '{key}' cannot be empty string", field="filters")
        
        logger.debug(f"Filters validation passed: {len(filters)} filters")
    
    def validate_top_k(self, top_k: int) -> None:
        """
        Validate top_k parameter.
        
        Validates that top_k is within valid range.
        
        Args:
            top_k: Number of results to return
            
        Raises:
            ValidationError: If top_k is invalid
        """
        if not isinstance(top_k, int):
            raise ValidationError(f"top_k must be an integer, got {type(top_k).__name__}", field="top_k")
        
        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1, got {top_k}", field="top_k")
        
        if top_k > 1000:
            raise ValidationError(f"top_k cannot exceed 1000, got {top_k}", field="top_k")
        
        logger.debug(f"top_k validation passed: {top_k}")
