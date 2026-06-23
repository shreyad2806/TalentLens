"""
Validator for Sparse Retrieval Service.

This module provides validation functionality for BM25 index, queries, and results.
It ensures data integrity and proper configuration throughout the retrieval pipeline.

Architecture Notes:
- Comprehensive validation for all inputs and outputs
- Custom validation exceptions
- Detailed error messages
- Validation for index state

SOLID Principles Applied:
- Single Responsibility: Handles only validation
- Open/Closed: Open for new validation rules
- Dependency Inversion: Depends on validation interface
"""

import logging
from typing import List, Dict, Any, Optional, Set

from .schema import SparseSearchResult, BM25Document, BM25IndexStats
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class SparseRetrievalValidator:
    """
    Validator for sparse retrieval operations.
    
    This class provides validation methods for:
    - Index state (empty, corrupted)
    - Queries (empty, invalid tokens)
    - Documents (duplicate IDs, missing metadata)
    - Results (duplicate candidates, invalid scores)
    - Tokens (invalid characters, empty)
    
    Validation Rules:
        - Index must not be empty for search operations
        - Query must not be empty
        - Document IDs must be unique
        - Metadata must be present
        - Scores must be non-negative
        - Tokens must be valid
    """
    
    def __init__(self):
        """Initialize the validator."""
        logger.info("SparseRetrievalValidator initialized")
    
    def validate_index(self, index: BM25Index) -> None:
        """
        Validate that the index is in a valid state.
        
        Args:
            index: BM25Index to validate
            
        Raises:
            ValidationError: If index is invalid
        """
        if index is None:
            raise ValidationError("Index cannot be None", field="index")
        
        if index.is_empty():
            raise ValidationError("Index is empty", field="index")
        
        # Check for consistency
        stats = index.get_statistics()
        if stats.num_documents != len(index.document_store):
            raise ValidationError(
                f"Index inconsistency: num_documents={stats.num_documents} "
                f"but document_store has {len(index.document_store)} documents",
                field="index"
            )
        
        logger.debug("Index validation passed")
    
    def validate_query(self, query: str) -> None:
        """
        Validate a search query.
        
        Args:
            query: Query string to validate
            
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
    
    def validate_tokens(self, tokens: List[str]) -> None:
        """
        Validate a list of tokens.
        
        Args:
            tokens: List of tokens to validate
            
        Raises:
            ValidationError: If tokens are invalid
        """
        if not tokens:
            raise ValidationError("Tokens cannot be empty", field="tokens")
        
        for i, token in enumerate(tokens):
            if not token or not token.strip():
                raise ValidationError(f"Token at index {i} is empty", field="tokens")
            
            # Check for invalid characters (only allow alphanumeric and underscores)
            if not token.replace('_', '').replace('-', '').isalnum():
                logger.warning(f"Token '{token}' contains non-alphanumeric characters")
        
        logger.debug(f"Token validation passed: {len(tokens)} tokens")
    
    def validate_document(self, document: BM25Document) -> None:
        """
        Validate a BM25Document.
        
        Args:
            document: BM25Document to validate
            
        Raises:
            ValidationError: If document is invalid
        """
        if document is None:
            raise ValidationError("Document cannot be None", field="document")
        
        if not document.chunk_id:
            raise ValidationError("Document chunk_id cannot be empty", field="chunk_id")
        
        if not document.resume_id:
            raise ValidationError("Document resume_id cannot be empty", field="resume_id")
        
        if not document.section:
            raise ValidationError("Document section cannot be empty", field="section")
        
        if not document.candidate_name:
            raise ValidationError("Document candidate_name cannot be empty", field="candidate_name")
        
        if not document.text:
            raise ValidationError("Document text cannot be empty", field="text")
        
        if not document.tokens:
            raise ValidationError("Document tokens cannot be empty", field="tokens")
        
        if document.document_length <= 0:
            raise ValidationError(
                f"Document length must be positive, got {document.document_length}",
                field="document_length"
            )
        
        logger.debug(f"Document validation passed: {document.chunk_id}")
    
    def validate_documents(self, documents: List[BM25Document]) -> None:
        """
        Validate a list of BM25Documents.
        
        Args:
            documents: List of BM25Documents to validate
            
        Raises:
            ValidationError: If documents are invalid
        """
        if not documents:
            raise ValidationError("Documents list cannot be empty", field="documents")
        
        # Check for duplicate IDs
        doc_ids = set()
        for doc in documents:
            if doc.chunk_id in doc_ids:
                raise ValidationError(
                    f"Duplicate document ID found: {doc.chunk_id}",
                    field="documents"
                )
            doc_ids.add(doc.chunk_id)
        
        # Validate each document
        for i, document in enumerate(documents):
            try:
                self.validate_document(document)
            except ValidationError as e:
                raise ValidationError(
                    f"Document at index {i} failed validation: {e.message}",
                    field="documents"
                )
        
        logger.debug(f"Documents validation passed: {len(documents)} documents")
    
    def validate_search_results(self, results: List[SparseSearchResult]) -> None:
        """
        Validate search results.
        
        Args:
            results: List of SparseSearchResult to validate
            
        Raises:
            ValidationError: If results are invalid
        """
        if not results:
            logger.debug("No results to validate")
            return
        
        # Check for duplicate candidates
        candidate_keys = set()
        for result in results:
            candidate_key = f"{result.resume_id}_{result.chunk_id}"
            if candidate_key in candidate_keys:
                raise ValidationError(
                    f"Duplicate candidate found: {result.resume_id} - {result.chunk_id}",
                    field="results"
                )
            candidate_keys.add(candidate_key)
        
        # Validate each result
        for i, result in enumerate(results):
            self._validate_search_result(result, i)
        
        logger.debug(f"Search results validation passed: {len(results)} results")
    
    def _validate_search_result(self, result: SparseSearchResult, index: int) -> None:
        """
        Validate a single search result.
        
        Args:
            result: SparseSearchResult to validate
            index: Index of the result in the list
            
        Raises:
            ValidationError: If result is invalid
        """
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
        
        if result.bm25_score < 0:
            raise ValidationError(
                f"Result {index}: BM25 score must be non-negative, got {result.bm25_score}",
                field="bm25_score"
            )
        
        if result.rank < 0:
            raise ValidationError(
                f"Result {index}: Rank must be non-negative, got {result.rank}",
                field="rank"
            )
    
    def validate_filters(self, filters: Optional[Dict[str, Any]]) -> None:
        """
        Validate search filters.
        
        Note: Unknown filter keys are silently ignored to avoid log pollution.
        Use SchemaAlignment for proper schema mapping between metadata and retrieval layers.
        
        Args:
            filters: Dictionary of filters to validate
            
        Raises:
            ValidationError: If filters are invalid
        """
        if filters is None:
            return
        
        if not isinstance(filters, dict):
            raise ValidationError("Filters must be a dictionary", field="filters")
        
        # Validate filter values (keys are not validated to allow extensibility)
        for key, value in filters.items():
            if value is None:
                raise ValidationError(f"Filter value for '{key}' cannot be None", field="filters")
            
            if isinstance(value, str) and not value.strip():
                raise ValidationError(f"Filter value for '{key}' cannot be empty string", field="filters")
        
        logger.debug(f"Filters validation passed: {len(filters)} filters")
    
    def validate_top_k(self, top_k: int) -> None:
        """
        Validate top_k parameter.
        
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
