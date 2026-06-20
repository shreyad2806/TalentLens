"""
Validator module - Validation for vector records and operations.

This module provides validation logic for vector records to ensure data integrity
before storage operations. It validates vectors, dimensions, IDs, and metadata
to prevent invalid data from being stored.

Architecture Notes:
- Validator follows the Single Responsibility Principle
- Separates validation logic from business logic
- Provides clear error messages for debugging
- Rejects invalid records before they reach the storage layer

SOLID Principles Applied:
- Single Responsibility: Handles only validation logic
- Open/Closed: Can be extended with new validation rules without modification
"""

import math
from typing import List, Dict, Any, Optional, Set
from .schema import VectorRecord


class ValidationError(Exception):
    """
    Exception raised when validation fails.
    
    This exception provides detailed information about validation failures,
    including the field that failed and the reason for failure.
    
    Architecture Pattern: Custom Exception
    - Provides specific error information
    - Allows application to handle validation errors specifically
    - Can be extended with field-specific exceptions
    """
    
    def __init__(self, field: str, message: str, record_id: Optional[str] = None):
        """
        Initialize the validation error.
        
        Args:
            field: The field that failed validation
            message: Description of the validation failure
            record_id: Optional record ID for context
        """
        self.field = field
        self.message = message
        self.record_id = record_id
        if record_id:
            super().__init__(f"Validation error for record '{record_id}': {field} - {message}")
        else:
            super().__init__(f"Validation error: {field} - {message}")


class VectorStoreValidator:
    """
    Validator for vector records and operations.
    
    This class provides comprehensive validation for vector records before
    they are stored in the vector store. It validates:
    - Empty vectors
    - Dimension mismatch
    - Duplicate IDs
    - Invalid metadata
    - NaN values
    
    Architecture Pattern: Strategy Pattern
    - Different validation strategies can be composed
    - Validation rules are independent and composable
    - Easy to add new validation rules
    
    SOLID Principles Applied:
    - Single Responsibility: Handles only validation
    - Open/Closed: Open for extension with new validators
    - Dependency Inversion: Depends on abstractions (VectorRecord schema)
    """
    
    def __init__(self, expected_dimension: Optional[int] = None):
        """
        Initialize the validator.
        
        Args:
            expected_dimension: Expected vector dimension. If None, dimension
                              validation is skipped (useful for heterogeneous vectors)
        """
        self.expected_dimension = expected_dimension
    
    def validate_record(self, record: VectorRecord) -> None:
        """
        Validate a single vector record.
        
        Performs all validation checks on a single record and raises
        ValidationError if any validation fails.
        
        Args:
            record: VectorRecord to validate
            
        Raises:
            ValidationError: If validation fails
        """
        self._validate_id(record.id)
        self._validate_vector(record.vector)
        self._validate_dimension(record.vector)
        self._validate_metadata(record.metadata)
    
    def validate_records(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Validate a batch of vector records.
        
        Performs validation on multiple records and returns a summary
        of validation results. Also checks for duplicate IDs across records.
        
        Args:
            records: List of VectorRecord objects to validate
            
        Returns:
            Dictionary with validation results:
            - valid: bool (True if all records are valid)
            - valid_count: int (number of valid records)
            - invalid_count: int (number of invalid records)
            - errors: List of ValidationError objects
            - duplicate_ids: Set of duplicate IDs found
        """
        errors = []
        valid_count = 0
        invalid_count = 0
        
        # Check for duplicate IDs
        ids = [record.id for record in records]
        duplicate_ids = self._find_duplicate_ids(ids)
        
        # Validate each record
        for record in records:
            try:
                self.validate_record(record)
                valid_count += 1
            except ValidationError as e:
                errors.append(e)
                invalid_count += 1
        
        # Add errors for duplicate IDs
        for duplicate_id in duplicate_ids:
            errors.append(ValidationError(
                field='id',
                message=f"Duplicate ID found in batch",
                record_id=duplicate_id
            ))
        
        return {
            'valid': len(errors) == 0,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'errors': errors,
            'duplicate_ids': duplicate_ids
        }
    
    def _validate_id(self, id: str) -> None:
        """
        Validate that an ID is not empty.
        
        Args:
            id: ID to validate
            
        Raises:
            ValidationError: If ID is empty
        """
        if not id or not id.strip():
            raise ValidationError(field='id', message='ID cannot be empty')
    
    def _validate_vector(self, vector: List[float]) -> None:
        """
        Validate that a vector is not empty and contains no NaN values.
        
        Args:
            vector: Vector to validate
            
        Raises:
            ValidationError: If vector is empty or contains NaN
        """
        if not vector or len(vector) == 0:
            raise ValidationError(field='vector', message='Vector cannot be empty')
        
        if any(math.isnan(x) for x in vector):
            raise ValidationError(field='vector', message='Vector contains NaN values')
    
    def _validate_dimension(self, vector: List[float]) -> None:
        """
        Validate that a vector has the expected dimension.
        
        Args:
            vector: Vector to validate
            
        Raises:
            ValidationError: If dimension doesn't match expected
        """
        if self.expected_dimension is not None:
            if len(vector) != self.expected_dimension:
                raise ValidationError(
                    field='vector',
                    message=f'Dimension mismatch: expected {self.expected_dimension}, got {len(vector)}'
                )
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate that metadata is a valid dictionary.
        
        Args:
            metadata: Metadata to validate
            
        Raises:
            ValidationError: If metadata is invalid
        """
        if metadata is None:
            raise ValidationError(field='metadata', message='Metadata cannot be None')
        
        if not isinstance(metadata, dict):
            raise ValidationError(
                field='metadata',
                message=f'Metadata must be a dictionary, got {type(metadata).__name__}'
            )
    
    def _find_duplicate_ids(self, ids: List[str]) -> Set[str]:
        """
        Find duplicate IDs in a list.
        
        Args:
            ids: List of IDs to check
            
        Returns:
            Set of duplicate IDs
        """
        seen = set()
        duplicates = set()
        
        for id in ids:
            if id in seen:
                duplicates.add(id)
            seen.add(id)
        
        return duplicates
    
    def validate_dimension_consistency(self, records: List[VectorRecord]) -> Dict[str, Any]:
        """
        Validate that all records have consistent vector dimensions.
        
        This is useful for ensuring that all vectors in a batch have the
        same dimension, which is required by most vector databases.
        
        Args:
            records: List of VectorRecord objects to validate
            
        Returns:
            Dictionary with validation results:
            - consistent: bool (True if all dimensions are consistent)
            - dimensions: Set of unique dimensions found
            - most_common_dimension: int (most common dimension)
            - inconsistent_count: int (number of records with inconsistent dimensions)
        """
        dimensions = set()
        dimension_counts = {}
        
        for record in records:
            dim = len(record.vector)
            dimensions.add(dim)
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        most_common_dimension = max(dimension_counts.items(), key=lambda x: x[1])[0] if dimension_counts else 0
        inconsistent_count = sum(count for dim, count in dimension_counts.items() if dim != most_common_dimension)
        
        return {
            'consistent': len(dimensions) == 1,
            'dimensions': dimensions,
            'most_common_dimension': most_common_dimension,
            'inconsistent_count': inconsistent_count
        }
    
    def validate_query_vector(self, vector: List[float]) -> None:
        """
        Validate a query vector before performing a search.
        
        Args:
            vector: Query vector to validate
            
        Raises:
            ValidationError: If vector is invalid
        """
        self._validate_vector(vector)
        
        if self.expected_dimension is not None:
            self._validate_dimension(vector)
