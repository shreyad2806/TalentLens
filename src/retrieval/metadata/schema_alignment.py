"""
Schema Alignment Layer for Metadata Filtering.

This module provides alignment between the MetadataFilter schema (used by the
metadata filtering engine) and the retrieval filter schema (used by dense/sparse
retrieval validators).

The metadata layer generates:
- minimum_experience
- maximum_experience
- skills

The retriever validates:
- experience
- location
- role
- education

This layer maps between these schemas to ensure compatibility.

SOLID Principles Applied:
- Single Responsibility: Handles only schema mapping
- Open/Closed: Open for new field mappings
- Dependency Inversion: Depends on schema abstractions
"""

import logging
from typing import Dict, Any, Optional, List

from .schema import MetadataFilter

logger = logging.getLogger(__name__)


class SchemaAlignmentError(Exception):
    """Custom exception for schema alignment errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class SchemaAlignment:
    """
    Aligns MetadataFilter schema with retrieval filter schema.
    
    This class provides methods to convert metadata filters into retrieval
    filters that are compatible with the retrieval validators.
    
    Mapping Rules:
        - minimum_experience + maximum_experience → experience (range)
        - skills → skills (direct pass-through)
        - location → location (direct pass-through)
        - education → education (direct pass-through)
        - degree → role (fallback mapping)
        - current_company → role (fallback mapping)
    """
    
    # Valid retrieval filter keys (from RetrievalValidator)
    VALID_RETRIEVAL_FILTER_KEYS = {
        "resume_id",
        "candidate_name", 
        "section",
        "experience",
        "location",
        "role",
        "education",
        "skills"  # Added for alignment
    }
    
    @staticmethod
    def align_metadata_to_retrieval(metadata_filter: MetadataFilter) -> Dict[str, Any]:
        """
        Align MetadataFilter to retrieval filter schema.
        
        This method converts metadata filter fields into the format expected
        by retrieval validators.
        
        Args:
            metadata_filter: MetadataFilter object from metadata filtering engine
            
        Returns:
            Dictionary of retrieval filters compatible with RetrievalValidator
            
        Raises:
            SchemaAlignmentError: If alignment fails
        """
        retrieval_filters: Dict[str, Any] = {}
        
        # Map experience range to single experience field
        if metadata_filter.minimum_experience is not None or metadata_filter.maximum_experience is not None:
            experience_range = {
                "min": metadata_filter.minimum_experience,
                "max": metadata_filter.maximum_experience
            }
            retrieval_filters["experience"] = experience_range
            logger.debug(f"Mapped experience range: {experience_range}")
        
        # Map location (direct pass-through)
        if metadata_filter.location:
            retrieval_filters["location"] = metadata_filter.location
            logger.debug(f"Mapped location: {metadata_filter.location}")
        
        # Map preferred_locations to location (OR logic)
        if metadata_filter.preferred_locations:
            # Use first preferred location as primary, or join with OR
            retrieval_filters["location"] = metadata_filter.preferred_locations[0]
            logger.debug(f"Mapped preferred_locations to location: {metadata_filter.preferred_locations[0]}")
        
        # Map skills (direct pass-through)
        if metadata_filter.skills:
            retrieval_filters["skills"] = metadata_filter.skills
            logger.debug(f"Mapped skills: {metadata_filter.skills}")
        
        # Map education (direct pass-through)
        if metadata_filter.education:
            retrieval_filters["education"] = metadata_filter.education
            logger.debug(f"Mapped education: {metadata_filter.education}")
        
        # Map degree to role (fallback mapping)
        if metadata_filter.degree:
            retrieval_filters["role"] = metadata_filter.degree
            logger.debug(f"Mapped degree to role: {metadata_filter.degree}")
        
        # Map current_company to role (fallback mapping if degree not set)
        if metadata_filter.current_company and "role" not in retrieval_filters:
            retrieval_filters["role"] = metadata_filter.current_company
            logger.debug(f"Mapped current_company to role: {metadata_filter.current_company}")
        
        # Map employment_type to role (fallback mapping)
        if metadata_filter.employment_type and "role" not in retrieval_filters:
            retrieval_filters["role"] = metadata_filter.employment_type
            logger.debug(f"Mapped employment_type to role: {metadata_filter.employment_type}")
        
        return retrieval_filters
    
    @staticmethod
    def validate_retrieval_filters(filters: Dict[str, Any]) -> None:
        """
        Validate retrieval filters against valid filter keys.
        
        This method checks that all filter keys are recognized by the
        retrieval validators. Unlike the validators, this does NOT emit
        warnings for unknown keys - it silently ignores them to avoid
        log pollution.
        
        Args:
            filters: Dictionary of retrieval filters
            
        Raises:
            SchemaAlignmentError: If filter values are invalid
        """
        if filters is None:
            return
        
        if not isinstance(filters, dict):
            raise SchemaAlignmentError("Filters must be a dictionary", field="filters")
        
        # Validate filter values (but not keys - we allow unknown keys)
        for key, value in filters.items():
            if value is None:
                raise SchemaAlignmentError(f"Filter value for '{key}' cannot be None", field="filters")
            
            if isinstance(value, str) and not value.strip():
                raise SchemaAlignmentError(f"Filter value for '{key}' cannot be empty string", field="filters")
        
        logger.debug(f"Retrieval filters validation passed: {len(filters)} filters")
    
    @staticmethod
    def get_valid_retrieval_filter_keys() -> set:
        """
        Get the set of valid retrieval filter keys.
        
        Returns:
            Set of valid filter keys recognized by retrieval validators
        """
        return SchemaAlignment.VALID_RETRIEVAL_FILTER_KEYS.copy()
    
    @staticmethod
    def add_custom_retrieval_filter_key(key: str) -> None:
        """
        Add a custom filter key to the valid retrieval filter keys.
        
        This allows extending the schema alignment for custom use cases.
        
        Args:
            key: New filter key to add
        """
        SchemaAlignment.VALID_RETRIEVAL_FILTER_KEYS.add(key)
        logger.info(f"Added custom retrieval filter key: {key}")
    
    @staticmethod
    def align_and_validate(metadata_filter: MetadataFilter) -> Dict[str, Any]:
        """
        Align metadata filter to retrieval schema and validate.
        
        This is a convenience method that combines alignment and validation.
        
        Args:
            metadata_filter: MetadataFilter object from metadata filtering engine
            
        Returns:
            Dictionary of validated retrieval filters
            
        Raises:
            SchemaAlignmentError: If alignment or validation fails
        """
        # Align schemas
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        # Validate
        SchemaAlignment.validate_retrieval_filters(retrieval_filters)
        
        return retrieval_filters


class FilterKeyNormalizer:
    """
    Normalizes filter keys to handle variations in naming conventions.
    
    This class provides methods to normalize filter keys to a standard format,
    helping to handle different naming conventions across the system.
    """
    
    # Key mapping for normalization
    KEY_MAPPINGS = {
        "min_experience": "minimum_experience",
        "max_experience": "maximum_experience",
        "min_exp": "minimum_experience",
        "max_exp": "maximum_experience",
        "exp_min": "minimum_experience",
        "exp_max": "maximum_experience",
        "exp": "experience",
        "years_of_experience": "experience",
        "yoe": "experience",
        "loc": "location",
        "preferred_loc": "preferred_locations",
        "preferred_loc": "preferred_locations",
        "skill": "skills",
        "tech_stack": "skills",
        "tech": "skills",
        "edu": "education",
        "qualifications": "education",
        "deg": "degree",
    }
    
    @staticmethod
    def normalize_key(key: str) -> str:
        """
        Normalize a filter key to standard format.
        
        Args:
            key: Filter key to normalize
            
        Returns:
            Normalized filter key
        """
        # Direct mapping
        if key in FilterKeyNormalizer.KEY_MAPPINGS:
            return FilterKeyNormalizer.KEY_MAPPINGS[key]
        
        # Case-insensitive check
        key_lower = key.lower()
        for mapped_key, standard_key in FilterKeyNormalizer.KEY_MAPPINGS.items():
            if mapped_key.lower() == key_lower:
                return standard_key
        
        # Return original if no mapping found
        return key
    
    @staticmethod
    def normalize_filter_dict(filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all keys in a filter dictionary.
        
        Args:
            filters: Dictionary of filters with potentially non-standard keys
            
        Returns:
            Dictionary with normalized keys
        """
        normalized = {}
        for key, value in filters.items():
            normalized_key = FilterKeyNormalizer.normalize_key(key)
            normalized[normalized_key] = value
        
        return normalized
