"""
Tests for Schema Alignment Layer.

This module contains tests for the SchemaAlignment and FilterKeyNormalizer classes,
which handle mapping between metadata filter schema and retrieval filter schema.
"""

import pytest
from typing import Dict, Any

from src.retrieval.metadata.schema import MetadataFilter
from src.retrieval.metadata.schema_alignment import (
    SchemaAlignment,
    SchemaAlignmentError,
    FilterKeyNormalizer,
)


class TestSchemaAlignment:
    """Tests for SchemaAlignment class."""
    
    def test_align_metadata_to_retrieval_empty_filter(self):
        """Test alignment with empty metadata filter."""
        metadata_filter = MetadataFilter()
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert retrieval_filters == {}
    
    def test_align_minimum_experience(self):
        """Test alignment of minimum_experience to experience range."""
        metadata_filter = MetadataFilter(minimum_experience=5.0)
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "experience" in retrieval_filters
        assert retrieval_filters["experience"]["min"] == 5.0
        assert retrieval_filters["experience"]["max"] is None
    
    def test_align_maximum_experience(self):
        """Test alignment of maximum_experience to experience range."""
        metadata_filter = MetadataFilter(maximum_experience=10.0)
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "experience" in retrieval_filters
        assert retrieval_filters["experience"]["min"] is None
        assert retrieval_filters["experience"]["max"] == 10.0
    
    def test_align_experience_range(self):
        """Test alignment of both minimum and maximum experience."""
        metadata_filter = MetadataFilter(minimum_experience=3.0, maximum_experience=7.0)
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "experience" in retrieval_filters
        assert retrieval_filters["experience"]["min"] == 3.0
        assert retrieval_filters["experience"]["max"] == 7.0
    
    def test_align_location(self):
        """Test alignment of location field."""
        metadata_filter = MetadataFilter(location="Bangalore")
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "location" in retrieval_filters
        assert retrieval_filters["location"] == "Bangalore"
    
    def test_align_preferred_locations(self):
        """Test alignment of preferred_locations to location."""
        metadata_filter = MetadataFilter(preferred_locations=["Bangalore", "Mumbai"])
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "location" in retrieval_filters
        assert retrieval_filters["location"] == "Bangalore"
    
    def test_align_skills(self):
        """Test alignment of skills field."""
        metadata_filter = MetadataFilter(skills=["Python", "Java"])
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "skills" in retrieval_filters
        assert retrieval_filters["skills"] == ["Python", "Java"]
    
    def test_align_education(self):
        """Test alignment of education field."""
        metadata_filter = MetadataFilter(education=["Computer Science", "Engineering"])
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "education" in retrieval_filters
        assert retrieval_filters["education"] == ["Computer Science", "Engineering"]
    
    def test_align_degree_to_role(self):
        """Test alignment of degree to role field."""
        metadata_filter = MetadataFilter(degree="Bachelor of Computer Science")
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "role" in retrieval_filters
        assert retrieval_filters["role"] == "Bachelor of Computer Science"
    
    def test_align_current_company_to_role(self):
        """Test alignment of current_company to role field."""
        metadata_filter = MetadataFilter(current_company="Google")
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "role" in retrieval_filters
        assert retrieval_filters["role"] == "Google"
    
    def test_align_employment_type_to_role(self):
        """Test alignment of employment_type to role field."""
        metadata_filter = MetadataFilter(employment_type="full-time")
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "role" in retrieval_filters
        assert retrieval_filters["role"] == "full-time"
    
    def test_align_degree_takes_precedence_over_company(self):
        """Test that degree takes precedence over current_company for role mapping."""
        metadata_filter = MetadataFilter(
            degree="Master's Degree",
            current_company="Microsoft"
        )
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "role" in retrieval_filters
        assert retrieval_filters["role"] == "Master's Degree"
    
    def test_align_comprehensive_filter(self):
        """Test alignment with comprehensive metadata filter."""
        metadata_filter = MetadataFilter(
            minimum_experience=5.0,
            maximum_experience=10.0,
            location="Bangalore",
            skills=["Python", "Django"],
            education=["Computer Science"],
            degree="Bachelor's"
        )
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        assert "experience" in retrieval_filters
        assert retrieval_filters["experience"]["min"] == 5.0
        assert retrieval_filters["experience"]["max"] == 10.0
        assert retrieval_filters["location"] == "Bangalore"
        assert retrieval_filters["skills"] == ["Python", "Django"]
        assert retrieval_filters["education"] == ["Computer Science"]
        assert retrieval_filters["role"] == "Bachelor's"
    
    def test_validate_retrieval_filters_none(self):
        """Test validation with None filters."""
        SchemaAlignment.validate_retrieval_filters(None)  # Should not raise
    
    def test_validate_retrieval_filters_valid(self):
        """Test validation with valid filters."""
        filters = {
            "experience": {"min": 5, "max": 10},
            "location": "Bangalore",
            "skills": ["Python"]
        }
        SchemaAlignment.validate_retrieval_filters(filters)  # Should not raise
    
    def test_validate_retrieval_filters_invalid_type(self):
        """Test validation with invalid filter type."""
        filters = "not a dict"
        
        with pytest.raises(SchemaAlignmentError) as exc_info:
            SchemaAlignment.validate_retrieval_filters(filters)
        
        assert "must be a dictionary" in str(exc_info.value)
    
    def test_validate_retrieval_filters_none_value(self):
        """Test validation with None value."""
        filters = {"location": None}
        
        with pytest.raises(SchemaAlignmentError) as exc_info:
            SchemaAlignment.validate_retrieval_filters(filters)
        
        assert "cannot be None" in str(exc_info.value)
    
    def test_validate_retrieval_filters_empty_string(self):
        """Test validation with empty string value."""
        filters = {"location": ""}
        
        with pytest.raises(SchemaAlignmentError) as exc_info:
            SchemaAlignment.validate_retrieval_filters(filters)
        
        assert "cannot be empty string" in str(exc_info.value)
    
    def test_validate_retrieval_filters_unknown_key_no_warning(self):
        """Test that unknown keys don't raise errors (extensibility)."""
        filters = {
            "location": "Bangalore",
            "custom_field": "custom_value"  # Unknown key
        }
        SchemaAlignment.validate_retrieval_filters(filters)  # Should not raise
    
    def test_get_valid_retrieval_filter_keys(self):
        """Test getting valid retrieval filter keys."""
        valid_keys = SchemaAlignment.get_valid_retrieval_filter_keys()
        
        assert "experience" in valid_keys
        assert "location" in valid_keys
        assert "role" in valid_keys
        assert "education" in valid_keys
        assert "skills" in valid_keys
    
    def test_add_custom_retrieval_filter_key(self):
        """Test adding a custom filter key."""
        initial_keys = SchemaAlignment.get_valid_retrieval_filter_keys()
        initial_count = len(initial_keys)
        
        SchemaAlignment.add_custom_retrieval_filter_key("custom_field")
        
        updated_keys = SchemaAlignment.get_valid_retrieval_filter_keys()
        assert len(updated_keys) == initial_count + 1
        assert "custom_field" in updated_keys
    
    def test_align_and_validate(self):
        """Test the convenience method that combines alignment and validation."""
        metadata_filter = MetadataFilter(
            minimum_experience=5.0,
            location="Bangalore",
            skills=["Python"]
        )
        
        retrieval_filters = SchemaAlignment.align_and_validate(metadata_filter)
        
        assert "experience" in retrieval_filters
        assert retrieval_filters["experience"]["min"] == 5.0
        assert retrieval_filters["location"] == "Bangalore"
        assert retrieval_filters["skills"] == ["Python"]


class TestFilterKeyNormalizer:
    """Tests for FilterKeyNormalizer class."""
    
    def test_normalize_key_min_experience(self):
        """Test normalization of min_experience variations."""
        assert FilterKeyNormalizer.normalize_key("min_experience") == "minimum_experience"
        assert FilterKeyNormalizer.normalize_key("min_exp") == "minimum_experience"
        assert FilterKeyNormalizer.normalize_key("exp_min") == "minimum_experience"
    
    def test_normalize_key_max_experience(self):
        """Test normalization of max_experience variations."""
        assert FilterKeyNormalizer.normalize_key("max_experience") == "maximum_experience"
        assert FilterKeyNormalizer.normalize_key("max_exp") == "maximum_experience"
        assert FilterKeyNormalizer.normalize_key("exp_max") == "maximum_experience"
    
    def test_normalize_key_experience(self):
        """Test normalization of experience variations."""
        assert FilterKeyNormalizer.normalize_key("exp") == "experience"
        assert FilterKeyNormalizer.normalize_key("years_of_experience") == "experience"
        assert FilterKeyNormalizer.normalize_key("yoe") == "experience"
    
    def test_normalize_key_location(self):
        """Test normalization of location variations."""
        assert FilterKeyNormalizer.normalize_key("loc") == "location"
    
    def test_normalize_key_preferred_locations(self):
        """Test normalization of preferred_locations variations."""
        assert FilterKeyNormalizer.normalize_key("preferred_loc") == "preferred_locations"
    
    def test_normalize_key_skills(self):
        """Test normalization of skills variations."""
        assert FilterKeyNormalizer.normalize_key("skill") == "skills"
        assert FilterKeyNormalizer.normalize_key("tech_stack") == "skills"
        assert FilterKeyNormalizer.normalize_key("tech") == "skills"
    
    def test_normalize_key_education(self):
        """Test normalization of education variations."""
        assert FilterKeyNormalizer.normalize_key("edu") == "education"
        assert FilterKeyNormalizer.normalize_key("qualifications") == "education"
    
    def test_normalize_key_degree(self):
        """Test normalization of degree variations."""
        assert FilterKeyNormalizer.normalize_key("deg") == "degree"
    
    def test_normalize_key_unknown(self):
        """Test normalization of unknown key (returns original)."""
        assert FilterKeyNormalizer.normalize_key("unknown_key") == "unknown_key"
    
    def test_normalize_key_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert FilterKeyNormalizer.normalize_key("MIN_EXPERIENCE") == "minimum_experience"
        assert FilterKeyNormalizer.normalize_key("Min_Experience") == "minimum_experience"
    
    def test_normalize_filter_dict(self):
        """Test normalization of entire filter dictionary."""
        filters = {
            "min_exp": 5,
            "max_exp": 10,
            "loc": "Bangalore",
            "tech_stack": ["Python", "Java"]
        }
        
        normalized = FilterKeyNormalizer.normalize_filter_dict(filters)
        
        assert normalized["minimum_experience"] == 5
        assert normalized["maximum_experience"] == 10
        assert normalized["location"] == "Bangalore"
        assert normalized["skills"] == ["Python", "Java"]
    
    def test_normalize_filter_dict_preserves_unknown(self):
        """Test that unknown keys are preserved in dictionary normalization."""
        filters = {
            "location": "Bangalore",
            "custom_field": "custom_value"
        }
        
        normalized = FilterKeyNormalizer.normalize_filter_dict(filters)
        
        assert normalized["location"] == "Bangalore"
        assert normalized["custom_field"] == "custom_value"


class TestIntegration:
    """Integration tests for schema alignment workflow."""
    
    def test_end_to_end_alignment_workflow(self):
        """Test complete workflow from metadata filter to retrieval filter."""
        # Create metadata filter with various fields
        metadata_filter = MetadataFilter(
            minimum_experience=3.0,
            maximum_experience=8.0,
            location="Bangalore",
            skills=["Python", "Django", "PostgreSQL"],
            education=["Computer Science", "Engineering"],
            degree="Bachelor's",
            current_company="TechCorp"
        )
        
        # Align to retrieval schema
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        # Validate retrieval filters
        SchemaAlignment.validate_retrieval_filters(retrieval_filters)
        
        # Verify alignment
        assert retrieval_filters["experience"]["min"] == 3.0
        assert retrieval_filters["experience"]["max"] == 8.0
        assert retrieval_filters["location"] == "Bangalore"
        assert retrieval_filters["skills"] == ["Python", "Django", "PostgreSQL"]
        assert retrieval_filters["education"] == ["Computer Science", "Engineering"]
        assert retrieval_filters["role"] == "Bachelor's"  # Degree takes precedence
    
    def test_alignment_with_normalizer(self):
        """Test alignment combined with key normalization."""
        # Create filter with non-standard keys
        non_standard_filters = {
            "min_exp": 5,
            "max_exp": 10,
            "loc": "Bangalore",
            "tech": ["Python"]
        }
        
        # Normalize keys
        normalized_filters = FilterKeyNormalizer.normalize_filter_dict(non_standard_filters)
        
        # Create MetadataFilter from normalized values
        metadata_filter = MetadataFilter(
            minimum_experience=normalized_filters.get("minimum_experience"),
            maximum_experience=normalized_filters.get("maximum_experience"),
            location=normalized_filters.get("location"),
            skills=normalized_filters.get("skills")
        )
        
        # Align to retrieval schema
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        # Verify
        assert retrieval_filters["experience"]["min"] == 5
        assert retrieval_filters["experience"]["max"] == 10
        assert retrieval_filters["location"] == "Bangalore"
        assert retrieval_filters["skills"] == ["Python"]
    
    def test_alignment_preserves_all_valid_fields(self):
        """Test that alignment preserves all valid metadata fields."""
        metadata_filter = MetadataFilter(
            minimum_experience=2.0,
            maximum_experience=15.0,
            location="Remote",
            preferred_locations=["Bangalore", "Mumbai"],
            skills=["Python", "Java", "Go"],
            excluded_skills=["PHP"],
            education=["Computer Science"],
            degree="Master's",
            current_company="Startup",
            previous_company="BigCorp",
            salary_min=15.0,
            salary_max=25.0,
            notice_period=30,
            work_mode="remote",
            employment_type="full-time",
            certifications=["AWS", "GCP"],
            languages=["English", "Hindi"],
            availability="immediate"
        )
        
        retrieval_filters = SchemaAlignment.align_metadata_to_retrieval(metadata_filter)
        
        # Verify that key fields are aligned
        assert "experience" in retrieval_filters
        assert "location" in retrieval_filters
        assert "skills" in retrieval_filters
        assert "education" in retrieval_filters
        assert "role" in retrieval_filters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
