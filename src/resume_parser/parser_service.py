"""
Parser Service module - Unified orchestration layer for resume parsing.

This module provides the main ParserService class that orchestrates the entire
parsing pipeline: text extraction, section detection, and metadata extraction.
"""

from typing import Union, Optional
from pathlib import Path
from datetime import datetime

from .extractor import TextExtractor
from .section_parser import SectionParser
from .metadata_parser import MetadataParser
from .schema import ResumeDocument


class ParserService:
    """
    Unified parser service for resume documents.
    
    This class orchestrates the entire parsing pipeline:
    1. Extract text from file (PDF/DOCX/TXT)
    2. Detect semantic sections using headings
    3. Extract structured metadata from sections
    4. Return unified ResumeDocument object
    
    The service provides a clean, high-level interface for parsing resumes
    while keeping the underlying components modular and testable.
    """
    
    def __init__(self):
        """
        Initialize the parser service with component parsers.
        """
        self.text_extractor = TextExtractor()
        self.section_parser = SectionParser()
        self.metadata_parser = MetadataParser()
    
    def parse_file(self, file_path: Union[str, Path]) -> ResumeDocument:
        """
        Parse a resume from a file path.
        
        This is the main entry point for parsing resumes. It handles the entire
        pipeline from file I/O to structured data extraction.
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            
        Returns:
            ResumeDocument object containing all extracted information
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
            Exception: For parsing errors
        """
        # Step 1: Extract text from file
        raw_text = self.text_extractor.extract_from_file(file_path)
        
        # Step 2: Parse the extracted text
        document = self.parse_text(raw_text)
        
        return document
    
    def parse_file_object(self, file_object) -> ResumeDocument:
        """
        Parse a resume from a file object (e.g., from Streamlit upload).
        
        Args:
            file_object: File-like object (must have read() and name attributes)
            
        Returns:
            ResumeDocument object containing all extracted information
            
        Raises:
            ValueError: If the file format is not supported
            Exception: For parsing errors
        """
        # Step 1: Extract text from file object
        raw_text = self.text_extractor.extract_from_file_object(file_object)
        
        # Step 2: Parse the extracted text
        return self.parse_text(raw_text)
    
    def parse_text(self, text: str) -> ResumeDocument:
        """
        Parse resume text (already extracted).
        
        This method is useful when text is already extracted from another source
        (e.g., database, API, or pre-processing).
        
        Args:
            text: Resume text (already extracted from document)
            
        Returns:
            ResumeDocument object containing all extracted information
        """
        # Step 1: Detect sections
        sections = self.section_parser.detect_sections(text)
        
        # Step 2: Extract contact information from header
        contact_info = self.section_parser.extract_contact_info(text)
        
        # Step 3: Extract section content
        summary_text = self.section_parser.extract_summary(text)
        skills_text = self.section_parser.extract_skills_section(text)
        experience_text = self.section_parser.extract_experience_section(text)
        education_text = self.section_parser.extract_education_section(text)
        projects_text = self.section_parser.extract_projects_section(text)
        certifications_text = self.section_parser.extract_certifications_section(text)
        languages_text = self.section_parser.extract_languages_section(text)
        
        # Step 4: Parse structured metadata from sections
        skills = self.metadata_parser.parse_skills(skills_text)
        experience = self.metadata_parser.parse_experience(experience_text)
        education = self.metadata_parser.parse_education(education_text)
        projects = self.metadata_parser.parse_projects(projects_text)
        certifications = self.metadata_parser.parse_certifications(certifications_text)
        languages = self.metadata_parser.parse_languages(languages_text)
        
        # Step 5: Extract additional metadata
        location = self.metadata_parser.extract_location(text)
        experience_years = self.metadata_parser.extract_experience_years(text)
        
        # Step 6: Build unified ResumeDocument
        document = ResumeDocument(
            name=contact_info.get('name'),
            email=contact_info.get('email'),
            phone=contact_info.get('phone'),
            summary=summary_text,
            skills=skills,
            experience=experience,
            projects=projects,
            education=education,
            certifications=certifications,
            languages=languages,
            raw_text=text,
            metadata={
                'parsed_at': datetime.now().isoformat(),
                'location': location,
                'total_experience_years': experience_years,
                'sections_detected': list(sections.keys()),
            }
        )
        
        return document
    
    def parse_resume_text_only(self, text: str) -> dict:
        """
        Parse resume text and return a simple dictionary (legacy compatibility).
        
        This method provides backward compatibility with the old parser interface.
        It returns a simple dictionary with basic fields instead of the full
        ResumeDocument schema.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with basic extracted fields
        """
        # Use the new parser but convert to legacy format
        document = self.parse_text(text)
        
        return {
            'skills': document.skills,
            'experience': document.metadata.get('total_experience_years', 0),
            'location': document.metadata.get('location', 'Not specified'),
            'role': document.experience[0].title if document.experience else 'Software Developer',
            'text': text
        }
