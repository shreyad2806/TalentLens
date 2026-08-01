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
from .quality_extractor import QualityMetadataExtractor
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
        self.quality_extractor = QualityMetadataExtractor()
    
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
    
    def parse_text(self, text: str, record: dict = None) -> ResumeDocument:
        """
        Parse resume text (already extracted).

        This method is useful when text is already extracted from another source
        (e.g., database, API, or pre-processing). If a CSV record is provided,
        its columns (Location, Education, Email, Phone) are used as trusted
        fallback sources.

        Args:
            text: Resume text (already extracted from document)
            record: Optional CSV record dict with known candidate fields.

        Returns:
            ResumeDocument object containing all extracted information
        """
        return self.quality_extractor.extract(text, record)
    
    def parse_resume_text_only(self, text: str, record: dict = None) -> dict:
        """
        Parse resume text and return a simple dictionary (legacy compatibility).

        This method provides backward compatibility with the old parser interface.
        It returns a simple dictionary with basic fields instead of the full
        ResumeDocument schema.

        Args:
            text: Resume text
            record: Optional CSV record dict with known candidate fields.

        Returns:
            Dictionary with basic extracted fields
        """
        # Use the new parser but convert to legacy format
        document = self.parse_text(text, record)

        return {
            'skills': document.skills,
            'experience': document.metadata.get('total_experience_years'),
            'location': document.metadata.get('location'),
            'role': document.experience[0].title if document.experience else None,
            'text': text
        }
