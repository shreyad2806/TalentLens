"""
Parser Package - Production-ready resume parsing architecture.

This package provides a clean, extensible parsing layer for extracting
structured data from resume documents (PDF, DOCX, TXT).

Modules:
    - schema: Data models for resume documents
    - extractor: Text extraction from various file formats
    - section_parser: Semantic section detection and parsing
    - metadata_parser: Structured field extraction
    - parser_service: Unified orchestration layer

The old parser (src/parser.py) remains untouched for backward compatibility.
"""

# Import schema (no external dependencies)
from .schema import ResumeDocument, Experience, Education, Project, Certification

# Lazy import parser service (requires PyPDF2, python-docx)
try:
    from .parser_service import ParserService
    _parser_service_available = True
except ImportError:
    _parser_service_available = False
    ParserService = None

__all__ = [
    "ResumeDocument",
    "Experience",
    "Education",
    "Project",
    "Certification",
]

if _parser_service_available:
    __all__.append("ParserService")
