"""
Bootstrap Package for Production System Initialization.

This package provides the bootstrap system that initializes the retrieval
system on application startup by loading resumes, parsing, chunking, embedding,
and indexing them into vector store and BM25 index.

Components:
- resume_loader: Loads resume files from configurable directories
- bootstrap_service: Main service orchestrating the bootstrap workflow
- startup_validator: Validates system state after bootstrap
- startup_report: Generates and prints startup statistics

Usage:
    from src.bootstrap import BootstrapService
    
    service = BootstrapService()
    service.bootstrap()  # Only runs if index is empty
    
    # Or force rebuild
    service.rebuild()
    
    # Check status
    status = service.status()
"""

from .bootstrap_service import BootstrapService
from .resume_loader import ResumeLoader
from .startup_validator import StartupValidator
from .startup_report import StartupReport

__all__ = [
    "BootstrapService",
    "ResumeLoader",
    "StartupValidator",
    "StartupReport",
]
