"""
Indexing Package - Production indexing service for resume documents.

This package provides a complete indexing pipeline for resume documents,
including ingestion, parsing, chunking, embedding, and storage in both
vector stores and BM25 sparse indexes.

Modules:
- indexing_service: Main indexing service with public API
- resume_ingestor: Resume file ingestion logic
- pipeline: End-to-end indexing workflow
"""

from .indexing_service import IndexingService
from .resume_ingestor import ResumeIngestor
from .pipeline import IndexingPipeline

__all__ = [
    'IndexingService',
    'ResumeIngestor',
    'IndexingPipeline'
]
