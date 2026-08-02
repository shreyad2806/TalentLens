"""
Shared data models for the TalentLens ingestion pipeline.

These models form the unified schema into which every resume dataset must
be converted before parsing, chunking, embedding, or vector-store insertion.
"""

from .display_name import get_display_name
from .resume_document import ResumeDocument
from .resume_metadata import ResumeMetadata

__all__ = [
    "get_display_name",
    "ResumeDocument",
    "ResumeMetadata",
]
