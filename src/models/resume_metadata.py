"""Canonical, single-source metadata model for resumes."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ResumeMetadata(BaseModel):
    """
    Single canonical metadata object for a resume.

    This is the ONLY metadata object in the system. It is extracted once by the
    resume parser and passed unchanged through every downstream stage:
    ResumeDocument, Chunk, EmbeddingRecord, VectorRecord, BM25Document,
    SearchResult, HybridSearchResult, and the UI.

    Fields that cannot be extracted remain as None or empty lists. No stage
    should ever reconstruct, normalize, or fill this object with placeholders.
    """

    resume_id: str = Field(..., description="Stable resume identifier")
    candidate_name: Optional[str] = Field(None, description="Candidate full name")
    role: Optional[str] = Field(None, description="Current or primary role")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    location: Optional[str] = Field(None, description="Geographic location")
    experience_years: Optional[float] = Field(None, ge=0, le=60, description="Total years of experience")
    education: List[str] = Field(default_factory=list, description="Education entries")
    projects: List[str] = Field(default_factory=list, description="Project names")
    certifications: List[str] = Field(default_factory=list, description="Certification names")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    summary: Optional[str] = Field(None, description="Professional summary")

    model_config = {"extra": "ignore"}

    def __repr__(self) -> str:
        return (
            f"ResumeMetadata(resume_id={self.resume_id!r}, "
            f"candidate_name={self.candidate_name!r}, "
            f"skills={self.skills}, location={self.location!r}, "
            f"experience_years={self.experience_years})"
        )
