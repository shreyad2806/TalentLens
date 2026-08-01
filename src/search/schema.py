"""Schemas for the upgraded TalentLens search service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models import ResumeMetadata


class SearchFilters(BaseModel):
    """Structured filters for resume search."""

    role: Optional[str] = Field(None, description="Role / job title substring")
    location: Optional[str] = Field(None, description="Location substring")
    experience_min: Optional[float] = Field(None, ge=0, description="Minimum years of experience")
    experience_max: Optional[float] = Field(None, ge=0, description="Maximum years of experience")
    skills: Optional[List[str]] = Field(None, description="Required skills")
    education: Optional[str] = Field(None, description="Education keyword")
    certifications: Optional[str] = Field(None, description="Certification keyword")
    source_dataset: Optional[str] = Field(None, description="Source dataset name")
    strict: bool = Field(False, description="If True, drop results that do not satisfy every filter")


class SearchResult(BaseModel):
    """
    Final production search result schema.

    Canonical candidate metadata lives only in `resume_metadata`.  All
    candidate-facing fields (name, role, skills, education, etc.) are read from
    that object.  The frontend dict is built by `to_frontend_dict`, which
    projects the values the Streamlit UI expects without duplicating them in
    the model.
    """

    resume_metadata: ResumeMetadata = Field(..., description="Canonical resume metadata")
    preview: str = Field(..., description="Short resume preview")
    matched_skills: List[str] = Field(default_factory=list, description="Skills that matched the query")
    matched_projects: List[str] = Field(default_factory=list, description="Projects that matched the query")
    matched_certifications: List[str] = Field(default_factory=list, description="Certifications that matched the query")
    matched_keywords: List[str] = Field(default_factory=list, description="Query keywords that had a match")
    matched_sections: List[str] = Field(default_factory=list, description="Resume sections that contributed evidence")
    matched_text: str = Field("", description="Text that matched the query")
    dense_score: float = Field(0.0, description="Dense (semantic) retrieval score")
    bm25_score: float = Field(0.0, description="BM25 sparse retrieval score")
    rrf_score: float = Field(0.0, description="RRF hybrid score")
    boost_score: float = Field(0.0, description="Field-boost score from query terms")
    rerank_score: float = Field(0.0, description="Cross-encoder reranker score")
    final_score: float = Field(0.0, description="Final combined score used for ranking")
    metadata_score: float = Field(0.0, description="Metadata filter match score")
    metadata_confidence: Dict[str, float] = Field(default_factory=dict, description="Per-field extraction confidence")
    source_dataset: str = Field(..., description="Originating dataset")

    def to_frontend_dict(self, evidence_offset: int = 0) -> Dict[str, Any]:
        """Return a dict that is compatible with the existing Streamlit UI."""
        d = self.model_dump()
        m = self.resume_metadata
        d["id"] = m.resume_id
        d["name"] = m.candidate_name or "Unknown"
        if m.experience_years is not None and m.experience_years > 0:
            d["experience"] = f"{m.experience_years:.1f} years"
        else:
            d["experience"] = "Not specified"
        d["top_skills"] = (m.skills or [])[:6]
        d["score"] = round(self.final_score, 4)
        d["section"] = self.matched_sections[0] if self.matched_sections else "unknown"
        d["evidence_offset"] = evidence_offset
        d["resume_preview"] = self.preview
        d["education"] = (m.education or [])[:6]
        d["skills"] = m.skills or []
        d["projects"] = m.projects or []
        d["certifications"] = m.certifications or []
        d["summary"] = m.summary or ""
        d["role"] = m.role or ""
        d["location"] = m.location or ""
        if m.skills:
            d["match_pct"] = round(len(self.matched_skills) / len(m.skills) * 100, 2)
        else:
            d["match_pct"] = 0.0
        if self.metadata_confidence:
            d["confidence"] = round(
                sum(self.metadata_confidence.values()) / len(self.metadata_confidence), 2
            )
        else:
            d["confidence"] = 1.0
        return d
