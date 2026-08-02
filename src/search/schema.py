"""Schemas for the upgraded TalentLens search service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models import ResumeMetadata, get_display_name


class SearchFilters(BaseModel):
    """Structured filters for resume search."""

    role: str | None = Field(None, description="Role / job title substring")
    location: str | None = Field(None, description="Location substring")
    experience_min: float | None = Field(None, ge=0, description="Minimum years of experience")
    experience_max: float | None = Field(None, ge=0, description="Maximum years of experience")
    skills: list[str] | None = Field(None, description="Required skills")
    education: str | None = Field(None, description="Education keyword")
    certifications: str | None = Field(None, description="Certification keyword")
    source_dataset: str | None = Field(None, description="Source dataset name")
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
    matched_skills: list[str] = Field(default_factory=list, description="Skills that matched the query")
    matched_projects: list[str] = Field(default_factory=list, description="Projects that matched the query")
    matched_certifications: list[str] = Field(default_factory=list, description="Certifications that matched the query")
    matched_keywords: list[str] = Field(default_factory=list, description="Query keywords that had a match")
    matched_sections: list[str] = Field(default_factory=list, description="Resume sections that contributed evidence")
    matched_text: str = Field("", description="Text that matched the query")
    dense_score: float = Field(0.0, description="Dense (semantic) retrieval score")
    bm25_score: float = Field(0.0, description="BM25 sparse retrieval score")
    rrf_score: float = Field(0.0, description="RRF hybrid score")
    boost_score: float = Field(0.0, description="Field-boost score from query terms")
    rerank_score: float = Field(0.0, description="Cross-encoder reranker score")
    final_score: float = Field(0.0, description="Final combined score used for ranking")
    metadata_score: float = Field(0.0, description="Metadata filter match score")
    metadata_confidence: dict[str, float] = Field(default_factory=dict, description="Per-field extraction confidence")
    source_dataset: str = Field(..., description="Originating dataset")
    source_filename: str = Field("", description="Original resume filename if available")
    skill_match_available: bool = Field(True, description="Whether the query contained explicit skills")
    score_breakdown: dict[str, Any] = Field(default_factory=dict, description="Normalized per-feature scores")

    def to_frontend_dict(self, evidence_offset: int = 0) -> dict[str, Any]:
        """Return a dict that is compatible with the existing Streamlit UI."""
        d = self.model_dump()
        m = self.resume_metadata
        s = self.score_breakdown
        d["id"] = m.resume_id
        d["name"] = get_display_name(m, self.source_filename)
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
        d["overall_match"] = round(s.get("overall", 0.0) * 100, 2)
        d["role_match"] = round(s.get("role", 0.0) * 100, 2)
        if self.skill_match_available:
            d["skill_match"] = round(s.get("skill", 0.0) * 100, 2)
        else:
            d["skill_match"] = "N/A"
        d["experience_match"] = round(s.get("experience", 0.0) * 100, 2)
        d["location_match"] = round(s.get("location", 0.0) * 100, 2)
        d["industry_match"] = round(s.get("industry", 0.0) * 100, 2)
        d["education_match"] = round(s.get("education", 0.0) * 100, 2)
        d["semantic_match"] = round(s.get("semantic", 0.0) * 100, 2)
        # Legacy UI key now reflects the overall composite score
        d["match_pct"] = d["overall_match"]
        if self.metadata_confidence:
            d["confidence"] = round(
                sum(self.metadata_confidence.values()) / len(self.metadata_confidence), 2
            )
        else:
            d["confidence"] = 1.0
        return d
