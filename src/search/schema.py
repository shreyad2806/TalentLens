"""Schemas for the upgraded TalentLens search service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models import ResumeMetadata, get_display_name
from src.normalization.role_normalizer import RoleNormalizer


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
    ai_summary: str = Field("", description="Concise recruiter-friendly summary generated from retrieved content")
    resume_text: str | None = Field(None, description="Full resume text for the View Resume drawer")
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
    explanation: list[str] = Field(default_factory=list, description="Human-readable why-matched bullets")
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list, description="Top chunks used for ranking")

    def _clean_skills(self, skills: list[str] | None, matched: set[str] | None = None) -> list[str]:
        """Normalize skills via the canonical taxonomy; drop noise and fragments."""
        from src.normalization.skill_normalizer import SkillNormalizer

        cleaned = SkillNormalizer.normalize_list(skills or [])
        # Preserve matched skills at the front, in query order.
        if matched:
            matched_norm = {SkillNormalizer._key(m) for m in matched}
            matched_front = [c for c in cleaned if SkillNormalizer._key(c) in matched_norm]
            others = [c for c in cleaned if SkillNormalizer._key(c) not in matched_norm]
            cleaned = matched_front + others
        return cleaned

    def _ai_summary(self, m: ResumeMetadata, top_skills: list[str]) -> str:
        """Generate a 2–3 line recruiter-friendly summary from metadata."""
        parts: list[str] = []
        name = get_display_name(m, self.source_filename)
        role = (m.role or "").strip().title() if m.role else ""
        years = f"{m.experience_years:g} years" if m.experience_years and m.experience_years > 0 else ""
        location = (m.location or "").strip() if m.location else ""

        # Sentence 1: identity
        if name and role:
            parts.append(f"{name} is a {role}.")
        elif name:
            parts.append(f"{name}.")
        elif role:
            parts.append(f"{role} professional.")
        else:
            parts.append("Candidate profile.")

        # Sentence 2: experience + location
        context_parts = []
        if years:
            context_parts.append(years)
        if location:
            context_parts.append(f"based in {location}")
        if context_parts:
            parts.append("Brings " + ", ".join(context_parts) + ".")

        # Sentence 3: top skills
        if top_skills:
            parts.append("Key skills: " + ", ".join(top_skills[:6]) + ".")

        summary = " ".join(parts)
        # Guard against long summaries.
        if len(summary) > 220:
            summary = summary[:217] + "..."
        return summary

    def to_frontend_dict(self, evidence_offset: int = 0) -> dict[str, Any]:
        """Return a dict that is compatible with the Streamlit UI."""
        d = self.model_dump()
        m = self.resume_metadata
        s = self.score_breakdown

        matched_set = {sk.lower() for sk in self.matched_skills}
        all_skills = self._clean_skills(m.skills or [])
        top_skills = all_skills[:5]
        extra_skills = max(0, len(all_skills) - 5)

        d["id"] = m.resume_id
        d["name"] = get_display_name(m, self.source_filename)
        if m.experience_years is not None and m.experience_years > 0:
            d["experience"] = f"{m.experience_years:g} years"
        else:
            d["experience"] = "Not specified"
        d["top_skills"] = top_skills
        d["extra_skills"] = extra_skills
        d["all_skills_count"] = len(all_skills)
        d["skills"] = all_skills
        d["matched_skills"] = self._clean_skills(self.matched_skills, matched=matched_set)
        d["score"] = round(self.final_score, 4)
        d["section"] = self.matched_sections[0] if self.matched_sections else "unknown"
        d["evidence_offset"] = evidence_offset
        d["resume_preview"] = self.preview
        d["resume_text"] = self.resume_text or self.preview
        d["ai_summary"] = (self.ai_summary or self._ai_summary(m, top_skills)).strip()
        d["education"] = [(e or "").strip() for e in (m.education or [])[:3] if e and e.strip()]
        d["projects"] = (m.projects or [])[:6]
        d["certifications"] = (m.certifications or [])[:6]
        d["summary"] = (m.summary or "")[:250]
        d["role"] = RoleNormalizer.normalize(m.role) or m.role or ""
        d["matched_role"] = d["role"] if s.get("role", 0.0) > 0.0 else ""
        d["matched_industry"] = s.get("matched_industry", [])
        d["matched_education"] = s.get("matched_education", [])
        d["retrieved_sections"] = self.matched_sections
        d["location"] = m.location or ""
        d["explanation"] = self.explanation
        d["retrieved_chunks"] = self.retrieved_chunks
        d["overall_match"] = round(self.final_score * 100, 2)
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
        d["keyword_match"] = round(s.get("keyword", 0.0) * 100, 2)
        d["match_pct"] = d["overall_match"]

        # Build a deterministic "Why this matched" list with confidence scores.
        match_details: list[dict[str, Any]] = []
        if s.get("role", 0.0) > 0.0 and m.role:
            match_details.append({
                "label": "Role Match",
                "score": round(s.get("role", 0.0) * 100, 2),
                "value": d["matched_role"],
                "type": "role",
            })
        for sk in self.matched_skills:
            match_details.append({
                "label": f"{sk.title()} Match",
                "score": 100,
                "value": sk.title(),
                "type": "skill",
            })
        if s.get("experience", 0.0) > 0.0 and m.experience_years is not None and m.experience_years > 0:
            match_details.append({
                "label": "Experience Match",
                "score": round(s.get("experience", 0.0) * 100, 2),
                "value": f"{m.experience_years:g} years",
                "type": "experience",
            })
        for dt in s.get("matched_industry", []):
            match_details.append({
                "label": f"{dt.title()} Industry Match",
                "score": round(s.get("industry", 0.0) * 100, 2),
                "value": dt.title(),
                "type": "industry",
            })
        for edu in s.get("matched_education", []):
            match_details.append({
                "label": f"{edu} Education Match",
                "score": round(s.get("education", 0.0) * 100, 2),
                "value": edu,
                "type": "education",
            })
        if s.get("semantic", 0.0) > 0.0:
            match_details.append({
                "label": "Semantic Similarity",
                "score": round(s.get("semantic", 0.0) * 100, 2),
                "value": "",
                "type": "semantic",
            })
        d["match_details"] = match_details

        # Confidence is a blend of extraction quality and retrieval strength.
        if self.metadata_confidence:
            meta_conf = sum(self.metadata_confidence.values()) / len(self.metadata_confidence)
        else:
            meta_conf = 1.0
        retrieval_conf = min(1.0, (self.rrf_score or 0.0) * 50.0)
        d["confidence"] = round(min(1.0, meta_conf * (0.6 + 0.4 * retrieval_conf)), 2)

        return d
