"""Schemas for the upgraded TalentLens search service."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.models import ResumeMetadata, get_display_name
from src.normalization.role_normalizer import RoleNormalizer
from src.normalization.skill_importance import SkillImportanceRanker


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

    @staticmethod
    def _match_label(score: float) -> str:
        """Convert an overall match score into a recruiter-friendly rating."""
        if score >= 90:
            return "Excellent"
        if score >= 80:
            return "Strong"
        if score >= 70:
            return "Good"
        if score >= 55:
            return "Fair"
        return "Weak"

    def _ai_summary(self, m: ResumeMetadata, top_skills: list[str]) -> str:
        """Generate a 3-sentence recruiter-friendly summary from metadata."""
        parts: list[str] = []
        role = (m.primary_role or m.role or "").strip()
        role_family = (m.role_family or "").strip()
        key_domain = role_family or role

        # Sentence 1: role + years of experience
        if role and m.experience_years is not None and m.experience_years > 0:
            parts.append(f"{role} with {m.experience_years:g} years of experience.")
        elif role:
            parts.append(f"{role} with relevant experience.")
        elif m.experience_years is not None and m.experience_years > 0:
            parts.append(f"Professional with {m.experience_years:g} years of experience.")
        else:
            parts.append("Candidate profile.")

        # Sentence 2: primary technologies and technical strengths
        if top_skills:
            tech = [s for s in top_skills if s][:6]
            if len(tech) == 1:
                tech_clause = tech[0]
            else:
                tech_clause = ", ".join(tech[:-1]) + f" and {tech[-1]}" if len(tech) > 1 else tech[0]
            extras: list[str] = []
            if m.projects:
                extras.extend([p for p in m.projects if p][:2])
            if m.certifications:
                extras.extend([c for c in m.certifications if c][:1])
            if extras:
                parts.append(f"Strong background in {tech_clause}, with experience in {' and '.join(extras)}.")
            else:
                parts.append(f"Strong background in {tech_clause}.")

        # Sentence 3: domain expertise, notable achievements or specialization
        if m.summary:
            clean = re.sub(r"\s+", " ", m.summary).strip()
            if clean:
                if len(clean) > 120:
                    clean = clean[:117] + "..."
                parts.append(f"Specialized in {clean}.")
        elif m.projects:
            projects_list = [p for p in m.projects if p][:2]
            if projects_list:
                parts.append(f"Notable projects include {' and '.join(projects_list)}.")
        elif m.certifications:
            certs_list = [c for c in m.certifications if c][:2]
            if certs_list:
                parts.append(f"Certified in {' and '.join(certs_list)}.")
        elif key_domain:
            parts.append(f"Domain focus: {key_domain}.")
        elif m.education and m.education[0]:
            parts.append(f"Background includes {str(m.education[0]).strip()}.")
        else:
            parts.append("Relevant background for this search.")

        summary = " ".join(parts)
        if len(summary) > 280:
            summary = summary[:277] + "..."
        return summary

    def to_frontend_dict(self, evidence_offset: int = 0) -> dict[str, Any]:
        """Return a dict that is compatible with the Streamlit UI."""
        d = self.model_dump()
        m = self.resume_metadata
        s = self.score_breakdown

        matched_set = {sk.lower() for sk in self.matched_skills}
        all_skills = self._clean_skills(m.skills or [])
        ranked = SkillImportanceRanker.rank(
            all_skills,
            role_family=m.role_family,
            primary_role=m.primary_role or m.role,
        )
        primary_skills = ranked["primary"]
        secondary_skills = ranked["secondary"]
        top_skills = primary_skills[:5]
        extra_skills = max(0, len(primary_skills) - 5)

        d["id"] = m.resume_id
        d["name"] = get_display_name(m, self.source_filename)
        if m.experience_years is not None and m.experience_years > 0:
            d["experience"] = f"{m.experience_years:g} years"
        else:
            d["experience"] = "Not specified"
        d["top_skills"] = top_skills
        d["extra_skills"] = extra_skills
        d["all_skills_count"] = len(all_skills)
        d["primary_skills_count"] = len(primary_skills)
        d["secondary_skills_count"] = len(secondary_skills)
        d["skills"] = all_skills
        d["primary_skills"] = primary_skills
        d["secondary_skills"] = secondary_skills
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
        d["primary_role"] = RoleNormalizer.normalize(m.primary_role) or m.primary_role or "Role not specified"
        d["role"] = d["primary_role"]
        d["role_family"] = m.role_family or ""
        d["seniority"] = m.seniority or ""
        d["matched_role"] = d["role"] if s.get("role", 0.0) > 0.0 else ""
        d["matched_industry"] = s.get("matched_industry", [])
        d["matched_education"] = s.get("matched_education", [])
        d["matched_experience"] = d["experience"] if s.get("experience", 0.0) > 0.0 else ""
        d["retrieved_sections"] = self.matched_sections
        d["location"] = m.location or ""
        d["explanation"] = self.explanation
        d["retrieved_chunks"] = self.retrieved_chunks
        d["retrieved_chunk_ids"] = [
            str(chunk.get("chunk_id", "")) for chunk in self.retrieved_chunks
        ]
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
        d["project_match"] = round(s.get("project", 0.0) * 100, 2)
        d["match_pct"] = d["overall_match"]
        d["match_label"] = self._match_label(d["overall_match"])
        d["score_breakdown"] = {
            "Role": round(s.get("role", 0.0) * 100, 2),
            "Skills": round(s.get("skill", 0.0) * 100, 2),
            "Experience": round(s.get("experience", 0.0) * 100, 2),
            "Semantic": round(s.get("semantic", 0.0) * 100, 2),
            "Industry": round(s.get("industry", 0.0) * 100, 2),
            "Education": round(s.get("education", 0.0) * 100, 2),
        }

        # Build a deterministic "Why this matched" list with confidence scores.
        match_details: list[dict[str, Any]] = []
        if s.get("role", 0.0) > 0.0 and m.primary_role:
            role_score = round(s.get("role", 0.0) * 100, 2)
            query_role = (s.get("query_role") or "").strip()
            if role_score >= 99:
                role_label = "Exact Role Match"
                role_value = m.primary_role
            elif role_score >= 85:
                role_label = "Similar Role"
                role_value = f"{m.primary_role} ↔ {RoleNormalizer.normalize(query_role) or query_role.title()}"
            elif role_score >= 60:
                role_label = "Related Role"
                role_value = f"{m.primary_role} ↔ {RoleNormalizer.normalize(query_role) or query_role.title()}"
            else:
                role_label = "Weak Role Match"
                role_value = m.primary_role
            match_details.append({
                "label": role_label,
                "score": role_score,
                "value": role_value,
                "type": "role",
            })
        skill_evidence = s.get("skill_evidence", {})
        for sk in self.matched_skills:
            sections = skill_evidence.get(sk, [])
            value = sections[0] if sections else "Skills"
            match_details.append({
                "label": sk.title(),
                "score": 100,
                "value": value,
                "evidence": sections,
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

        # Confidence is the recruiter-oriented overall match score (0-100).
        d["suitability_score"] = round(self.final_score * 100, 2)
        d["confidence"] = d["suitability_score"]
        d["match_label"] = self._match_label(d["overall_match"])

        return d
