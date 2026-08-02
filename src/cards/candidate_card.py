"""CandidateCard builder that consumes ResumeDocument metadata directly."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument
from src.preview import ResumePreviewGenerator
from src.search import SearchResult
from src.search.search_service import OVERALL_SCORING_WEIGHTS, QUERY_DOMAINS

# Lazy load cache
_RESUME_CACHE: dict[str, ResumeDocument] | None = None


def _load_resume_cache() -> dict[str, ResumeDocument]:
    """Load the unified production dataset and index by candidate_id."""
    global _RESUME_CACHE
    if _RESUME_CACHE is not None:
        return _RESUME_CACHE

    dataset_path = PROJECT_ROOT / "combined" / "production_dataset.json"
    cache: dict[str, ResumeDocument] = {}

    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw in data:
            try:
                doc = ResumeDocument.model_validate(raw)
                cache[doc.candidate_id] = doc
            except Exception:
                continue

    _RESUME_CACHE = cache
    return _RESUME_CACHE


def _compute_match_score(resume_skills: list[str], jd_skills: list[str]) -> tuple[float, list[str]]:
    """Compute skill-match percentage and matched skills."""
    if not jd_skills:
        return 0.0, []

    resume_set = {s.lower().strip() for s in resume_skills if s}
    jd_set = {s.lower().strip() for s in jd_skills if s}
    matched = sorted(resume_set & jd_set)
    score = min((len(matched) / len(jd_set)) * 100, 100.0) if jd_set else 0.0
    return round(score, 2), matched


def _format_education(edu) -> str:
    """Format a single education entry (handles both string and Education objects)."""
    if isinstance(edu, str):
        return edu
    parts = [
        p for p in [
            getattr(edu, 'degree', None),
            getattr(edu, 'field', None),
            getattr(edu, 'field_of_study', None),
            getattr(edu, 'university', None),
            getattr(edu, 'institution', None),
        ] if p
    ]
    return " ".join(parts) if parts else "Education"


def _format_project(proj: Any) -> str:
    """Format a single project entry (handles both string and Project objects)."""
    if isinstance(proj, str):
        return proj
    parts = [proj.name] if getattr(proj, "name", None) else []
    if getattr(proj, "technologies", None):
        parts.append(f"({', '.join(proj.technologies)})")
    return " ".join(parts) if parts else "Project"


def _compute_confidence(resume: ResumeDocument) -> float:
    """Aggregate per-field confidence into a single score."""
    conf = resume.metadata_confidence or {}
    values = [v for v in conf.values() if isinstance(v, (int, float))]
    if values:
        return round(sum(values) / len(values), 2)
    return 1.0


def build_candidate_card(
    resume_id: str,
    rrf_score: float,
    jd_skills: list[str],
    matched_text: str = "",
    evidence_offset: int = 0,
    section: str = "unknown",
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
    query: str = "",
) -> dict[str, Any] | None:
    """
    Build a complete frontend candidate card from ResumeDocument metadata.

    The result is assembled into the canonical SearchResult schema and then
    converted to the existing frontend dict. No candidate assembly or lookup
    happens in the UI.
    """
    cache = _load_resume_cache()
    resume = cache.get(resume_id)
    if not resume:
        return None

    # --- query tokens and domain/skill split ---
    raw_query_terms = {t for t in (query or " ").lower().split() if t.isalnum() and len(t) > 2}
    domain_terms = {t for t in raw_query_terms if t in QUERY_DOMAINS}
    skill_terms = raw_query_terms - domain_terms

    # Skill matches: use explicit JD skills when available
    wanted_skills = [s.lower().strip() for s in jd_skills if s] or sorted(skill_terms)
    skill_match_available = bool(wanted_skills)
    resume_skills = [s.strip() for s in (resume.skills or []) if s]
    matched_skills = sorted({
        s for s in resume_skills
        if any(t in s.lower() for t in wanted_skills)
    })

    # Derived matches from query terms
    matched_projects = [
        p for p in (resume.projects or [])
        if any(t in p.lower() for t in raw_query_terms)
    ][:3]

    matched_certifications = [
        c for c in (resume.certifications or [])
        if any(t in c.lower() for t in raw_query_terms)
    ][:3]

    # Matched sections
    matched_sections: list[str] = []
    if section and section.lower() != "unknown":
        matched_sections.append(section.capitalize())
    if matched_skills:
        matched_sections.append("Skills")
    if matched_projects:
        matched_sections.append("Projects")
    if matched_certifications:
        matched_sections.append("Certifications")
    if resume.education:
        matched_sections.append("Education")
    if resume.experience_years is not None and resume.experience_years > 0:
        matched_sections.append("Experience")
    matched_sections = list(dict.fromkeys(matched_sections))  # dedupe

    # --- weighted match scoring ----------------------------------------
    role_text = (resume.role or "").lower()
    role_hits = {t for t in raw_query_terms if t in role_text}
    role_score = 1.0 if domain_terms and any(d in role_text for d in domain_terms) else (
        len(role_hits) / len(raw_query_terms) if raw_query_terms else 0.0
    )

    resume_text = (resume.resume_text or "").lower()
    summary_text = (resume.summary or "").lower()
    searchable_text = f"{resume_text} {summary_text}"
    industry_hits = {d for d in domain_terms if d in searchable_text}
    industry_score = (
        len(industry_hits) / len(domain_terms) if domain_terms else 0.0
    )

    exp_hits = {t for t in raw_query_terms if t in resume_text}
    experience_score = (
        len(exp_hits) / len(raw_query_terms) if raw_query_terms else 0.0
    )

    edu_hits = {
        t for t in raw_query_terms
        if any(t in str(e).lower() for e in resume.education)
    }
    education_score = (
        len(edu_hits) / len(raw_query_terms) if raw_query_terms and resume.education else 0.0
    )

    location_text = (resume.location or "").lower()
    location_score = 1.0 if (
        resume.location and any(t in location_text for t in raw_query_terms)
    ) else 0.0

    skill_score = (
        min(1.0, len(matched_skills) / len(wanted_skills)) if skill_match_available and wanted_skills else 0.0
    )

    semantic_similarity = min(1.0, max(0.0, rrf_score))

    weighted = (
        role_score * OVERALL_SCORING_WEIGHTS["role"]
        + (skill_score if skill_match_available else 0.0) * OVERALL_SCORING_WEIGHTS["skill"]
        + industry_score * OVERALL_SCORING_WEIGHTS["industry"]
        + experience_score * OVERALL_SCORING_WEIGHTS["experience"]
        + education_score * OVERALL_SCORING_WEIGHTS["education"]
        + location_score * OVERALL_SCORING_WEIGHTS["location"]
        + semantic_similarity * OVERALL_SCORING_WEIGHTS["semantic"]
    )
    overall_match = round(min(1.0, weighted), 4)

    score_breakdown = {
        "overall": overall_match,
        "dense": round(dense_score, 4),
        "sparse": round(bm25_score, 4),
        "skill": round(skill_score, 4),
        "role": round(role_score, 4),
        "industry": round(industry_score, 4),
        "experience": round(experience_score, 4),
        "location": round(location_score, 4),
        "education": round(education_score, 4),
        "semantic": round(semantic_similarity, 4),
    }

    # Resume preview generated by the backend preview generator
    preview = ResumePreviewGenerator().generate(resume)

    result = SearchResult(
        resume_metadata=resume.resume_metadata,
        preview=preview,
        matched_skills=matched_skills,
        matched_projects=matched_projects,
        matched_certifications=matched_certifications,
        matched_keywords=sorted(raw_query_terms),
        matched_sections=matched_sections,
        matched_text=matched_text or "",
        dense_score=round(dense_score, 4),
        bm25_score=round(bm25_score, 4),
        rrf_score=round(rrf_score, 4),
        rerank_score=round(rrf_score, 4),
        final_score=overall_match,
        metadata_confidence=resume.metadata_confidence or {},
        source_dataset=resume.source_dataset or "unknown",
        source_filename=resume.metadata_source.get("source_filename", ""),
        skill_match_available=skill_match_available,
        score_breakdown=score_breakdown,
    )

    return result.to_frontend_dict(evidence_offset=evidence_offset)
