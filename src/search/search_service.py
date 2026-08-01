"""SearchService: semantic + metadata hybrid scoring over ResumeDocuments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument
from src.preview import ResumePreviewGenerator
from .schema import SearchFilters, SearchResult

# How much each filter field contributes to the metadata score.
FIELD_WEIGHTS = {
    "role": 0.25,
    "location": 0.15,
    "experience": 0.15,
    "skills": 0.20,
    "education": 0.10,
    "certifications": 0.10,
    "source_dataset": 0.10,
}

_RESUME_CACHE: Optional[Dict[str, ResumeDocument]] = None


def _load_resume_cache() -> Dict[str, ResumeDocument]:
    """Lazy-load the unified production dataset keyed by candidate_id."""
    global _RESUME_CACHE
    if _RESUME_CACHE is not None:
        return _RESUME_CACHE

    dataset_path = PROJECT_ROOT / "combined" / "production_dataset.json"
    cache: Dict[str, ResumeDocument] = {}

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


def _fmt_experience(years: Optional[float]) -> str:
    if years is not None and years > 0:
        return f"{years:.1f} years"
    return "Not specified"


def _fmt_education(edu) -> str:
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
    return " ".join(parts) if parts else ""


def _fmt_certification(cert) -> str:
    return getattr(cert, "name", cert) if not isinstance(cert, str) else cert


def _fmt_project(proj: Any) -> str:
    """Format a project entry handling both string and Project objects."""
    if isinstance(proj, str):
        return proj
    parts = [proj.name] if getattr(proj, "name", None) else []
    if getattr(proj, "technologies", None):
        parts.append(f"({', '.join(proj.technologies)})")
    return " ".join(parts) if parts else ""


class SearchService:
    """
    TalentLens search upgrade: semantic + metadata hybrid.

    - Dense (semantic) and BM25 (sparse) scores come from the injected hybrid
      retrieval service.
    - Metadata score is computed by matching the ResumeDocument against the
      structured `SearchFilters`.
    - Final score is a weighted combination of the vector fusion score and the
      metadata score.
    """

    def __init__(
        self,
        hybrid_service: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ):
        self.hybrid_service = hybrid_service
        self.reranker = reranker
        self._resume_cache = _load_resume_cache()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Search resumes using vector + metadata hybrid scoring.

        If a `hybrid_service` is not provided, the vector scores are zero and the
        ranking is purely by metadata score, which is useful for validating
        metadata filters before the vector index is fully populated.
        """
        filters = filters or SearchFilters()

        # 1. Vector retrieval (when available)
        hybrid_results = []
        if self.hybrid_service is not None:
            try:
                hybrid_results = self.hybrid_service.search(
                    query, top_k=top_k * 5, filters=None
                )
            except Exception as exc:
                print(f"[SearchService] vector retrieval unavailable: {exc}")
                hybrid_results = []

        # 2. Build a working list of (resume, hybrid_result) pairs
        if hybrid_results:
            seen_ids: set[str] = set()
            pairs = []
            for r in hybrid_results:
                if r.resume_id in seen_ids or r.resume_id not in self._resume_cache:
                    continue
                seen_ids.add(r.resume_id)
                pairs.append((self._resume_cache.get(r.resume_id), r))
        else:
            # Metadata-only fallback: scan the full production dataset
            pairs = [
                (resume, None)
                for resume in self._resume_cache.values()
            ]

        # 3. Score each resume
        scored = []
        for resume, hybrid in pairs:
            result = self._score_resume(resume, hybrid, filters, query)
            if result is None:
                continue
            scored.append(result)

        # 4. Strict filtering (optional) before ranking
        if filters.strict:
            scored = [r for r in scored if r.metadata_score == 1.0]

        if not scored:
            return []

        # 5. Initial ranking to select reranker candidates
        max_rrf = max(r.rrf_score for r in scored)
        max_boost = max(r.boost_score for r in scored) or 1.0

        def _prelim_score(r: SearchResult) -> float:
            rrf_norm = r.rrf_score / max_rrf if max_rrf > 0 else 0.0
            boost_norm = r.boost_score / max_boost if max_boost > 0 else 0.0
            if self.reranker is None:
                return 0.6 * rrf_norm + 0.4 * boost_norm
            return 0.5 * rrf_norm + 0.5 * boost_norm

        scored.sort(key=_prelim_score, reverse=True)
        rerank_pool = scored[: min(top_k * 5, len(scored))]

        # 6. Cross-encoder reranking
        if self.reranker is not None:
            passages = []
            for r in rerank_pool:
                resume = self._resume_cache.get(r.resume_id)
                text = r.matched_text
                if not text and resume is not None:
                    text = ResumePreviewGenerator().generate(resume)
                passages.append(text or "")

            rerank_scores = self.reranker.rerank(query, passages)
            max_rerank = max(rerank_scores) if rerank_scores else 0.0
            for r, rs in zip(rerank_pool, rerank_scores):
                r.rerank_score = round(rs / max_rerank, 4) if max_rerank > 0 else 0.0

        # 7. Final score combination: RRF + rerank + field boosts
        max_rrf = max(r.rrf_score for r in rerank_pool)
        max_rerank = max(r.rerank_score for r in rerank_pool) or 1.0
        max_boost = max(r.boost_score for r in rerank_pool) or 1.0

        for r in rerank_pool:
            rrf_norm = r.rrf_score / max_rrf if max_rrf > 0 else 0.0
            rerank_norm = r.rerank_score / max_rerank if max_rerank > 0 else 0.0
            boost_norm = r.boost_score / max_boost if max_boost > 0 else 0.0
            r.final_score = round(0.4 * rrf_norm + 0.3 * rerank_norm + 0.3 * boost_norm, 4)

        rerank_pool.sort(key=lambda x: x.final_score, reverse=True)
        return rerank_pool[:top_k]

    def _score_resume(
        self,
        resume: ResumeDocument,
        hybrid: Optional[Any],
        filters: SearchFilters,
        query: str,
    ) -> Optional[SearchResult]:
        """Compute all scores for a single resume."""
        query_terms = {t for t in query.lower().split() if t.isalnum() and len(t) > 2}
        matched_fields: List[str] = []
        score_sum = 0.0
        total_weight = 0.0

        # role
        if filters.role:
            total_weight += FIELD_WEIGHTS["role"]
            if (resume.role or "").lower() in filters.role.lower() or filters.role.lower() in (resume.role or "").lower():
                score_sum += FIELD_WEIGHTS["role"]
                matched_fields.append("role")

        # location
        if filters.location:
            total_weight += FIELD_WEIGHTS["location"]
            if filters.location.lower() in (resume.location or "").lower():
                score_sum += FIELD_WEIGHTS["location"]
                matched_fields.append("location")

        # experience
        if filters.experience_min is not None or filters.experience_max is not None:
            total_weight += FIELD_WEIGHTS["experience"]
            years = resume.experience_years or 0.0
            min_y = filters.experience_min or 0.0
            max_y = filters.experience_max or float("inf")
            if min_y <= years <= max_y:
                score_sum += FIELD_WEIGHTS["experience"]
                matched_fields.append("experience")

        # skills
        matched_skills: List[str] = []
        if filters.skills:
            total_weight += FIELD_WEIGHTS["skills"]
            resume_skills = {s.lower().strip() for s in (resume.skills or [])}
            wanted = {s.lower().strip() for s in filters.skills}
            matched = resume_skills & wanted
            if matched:
                score_sum += FIELD_WEIGHTS["skills"] * (len(matched) / len(wanted))
                matched_fields.append("skills")
                matched_skills = sorted(matched)

        # education
        if filters.education:
            total_weight += FIELD_WEIGHTS["education"]
            edu_strings = [_fmt_education(e) for e in resume.education if _fmt_education(e)]
            if any(filters.education.lower() in s.lower() for s in edu_strings):
                score_sum += FIELD_WEIGHTS["education"]
                matched_fields.append("education")

        # certifications
        if filters.certifications:
            total_weight += FIELD_WEIGHTS["certifications"]
            cert_strings = [_fmt_certification(c) for c in resume.certifications if _fmt_certification(c)]
            if any(filters.certifications.lower() in s.lower() for s in cert_strings):
                score_sum += FIELD_WEIGHTS["certifications"]
                matched_fields.append("certifications")

        # source dataset
        if filters.source_dataset:
            total_weight += FIELD_WEIGHTS["source_dataset"]
            if resume.source_dataset == filters.source_dataset:
                score_sum += FIELD_WEIGHTS["source_dataset"]
                matched_fields.append("source_dataset")

        metadata_score = score_sum / total_weight if total_weight > 0 else 0.0

        # Field boosts from the query (no hard filtering)
        boost_score = 0.0
        boost_sum = 0.0
        boost_total = 0.0

        # role
        if resume.role:
            boost_total += FIELD_WEIGHTS["role"]
            if any(t in (resume.role or "").lower() for t in query_terms):
                boost_sum += FIELD_WEIGHTS["role"]

        # location
        if resume.location:
            boost_total += FIELD_WEIGHTS["location"]
            if any(t in (resume.location or "").lower() for t in query_terms):
                boost_sum += FIELD_WEIGHTS["location"]

        # experience
        if resume.experience_years is not None and resume.experience_years > 0:
            boost_total += FIELD_WEIGHTS["experience"]
            if any(k in query_terms for k in ("years", "experience", "yr")):
                boost_sum += FIELD_WEIGHTS["experience"]

        # skills
        if resume.skills:
            boost_total += FIELD_WEIGHTS["skills"]
            resume_skills = {s.lower().strip() for s in resume.skills}
            if resume_skills & query_terms:
                boost_sum += FIELD_WEIGHTS["skills"] * (len(resume_skills & query_terms) / len(query_terms))

        # education
        if resume.education:
            boost_total += FIELD_WEIGHTS["education"]
            edu_strings = [_fmt_education(e) for e in resume.education if _fmt_education(e)]
            if any(t in s.lower() for s in edu_strings for t in query_terms):
                boost_sum += FIELD_WEIGHTS["education"]

        boost_score = boost_sum / boost_total if boost_total > 0 else 0.0

        # vector scores
        semantic_score = 0.0
        bm25_score = 0.0
        rrf_score = 0.0
        matched_text = ""
        evidence_offset = 0
        section = ""

        if hybrid is not None:
            rrf_score = float(hybrid.rrf_score or 0.0)
            section = hybrid.section or ""
            if hybrid.matched_chunks:
                # Use the first matched chunk for evidence
                first = hybrid.matched_chunks[0]
                matched_text = first.matched_text or ""
                evidence_offset = first.offset or 0

            for chunk in hybrid.matched_chunks:
                src = str(chunk.retrieval_source).lower() if chunk.retrieval_source else ""
                if "dense" in src:
                    semantic_score = max(semantic_score, float(chunk.score or 0.0))
                elif "sparse" in src:
                    bm25_score = max(bm25_score, float(chunk.score or 0.0))

        # Skill matches: from explicit filters if present, otherwise from query terms
        if not matched_skills and resume.skills:
            resume_skills_set = {s.lower().strip() for s in resume.skills}
            matched_skills = sorted(resume_skills_set & query_terms)

        # Derived matches from query terms
        query_terms_lower = {t for t in query_terms}
        matched_projects = [p for p in (resume.projects or []) if any(t in p.lower() for t in query_terms_lower)][:3]
        matched_certifications = [c for c in (resume.certifications or []) if any(t in c.lower() for t in query_terms_lower)][:3]

        matched_sections: List[str] = []
        if section:
            matched_sections.append(section.capitalize())
        for field in matched_fields:
            matched_sections.append(field.capitalize())
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
        matched_sections = list(dict.fromkeys(matched_sections))

        # Resume preview
        preview = ResumePreviewGenerator().generate(resume)

        return SearchResult(
            resume_metadata=resume.resume_metadata,
            preview=preview,
            matched_skills=matched_skills,
            matched_projects=matched_projects,
            matched_certifications=matched_certifications,
            matched_keywords=sorted(query_terms),
            matched_sections=matched_sections,
            matched_text=matched_text,
            dense_score=round(semantic_score, 4),
            bm25_score=round(bm25_score, 4),
            rrf_score=round(rrf_score, 4),
            boost_score=round(boost_score, 4),
            rerank_score=0.0,  # set later
            final_score=0.0,  # set later
            metadata_score=round(metadata_score, 4),
            metadata_confidence=resume.metadata_confidence or {},
            source_dataset=resume.source_dataset or "unknown",
        )


