"""SearchService: semantic + metadata hybrid scoring over ResumeDocuments."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument, get_display_name
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

# Default weights for the enhanced ranking model.
# Exposed through the RANKING_WEIGHTS env var as JSON.
ENHANCED_RANKING_DEFAULT_WEIGHTS = {
    "dense": 0.25,
    "sparse": 0.15,
    "skill": 0.20,
    "role": 0.15,
    "experience": 0.10,
    "location": 0.10,
    "education": 0.05,
}

# Domain keywords that should not be treated as skills. They are matched against
# role, summary, experience, and education for industry/role relevance.
QUERY_DOMAINS = {
    "banking", "finance", "financial", "insurance", "accounting",
    "healthcare", "medical", "clinical", "pharma", "biotech",
    "construction", "civil", "architecture",
    "marketing", "sales", "advertising", "branding",
    "retail", "ecommerce", "logistics", "supply chain",
    "education", "teaching", "academic",
    "legal", "law", "hr", "human resources", "consulting",
    "manufacturing", "production", "operations",
    "hospitality", "hotel", "travel", "tourism",
    "real estate", "realestate",
}

# Weighted components for the new Overall Match score.
OVERALL_SCORING_WEIGHTS = {
    "role": 0.30,
    "skill": 0.30,
    "industry": 0.15,
    "experience": 0.10,
    "education": 0.05,
    "location": 0.05,
    "semantic": 0.05,
}

_RESUME_CACHE: dict[str, ResumeDocument] | None = None


def _load_resume_cache() -> dict[str, ResumeDocument]:
    """Lazy-load the unified production dataset keyed by candidate_id."""
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


def _fmt_experience(years: float | None) -> str:
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


def _tokenize_skills(skills: list[str] | None) -> set[str]:
    """Normalize a list of skill entries into lowercased, whitespace-free tokens.

    Handles both:
      - ['python', 'sql', 'aws']  (pre-tokenized list)
      - ['Python SQL AWS']        (single string with spaces)
    """
    if not skills:
        return set()
    tokens: set[str] = set()
    for entry in skills:
        if not entry:
            continue
        for raw in re.split(r"[,;\s]+", str(entry).strip()):
            token = raw.strip().lower()
            if token:
                tokens.add(token)
    return tokens


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
        hybrid_service: Any | None = None,
        reranker: Any | None = None,
        use_enhanced_ranking: bool | None = None,
        ranking_weights: dict[str, float] | None = None,
    ):
        self.hybrid_service = hybrid_service
        self.reranker = reranker
        self._resume_cache = _load_resume_cache()
        self.use_enhanced_ranking = (
            use_enhanced_ranking
            if use_enhanced_ranking is not None
            else os.getenv("USE_ENHANCED_RANKING", "false").lower() == "true"
        )
        self.ranking_weights = self._load_ranking_weights(ranking_weights)

    @staticmethod
    def _load_ranking_weights(overrides: dict[str, float] | None = None) -> dict[str, float]:
        """Load ranking weights from env, overrides, or defaults."""
        weights = dict(ENHANCED_RANKING_DEFAULT_WEIGHTS)
        env_weights = os.getenv("RANKING_WEIGHTS")
        if env_weights:
            try:
                parsed = json.loads(env_weights)
                if isinstance(parsed, dict):
                    weights.update(parsed)
            except Exception:
                pass
        if overrides:
            weights.update(overrides)
        return weights

    def _normalize_weights(self) -> dict[str, float]:
        """Normalize ranking weights so they sum to 1.0."""
        total = sum(self.ranking_weights.values())
        if total <= 0:
            return dict(ENHANCED_RANKING_DEFAULT_WEIGHTS)
        return {k: round(v / total, 4) for k, v in self.ranking_weights.items()}

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
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

        if self.use_enhanced_ranking:
            return self._rank_enhanced(scored, top_k)

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

    def _rank_enhanced(
        self,
        scored: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Normalize all feature scores and combine them with configurable weights."""
        if not scored:
            return []

        weights = self._normalize_weights()
        components = ["dense", "sparse", "skill", "role", "experience", "location", "education"]

        # Normalize each component globally across the result pool
        maxes = {c: max((r.score_breakdown.get(c, 0.0) for r in scored), default=1.0) for c in components}
        for r in scored:
            for c in components:
                raw = r.score_breakdown.get(c, 0.0)
                max_v = maxes[c]
                r.score_breakdown[c] = round(raw / max_v, 4) if max_v > 0 else 0.0

        # Compute final combined score
        for r in scored:
            final = sum(r.score_breakdown.get(c, 0.0) * weights.get(c, 0.0) for c in components)
            r.final_score = round(final, 4)

        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[:top_k]

        # Print score breakdown
        print("\n" + "=" * 70)
        print("🎯 Candidate Score Breakdown (Enhanced Ranking)")
        print("=" * 70)
        for i, r in enumerate(top, start=1):
            meta = r.resume_metadata
            name = get_display_name(meta, r.source_filename)
            skill_val = f"{r.score_breakdown.get('skill', 0.0):.4f}" if r.skill_match_available else "N/A"
            print(f"\n#{i} {name} (id={meta.resume_id})")
            print(f"   Overall:      {r.score_breakdown.get('overall', 0.0):.4f}")
            print(f"   Dense:        {r.score_breakdown.get('dense', 0.0):.4f}")
            print(f"   Sparse:       {r.score_breakdown.get('sparse', 0.0):.4f}")
            print(f"   Role:         {r.score_breakdown.get('role', 0.0):.4f}")
            print(f"   Skill:        {skill_val}")
            print(f"   Industry:     {r.score_breakdown.get('industry', 0.0):.4f}")
            print(f"   Experience:   {r.score_breakdown.get('experience', 0.0):.4f}")
            print(f"   Location:     {r.score_breakdown.get('location', 0.0):.4f}")
            print(f"   Education:    {r.score_breakdown.get('education', 0.0):.4f}")
            print(f"   Semantic:     {r.score_breakdown.get('semantic', 0.0):.4f}")
            print(f"   Final:        {r.final_score:.4f}")
        print("=" * 70 + "\n")

        return top

    def _score_resume(
        self,
        resume: ResumeDocument,
        hybrid: Any | None,
        filters: SearchFilters,
        query: str,
    ) -> SearchResult | None:
        """Compute all scores for a single resume using a weighted Overall Match."""
        raw_terms = query.lower().split()
        query_terms = {t for t in raw_terms if t.isalnum() and len(t) > 2}
        domain_terms = {t for t in query_terms if t in QUERY_DOMAINS}
        skill_terms = query_terms - domain_terms

        matched_fields: list[str] = []
        score_sum = 0.0
        total_weight = 0.0

        # --- metadata filter score (unchanged behavior) ---
        if filters.role:
            total_weight += FIELD_WEIGHTS["role"]
            if (resume.role or "").lower() in filters.role.lower() or filters.role.lower() in (resume.role or "").lower():
                score_sum += FIELD_WEIGHTS["role"]
                matched_fields.append("role")

        if filters.location:
            total_weight += FIELD_WEIGHTS["location"]
            if filters.location.lower() in (resume.location or "").lower():
                score_sum += FIELD_WEIGHTS["location"]
                matched_fields.append("location")

        if filters.experience_min is not None or filters.experience_max is not None:
            total_weight += FIELD_WEIGHTS["experience"]
            years = resume.experience_years or 0.0
            min_y = filters.experience_min or 0.0
            max_y = filters.experience_max or float("inf")
            if min_y <= years <= max_y:
                score_sum += FIELD_WEIGHTS["experience"]
                matched_fields.append("experience")

        matched_skills: list[str] = []
        if filters.skills:
            total_weight += FIELD_WEIGHTS["skills"]
            resume_skills = _tokenize_skills(resume.skills)
            wanted = _tokenize_skills(filters.skills)
            matched = resume_skills & wanted
            if matched:
                score_sum += FIELD_WEIGHTS["skills"] * (len(matched) / len(wanted))
                matched_fields.append("skills")
                matched_skills = sorted(matched)

        if filters.education:
            total_weight += FIELD_WEIGHTS["education"]
            edu_strings = [_fmt_education(e) for e in resume.education if _fmt_education(e)]
            if any(filters.education.lower() in s.lower() for s in edu_strings):
                score_sum += FIELD_WEIGHTS["education"]
                matched_fields.append("education")

        if filters.certifications:
            total_weight += FIELD_WEIGHTS["certifications"]
            cert_strings = [_fmt_certification(c) for c in resume.certifications if _fmt_certification(c)]
            if any(filters.certifications.lower() in s.lower() for s in cert_strings):
                score_sum += FIELD_WEIGHTS["certifications"]
                matched_fields.append("certifications")

        if filters.source_dataset:
            total_weight += FIELD_WEIGHTS["source_dataset"]
            if resume.source_dataset == filters.source_dataset:
                score_sum += FIELD_WEIGHTS["source_dataset"]
                matched_fields.append("source_dataset")

        metadata_score = score_sum / total_weight if total_weight > 0 else 0.0

        # Make sure education strings are available for the breakdown
        edu_strings = [_fmt_education(e) for e in resume.education if _fmt_education(e)]

        # --- vector scores (retrieval) ---
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
                first = hybrid.matched_chunks[0]
                matched_text = first.matched_text or ""
                evidence_offset = first.offset or 0

            for chunk in hybrid.matched_chunks:
                src = str(chunk.retrieval_source).lower() if chunk.retrieval_source else ""
                if "dense" in src:
                    semantic_score = max(semantic_score, float(chunk.score or 0.0))
                elif "sparse" in src:
                    bm25_score = max(bm25_score, float(chunk.score or 0.0))

        # --- skill matching (only when actual skills exist) ---
        wanted_skills = _tokenize_skills(filters.skills) or skill_terms
        skill_match_available = bool(wanted_skills)
        if skill_match_available and resume.skills:
            resume_skills = _tokenize_skills(resume.skills)
            matched_skills = sorted(resume_skills & wanted_skills)
            skill_score = min(1.0, len(matched_skills) / len(wanted_skills))
        else:
            skill_score = 0.0

        skill_display = "N/A" if not skill_match_available else f"{round(skill_score * 100, 2)}%"

        # --- role match (whole or partial query term presence in role) ---
        role_text = (resume.role or "").lower()
        role_hits = {t for t in query_terms if t in role_text}
        role_score = 1.0 if domain_terms and any(d in role_text for d in domain_terms) else (
            len(role_hits) / len(query_terms) if query_terms else 0.0
        )

        # --- industry match (domain terms in resume text) ---
        resume_text = (resume.resume_text or "").lower()
        summary_text = (resume.summary or "").lower()
        searchable_text = f"{resume_text} {summary_text}"
        industry_hits = {d for d in domain_terms if d in searchable_text}
        industry_score = (
            1.0 if domain_terms and (domain_terms & industry_hits) else
            len(industry_hits) / len(domain_terms) if domain_terms else 0.0
        )

        # --- experience match (query terms in resume text, plus explicit filter) ---
        min_y = filters.experience_min or 0.0
        max_y = filters.experience_max or float("inf")
        if filters.experience_min is not None or filters.experience_max is not None:
            years = resume.experience_years or 0.0
            exp_filter_match = min_y <= years <= max_y
        else:
            exp_filter_match = False

        exp_hits = {t for t in query_terms if t in resume_text}
        experience_score = 1.0 if exp_filter_match else (
            len(exp_hits) / len(query_terms) if query_terms else 0.0
        )

        # --- education match (query terms in education entries) ---
        edu_hits = {
            t for t in query_terms
            if any(t in s.lower() for s in edu_strings)
        }
        education_score = (
            len(edu_hits) / len(query_terms) if query_terms and edu_strings else 0.0
        )

        # --- location match ---
        location_text = (resume.location or "").lower()
        location_score = 1.0 if (
            (resume.location and any(t in location_text for t in query_terms))
            or (filters.location and filters.location.lower() in location_text)
        ) else 0.0

        # --- semantic similarity from dense retrieval ---
        semantic_similarity = min(1.0, max(0.0, semantic_score))

        # --- Overall Match (dynamically normalized weighted composite) ---
        component_scores = {
            "role": role_score,
            "skill": skill_score,
            "industry": industry_score,
            "experience": experience_score,
            "education": education_score,
            "location": location_score,
            "semantic": semantic_similarity,
        }

        applicability = {
            "role": filters.role is not None or role_score > 0.0,
            "skill": (skill_match_available and skill_score > 0.0) or (filters.skills is not None and len(filters.skills) > 0),
            "industry": domain_terms and industry_score > 0.0,
            "experience": filters.experience_min is not None or filters.experience_max is not None or experience_score > 0.0,
            "education": filters.education is not None or education_score > 0.0,
            "location": filters.location is not None or location_score > 0.0,
            "semantic": semantic_similarity > 0.0,
        }

        numerator = 0.0
        denominator = 0.0
        for key, applicable in applicability.items():
            if applicable:
                weight = OVERALL_SCORING_WEIGHTS[key]
                numerator += component_scores[key] * weight
                denominator += weight

        overall_match = round(min(1.0, numerator / denominator), 4) if denominator > 0 else 0.0

        # --- derived match sections for the UI ---
        query_terms_lower = {t for t in query_terms}
        matched_projects = [p for p in (resume.projects or []) if any(t in p.lower() for t in query_terms_lower)][:3]
        matched_certifications = [c for c in (resume.certifications or []) if any(t in c.lower() for t in query_terms_lower)][:3]

        matched_sections: list[str] = []
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

        score_breakdown = {
            "overall": overall_match,
            "dense": round(semantic_score, 4),
            "sparse": round(bm25_score, 4),
            "skill": round(skill_score, 4),
            "role": round(role_score, 4),
            "industry": round(industry_score, 4),
            "experience": round(experience_score, 4),
            "location": round(location_score, 4),
            "education": round(education_score, 4),
            "semantic": round(semantic_similarity, 4),
            "applicable": [k for k, v in applicability.items() if v],
            "raw_scores": {k: round(v, 4) for k, v in component_scores.items()},
            "denominator": round(denominator, 4),
        }

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
            boost_score=round(metadata_score, 4),
            rerank_score=0.0,
            final_score=overall_match,
            metadata_score=round(metadata_score, 4),
            metadata_confidence=resume.metadata_confidence or {},
            source_dataset=resume.source_dataset or "unknown",
            source_filename=resume.metadata_source.get("source_filename", ""),
            skill_match_available=skill_match_available,
            score_breakdown=score_breakdown,
        )


