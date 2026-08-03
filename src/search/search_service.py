"""SearchService: semantic + metadata hybrid scoring over ResumeDocuments."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument, get_display_name
from src.preview import ResumePreviewGenerator
from src.summarization import generate_resume_summary

from .query_parser import ParsedQuery, QueryParser
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
# Keyword Match is sparse/term-overlap based; Semantic is dense/vector similarity.
OVERALL_SCORING_WEIGHTS = {
    "role": 0.25,
    "skill": 0.20,
    "keyword": 0.15,
    "industry": 0.10,
    "experience": 0.10,
    "education": 0.05,
    "location": 0.05,
    "semantic": 0.10,
}

FINAL_SCORE_WEIGHTS = {
    "cross_encoder": 0.40,
    "semantic": 0.25,
    "skill": 0.20,
    "role": 0.10,
    "experience": 0.05,
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
        if reranker is None and os.getenv("DISABLE_RERANKER", "false").lower() != "true":
            try:
                from src.search.cross_encoder_reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
            except Exception:
                reranker = None
        self.reranker = reranker
        self._resume_cache = _load_resume_cache()
        self._summary_cache: dict[str, str] = {}
        self.query_parser = QueryParser()
        self.last_parsed_query: ParsedQuery | None = None
        self.last_search_metrics: dict[str, Any] | None = None
        self.use_enhanced_ranking = use_enhanced_ranking or False
        self.ranking_weights = self._load_ranking_weights(ranking_weights)

    def parse_query(self, query: str) -> ParsedQuery:
        """Return the structured intent for a query (used by UI)."""
        return self.query_parser.parse(query)

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
        search_start = time.perf_counter()

        # Parse the query and enrich filters with extracted intent.
        parsed = self.query_parser.parse(query)
        self.last_parsed_query = parsed
        fd = filters.model_dump()
        if parsed.role and not fd.get("role"):
            fd["role"] = parsed.role
        if parsed.industry and not fd.get("industry"):
            fd["industry"] = parsed.industry
        if parsed.skills:
            existing = set(fd.get("skills") or [])
            existing.update(s.lower() for s in parsed.skills)
            fd["skills"] = sorted(existing)
        if parsed.education and not fd.get("education"):
            fd["education"] = parsed.education
        if parsed.location and not fd.get("location"):
            fd["location"] = parsed.location
        if parsed.experience_min is not None and fd.get("experience_min") is None:
            fd["experience_min"] = parsed.experience_min
        if parsed.experience_max is not None and fd.get("experience_max") is None:
            fd["experience_max"] = parsed.experience_max
        filters = SearchFilters(**fd)

        # Retrieval pipeline configuration:
        #   dense top 50 + sparse top 50 → RRF fuse → rerank top 20 → return top 10
        FUSED_POOL = 50
        RERANK_POOL = 20

        # 1. Hybrid retrieval: dense top 50 + sparse top 50 fused with RRF
        hybrid_results = []
        if self.hybrid_service is not None:
            for attempt in range(3):
                try:
                    hybrid_results = self.hybrid_service.search(
                        query, top_k=FUSED_POOL, filters=filters.model_dump(exclude_none=True)
                    )
                    break
                except Exception as exc:
                    print(f"[SearchService] vector retrieval attempt {attempt + 1} failed: {exc}")
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                    else:
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
        scoring_start = time.perf_counter()
        scored = []
        for resume, hybrid in pairs:
            result = self._score_resume(resume, hybrid, filters, query)
            if result is None:
                continue
            scored.append(result)
        scoring_time = time.perf_counter() - scoring_start

        # 4. Strict filtering (optional) before ranking
        if filters.strict:
            scored = [r for r in scored if r.metadata_score == 1.0]

        if not scored:
            self.last_search_metrics = {
                "total_time_ms": (time.perf_counter() - search_start) * 1000,
                "scoring_time_ms": scoring_time * 1000,
            }
            return []

        if self.use_enhanced_ranking:
            hybrid_metrics = (
                self.hybrid_service.last_metrics.model_dump()
                if self.hybrid_service and getattr(self.hybrid_service, "last_metrics", None)
                else {}
            )
            self.last_search_metrics = {
                "dense_candidates": hybrid_metrics.get("dense_candidate_count", 0),
                "sparse_candidates": hybrid_metrics.get("sparse_candidate_count", 0),
                "rrf_candidates": len(scored),
                "cross_encoder_candidates": 0,
                "returned_candidates": min(len(scored), top_k),
                "latency_ms": (time.perf_counter() - search_start) * 1000,
                "embedding_time_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
                "retrieval_time_ms": (hybrid_metrics.get("sparse_latency", 0.0) + hybrid_metrics.get("fusion_latency", 0.0)) * 1000,
                "rerank_time_ms": 0.0,
                "generation_time_ms": scoring_time * 1000,
            }
            return self._rank_enhanced(scored, top_k)

        # 5. Select top candidates by normalized weighted match for reranking
        scored.sort(key=lambda x: x.score_breakdown.get("overall", 0.0), reverse=True)
        rerank_pool = scored[: min(RERANK_POOL, len(scored))]

        # 6. Cross-encoder reranking (top 20 candidates)
        rerank_start = time.perf_counter()
        rerank_time = 0.0
        if self.reranker is not None:
            passages = []
            for r in rerank_pool:
                resume = self._resume_cache.get(r.resume_metadata.resume_id)
                text = r.matched_text
                if not text and resume is not None:
                    text = ResumePreviewGenerator().generate(resume)
                passages.append(text or "")

            rerank_scores: list[float] = []
            for attempt in range(3):
                try:
                    rerank_scores = self.reranker.rerank(query, passages)
                    break
                except Exception as exc:
                    print(f"[SearchService] rerank attempt {attempt + 1} failed: {exc}")
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        rerank_scores = [0.0] * len(passages)
            max_rerank = max(rerank_scores) if rerank_scores else 0.0
            for r, rs in zip(rerank_pool, rerank_scores):
                r.rerank_score = round(rs / max_rerank, 4) if max_rerank > 0 else 0.0
        rerank_time = time.perf_counter() - rerank_start

        # 7. Final ranking: hybrid rerank score.
        for r in rerank_pool:
            final = (
                FINAL_SCORE_WEIGHTS["cross_encoder"] * r.rerank_score
                + FINAL_SCORE_WEIGHTS["semantic"] * r.score_breakdown.get("semantic", 0.0)
                + FINAL_SCORE_WEIGHTS["skill"] * r.score_breakdown.get("skill", 0.0)
                + FINAL_SCORE_WEIGHTS["role"] * r.score_breakdown.get("role", 0.0)
                + FINAL_SCORE_WEIGHTS["experience"] * r.score_breakdown.get("experience", 0.0)
            )
            r.final_score = round(final, 4)

        rerank_pool.sort(key=lambda x: x.final_score, reverse=True)
        final = rerank_pool[:top_k]

        # 8. Retrieval quality log (per query)
        hybrid_metrics = (
            self.hybrid_service.last_metrics.model_dump()
            if self.hybrid_service and getattr(self.hybrid_service, "last_metrics", None)
            else {}
        )

        self.last_search_metrics = {
            "dense_candidates": hybrid_metrics.get("dense_candidate_count", 0),
            "sparse_candidates": hybrid_metrics.get("sparse_candidate_count", 0),
            "rrf_candidates": hybrid_metrics.get("fused_candidate_count", len(scored)),
            "cross_encoder_candidates": len(rerank_pool),
            "returned_candidates": len(final),
            "latency_ms": (time.perf_counter() - search_start) * 1000,
            "embedding_time_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
            "retrieval_time_ms": (hybrid_metrics.get("sparse_latency", 0.0) + hybrid_metrics.get("fusion_latency", 0.0)) * 1000,
            "rerank_time_ms": rerank_time * 1000,
            "generation_time_ms": scoring_time * 1000,
            "overlap": hybrid_metrics.get("overlap_count", 0),
            "dense_only": hybrid_metrics.get("dense_only_count", 0),
            "sparse_only": hybrid_metrics.get("sparse_only_count", 0),
        }

        logger.info(
            "Reranking complete | query=%r | fused=%d | reranked=%d (cross-encoder %s) | returned=%d",
            query[:80], len(scored), len(rerank_pool),
            "ON" if self.reranker is not None else "OFF", len(final),
        )
        for rank_i, r in enumerate(final, start=1):
            logger.info(
                "Top candidate %d | id=%s name=%s | final=%.4f semantic=%.4f bm25=%.4f "
                "skill=%.4f role=%.4f experience=%.4f | matched_skills=%s matched_role=%r industry=%.4f",
                rank_i,
                r.resume_metadata.resume_id,
                get_display_name(r.resume_metadata, r.source_filename),
                r.final_score,
                r.score_breakdown.get("semantic", 0.0),
                r.score_breakdown.get("sparse", 0.0),
                r.score_breakdown.get("skill", 0.0),
                r.score_breakdown.get("role", 0.0),
                r.score_breakdown.get("experience", 0.0),
                r.matched_skills,
                r.resume_metadata.role or "",
                r.score_breakdown.get("industry", 0.0),
            )

        return final

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

    def _get_ai_summary(
        self,
        resume: ResumeDocument,
        matched_text: str,
        query: str,
        matched_skills: list[str],
    ) -> str:
        """Generate and cache a concise recruiter-friendly summary from retrieved text."""
        key = f"{resume.resume_metadata.resume_id}::{query.strip().lower()}"
        if key in self._summary_cache:
            return self._summary_cache[key]

        m = resume.resume_metadata
        summary = generate_resume_summary(
            resume_text=resume.resume_text or "",
            matched_text=matched_text,
            query=query,
            role=m.role or "",
            education=m.education or [],
            skills=m.skills or [],
            matched_skills=matched_skills,
        )
        self._summary_cache[key] = summary
        return summary

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

        # --- keyword match (query term presence in resume text / matched text) ---
        keyword_hits = {t for t in query_terms if t in searchable_text or t in (matched_text or "").lower()}
        keyword_score = len(keyword_hits) / len(query_terms) if query_terms else 0.0

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
            "keyword": keyword_score,
        }

        applicability = {
            "role": filters.role is not None or role_score > 0.0,
            "skill": (skill_match_available and skill_score > 0.0) or (filters.skills is not None and len(filters.skills) > 0),
            "industry": domain_terms and industry_score > 0.0,
            "experience": filters.experience_min is not None or filters.experience_max is not None or experience_score > 0.0,
            "education": filters.education is not None or education_score > 0.0,
            "location": filters.location is not None or location_score > 0.0,
            "semantic": semantic_similarity > 0.0,
            "keyword": len(query_terms) > 0,
        }

        numerator = 0.0
        denominator = 0.0
        for key, applicable in applicability.items():
            if applicable:
                weight = OVERALL_SCORING_WEIGHTS[key]
                numerator += component_scores[key] * weight
                denominator += weight

        overall_match = round(min(1.0, numerator / denominator), 4) if denominator > 0 else 0.0

        # --- explainability reasons (uses only real evidence) ---
        explanation: list[str] = []
        if semantic_similarity > 0.05:
            explanation.append("Semantic similarity with query")

        _text_lower = resume_text
        for term in list(query_terms)[:4]:
            if term in _text_lower:
                explanation.append(f"{term.title()} keywords found")

        if role_score > 0.0 and resume.resume_metadata.role:
            explanation.append(f"{resume.resume_metadata.role} role detected")

        for sk in matched_skills[:4]:
            explanation.append(f"{sk.title()} matched")

        if industry_score > 0.0:
            for dt in list(domain_terms)[:2]:
                if dt in _text_lower:
                    explanation.append(f"{dt.title()} domain detected")

        if resume.resume_metadata.experience_years is not None and resume.resume_metadata.experience_years > 0:
            explanation.append(f"{resume.resume_metadata.experience_years:g} years experience")

        for edu in (resume.resume_metadata.education or [])[:2]:
            edu_lower = edu.lower()
            if any(term in edu_lower for term in query_terms) or any(dt in edu_lower for dt in domain_terms):
                explanation.append(f"{edu} detected")

        if location_score > 0.0 and resume.resume_metadata.location:
            explanation.append(f"{resume.resume_metadata.location} location match")

        _seen_reasons: set[str] = set()
        explanation = [r for r in explanation if not (r.lower() in _seen_reasons or _seen_reasons.add(r.lower()))]

        # --- top retrieved chunks for the drawer ---
        retrieved_chunks: list[dict[str, Any]] = []
        if hybrid is not None:
            for chunk in hybrid.matched_chunks[:5]:
                text = chunk.matched_text or getattr(chunk, "text", None) or ""
                retrieved_chunks.append({
                    "source": str(chunk.retrieval_source or "hybrid"),
                    "score": round(float(chunk.score or 0.0), 4),
                    "offset": chunk.offset or 0,
                    "text": text[:320] + ("..." if len(text) > 320 else ""),
                })

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

        # AI summary from retrieved resume content (cached per resume/query)
        ai_summary = self._get_ai_summary(
            resume=resume,
            matched_text=matched_text,
            query=query,
            matched_skills=matched_skills,
        )

        score_breakdown = {
            "overall": overall_match,
            "dense": round(semantic_score, 4),
            "sparse": round(bm25_score, 4),
            "keyword": round(keyword_score, 4),
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
            ai_summary=ai_summary,
            resume_text=resume.resume_text or "",
            explanation=explanation,
            retrieved_chunks=retrieved_chunks,
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


