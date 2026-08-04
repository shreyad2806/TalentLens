"""SearchService: semantic + metadata hybrid scoring over ResumeDocuments."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import functools
import time
from pathlib import Path

logger = logging.getLogger(__name__)
DEBUG = os.environ.get("TALENTLENS_DEBUG", "0") == "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument, ResumeMetadata, get_display_name
from src.resume_parser.occupation_extractor import PrimaryOccupationExtractor
from src.preview import ResumePreviewGenerator
from src.summarization import generate_resume_summary

from .query_parser import ParsedQuery, get_query_parser
from .role_similarity import RoleSimilarityScorer
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
    "role": 0.40,
    "industry": 0.20,
    "skill": 0.20,
    "experience": 0.10,
    "dense": 0.10,  # semantic/vector contribution
    "sparse": 0.00,
    "location": 0.00,
    "education": 0.00,
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
# Role dominates; industry and skill are secondary; experience and semantic are supporting.
OVERALL_SCORING_WEIGHTS = {
    "role": 0.40,
    "industry": 0.20,
    "skill": 0.20,
    "experience": 0.10,
    "education": 0.00,
    "semantic": 0.10,
    "keyword": 0.0,
    "location": 0.0,
}

FINAL_SCORE_WEIGHTS = {
    "cross_encoder": 0.00,
    "semantic": 0.10,
    "skill": 0.20,
    "role": 0.40,
    "industry": 0.20,
    "experience": 0.10,
}

# Role-aware final ranking weights.
SUITABILITY_SCORE_WEIGHTS = {
    "role": 0.40,
    "semantic": 0.20,
    "skill": 0.20,
    "experience": 0.10,
    "industry": 0.05,
    "education": 0.05,
}


@functools.lru_cache(maxsize=2048)
def _recruiter_role_score(query_role: str | None, candidate_role: str | None) -> int:
    """Return a discrete 0-100 recruiter role similarity score.

    Same Role            -> 100
    Same Role Family     ->  90
    Related              ->  70
    Different Engineering->  20
    Completely Different ->   0
    """
    if not query_role or not candidate_role:
        return 0

    q = RoleSimilarityScorer.normalize(query_role)
    c = RoleSimilarityScorer.normalize(candidate_role)
    if not q or not c:
        return 0

    if q == c or q in c or c in q:
        return 100

    raw = RoleSimilarityScorer.score(query_role, candidate_role)
    if raw >= 0.95:
        return 90
    if raw >= 0.70:
        return 70
    if raw >= 0.20:
        return 20
    return 0


def _print_retrieval_quality(results: list[SearchResult], query: str) -> None:
    """Log one line per returned result for recruiter audit."""
    if not results:
        print(f"[DEBUG] Query='{query}' | No candidates returned")
        return
    for r in results:
        m = r.resume_metadata
        name = get_display_name(m, r.source_filename, r.resume_text)
        primary_role = m.primary_role or m.role_family or "Role not specified"
        role_sim = round(r.score_breakdown.get("role", 0.0) * 100, 2)
        hybrid = round(getattr(r, "rrf_score", r.score_breakdown.get("rrf", 0.0)) * 100, 2)
        cross = round(r.score_breakdown.get("cross_encoder", 0.0) * 100, 2)
        skill = round(r.score_breakdown.get("skill", 0.0) * 100, 2)
        experience = round(r.score_breakdown.get("experience", 0.0) * 100, 2)
        metadata = round(r.metadata_score * 100, 2)
        final = round(r.final_score * 100, 2)
        reason = "; ".join(r.explanation) if r.explanation else "No explicit reason"
        print(
            f"[DEBUG] Candidate={name} "
            f"| Primary Role={primary_role} "
            f"| Role Similarity={role_sim} "
            f"| Hybrid={hybrid} "
            f"| Cross={cross} "
            f"| Skill={skill} "
            f"| Experience={experience} "
            f"| Metadata={metadata} "
            f"| Final={final} "
            f"| Reason Included={reason}"
        )


_RESUME_CACHE: dict[str, ResumeDocument] | None = None
_PRECOMPUTED_SUMMARIES: dict[str, str] = {}
_RESUME_INDEX: dict[str, dict[str, Any]] = {}


def _build_resume_index(cache: dict[str, ResumeDocument]) -> dict[str, dict[str, Any]]:
    """Precompute searchable metadata text and token sets once per resume at load."""
    index: dict[str, dict[str, Any]] = {}
    for doc in cache.values():
        m = doc.resume_metadata
        exp_parts: list[str] = []
        for e in (m.experience or []):
            if isinstance(e, dict):
                exp_parts.append(" ".join(str(e.get(k, "")) for k in ("title", "company", "description")))
            else:
                exp_parts.append(" ".join(str(getattr(e, k, "")) for k in ("title", "company", "description")))
        exp_text = " ".join(exp_parts)
        proj_text = " ".join(m.projects or [])
        edu_text = " ".join(m.education or [])
        cert_text = " ".join(m.certifications or [])
        skills_text = ", ".join(m.skills or [])
        search_text = (
            f"Work Experience\n{exp_text}\n\n"
            f"Projects\n{proj_text}\n\n"
            f"Skills\n{skills_text}\n\n"
            f"Education\n{edu_text}\n\n"
            f"Certifications\n{cert_text}\n\n"
            f"Summary\n{m.summary or ''}"
        )
        search_text_lower = search_text.lower()
        index[m.resume_id] = {
            "search_text": search_text_lower,
            "search_text_tokens": _tokenize_skills(search_text_lower.split()),
            "skills_tokens": _tokenize_skills(m.skills),
            "summary_text": (m.summary or "").lower(),
            "project_text": proj_text.lower(),
            "experience_years": m.experience_years or 0.0,
        }
    return index


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

    # Backfill primary occupation for legacy indexes where it was never stored.
    for doc in cache.values():
        if not doc.resume_metadata.primary_role:
            extracted = PrimaryOccupationExtractor.extract(doc, category=doc.resume_metadata.role or "")
            if extracted.get("primary_role"):
                doc.resume_metadata.primary_role = extracted.get("primary_role")
            else:
                # Fallback to a normalized dataset role so the UI has a display role.
                fallback = (doc.resume_metadata.role or "Professional").replace("-", " ").title().strip()
                if fallback:
                    doc.resume_metadata.primary_role = fallback
            doc.resume_metadata.role_family = (
                extracted.get("role_family")
                or PrimaryOccupationExtractor._derive_role_family(
                    doc.resume_metadata.primary_role or "",
                    doc.resume_metadata.role or "",
                )
                or "Other"
            )
            doc.resume_metadata.seniority = extracted.get("seniority")

    # Precompute a deterministic, metadata-only summary once per resume at index load.
    global _PRECOMPUTED_SUMMARIES
    for doc in cache.values():
        m = doc.resume_metadata
        _PRECOMPUTED_SUMMARIES[m.resume_id] = generate_resume_summary(
            resume_text="",
            matched_text="",
            retrieved_chunks=[],
            primary_role=m.primary_role or m.role,
            role_family=m.role_family,
            experience_years=m.experience_years,
            education=m.education or [],
            skills=m.skills or [],
            matched_skills=None,
            summary=m.summary or doc.summary,
            projects=m.projects or [],
            certifications=m.certifications or [],
        )

    global _RESUME_INDEX
    _RESUME_INDEX = _build_resume_index(cache)

    _RESUME_CACHE = cache
    return _RESUME_CACHE


def _stem_term(term: str) -> str:
    """Return a very lightweight stem for industry/role matching.

    Currently strips a trailing 'ing' so that 'banking' also matches 'bank',
    'corporate banking', 'bank manager', etc.
    """
    if term.endswith("ing"):
        return term[:-3]
    return term


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


class _StepTimer:
    """Lightweight step timer that prints per-step latency for _score_resume()."""

    def __init__(self, step: str, resume_id: str, extra: str | None = None, threshold_ms: float = 20.0) -> None:
        self.step = step
        self.resume_id = resume_id
        self.extra = extra
        self.threshold_ms = threshold_ms
        self.start = time.perf_counter()

    def stop(self) -> None:
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        extra_str = f" | {self.extra}" if self.extra else ""
        print(f"[PROFILE _score_resume] resume_id={self.resume_id} step={self.step} ms={elapsed_ms:.2f}{extra_str}")
        if elapsed_ms > self.threshold_ms:
            print(f"[PROFILE _score_resume] SLOW STEP: {self.step} took {elapsed_ms:.2f} ms{extra_str}")


# Evidence-tier weights for where a term appears in a resume.
# Work/professional experience is strongest, education/coursework is weakest.
SECTION_TIER_WEIGHTS = {
    "work_experience": 1.00,  # 50%
    "projects": 0.40,         # 20%
    "skills": 0.30,           # 15%
    "certifications": 0.20,   # 10%
    "education": 0.10,        # 5%
}

# Display labels for resume sections (used in explainability).
SECTION_DISPLAY = {
    "work_experience": "Experience",
    "projects": "Projects",
    "skills": "Skills",
    "certifications": "Certifications",
    "education": "Education",
}

# Section heading patterns mapped to the tier keys above.
SECTION_HEADING_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|\n)\s*(?:work|professional|employment|career).*?(?:experience|history)\s*(?:[:|\-]|\n|$)", "work_experience"),
    (r"(?:^|\n)\s*(?:projects?|personal projects|side projects?)\s*(?:[:|\-]|\n|$)", "projects"),
    (r"(?:^|\n)\s*(?:technical\s+)?skills?\s*(?:[:|\-]|\n|$)", "skills"),
    (r"(?:^|\n)\s*(?:certifications?|licenses?|accreditations?)\s*(?:[:|\-]|\n|$)", "certifications"),
    (r"(?:^|\n)\s*(?:education|academic|coursework|qualifications?|degrees?)\s*(?:[:|\-]|\n|$)", "education"),
]


def _get_section_intervals(text: str) -> list[tuple[int, int, float, str]]:
    """Return (start, end, weight, section_key) intervals for each section body."""
    lower = text.lower()
    matches: list[tuple[int, int, float, str]] = []
    for pattern, key in SECTION_HEADING_PATTERNS:
        for match in re.finditer(pattern, lower, re.IGNORECASE):
            matches.append((match.start(), match.end(), SECTION_TIER_WEIGHTS[key], key))
    if not matches:
        return [(0, len(lower), SECTION_TIER_WEIGHTS["work_experience"], "work_experience")]
    matches.sort(key=lambda x: x[0])
    intervals: list[tuple[int, int, float, str]] = []
    if matches[0][0] > 0:
        intervals.append((0, matches[0][0], SECTION_TIER_WEIGHTS["work_experience"], "work_experience"))
    for i, (start, end, weight, key) in enumerate(matches):
        body_start = end
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(lower)
        if body_start < body_end:
            intervals.append((body_start, body_end, weight, key))
    return intervals


def _section_weighted_term_score(text: str, terms: set[str]) -> float:
    """Score query term presence weighted by the resume section in which it appears.

    For each query term the highest-weighted section containing the term is
    used. The final score is the average of those best weights, capped at 1.0.
    """
    if not text or not terms:
        return 0.0
    intervals = _get_section_intervals(text)
    lower = text.lower()
    term_weights: dict[str, float] = {}
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        for match in pattern.finditer(lower):
            pos = match.start()
            for start, end, weight, _ in intervals:
                if start <= pos < end:
                    if weight > term_weights.get(term, 0.0):
                        term_weights[term] = weight
                    break
    if not term_weights:
        return 0.0
    return min(1.0, sum(term_weights.values()) / len(terms))


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
        cache_start = time.perf_counter()
        self._resume_cache = _load_resume_cache()
        self.metadata_load_time = time.perf_counter() - cache_start
        self._summary_cache: dict[str, str] = {}
        self._ai_summary_time: float = 0.0
        self._metadata_scoring_time: float = 0.0
        self._resume_index = _RESUME_INDEX
        self._search_context: dict[str, Any] = {}
        self.query_parser = get_query_parser()
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
        self._ai_summary_time = 0.0
        self._metadata_scoring_time = 0.0

        # Parse the query and enrich filters with extracted intent.
        parse_start = time.perf_counter()
        parsed = self.query_parser.parse(query)
        parse_time = time.perf_counter() - parse_start
        self.last_parsed_query = parsed
        fd = filters.model_dump()
        if parsed.role and not fd.get("role"):
            fd["role"] = parsed.role
        if parsed.industry and not fd.get("industry"):
            fd["industry"] = parsed.industry
        if parsed.skills:
            existing = set(fd.get("skills") or [])
            existing.update(s.lower() for s in parsed.skills)
            if parsed.expanded_terms:
                existing.update(t.lower() for t in parsed.expanded_terms)
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

        # Build an expanded retrieval query for dense/sparse search and metadata
        # scoring. The original user query is preserved for display/explanation.
        expanded_terms = parsed.expanded_terms or []
        expanded_query = query
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"

        # Precompute per-search query context once instead of inside every _score_resume loop.
        raw_terms = expanded_query.lower().split()
        query_terms = {t for t in raw_terms if t.isalnum() and len(t) > 2}
        domain_terms = {t for t in query_terms if t in QUERY_DOMAINS}
        skill_terms = query_terms - domain_terms
        wanted_skills = _tokenize_skills(filters.skills) or skill_terms
        self._search_context = {
            "query": query,
            "expanded_query": expanded_query,
            "raw_terms": raw_terms,
            "query_terms": query_terms,
            "domain_terms": domain_terms,
            "skill_terms": skill_terms,
            "wanted_skills": wanted_skills,
            "filter_role_lower": (filters.role or "").lower(),
            "occupation_query": (filters.role or query).lower(),
            "min_y": filters.experience_min or 0.0,
            "max_y": filters.experience_max or float("inf"),
        }

        # Retrieval pipeline configuration:
        #   dense top 50 + sparse top 50 → RRF fuse → rerank top 20 → return top 10
        FUSED_POOL = 50
        RERANK_POOL = 20

        # 1. Hybrid retrieval: dense top 50 + sparse top 50 fused with RRF
        # Query-parsed role/skills are semantically soft, so they are enforced
        # during _score_resume, not as exact Qdrant filters that can choke dense.
        retrieval_filter_data = filters.model_dump(exclude_none=True)
        for soft_key in ("role", "skills", "industry", "certifications", "source_dataset"):
            retrieval_filter_data.pop(soft_key, None)
        retrieval_filters = retrieval_filter_data or None

        hybrid_results = []
        if self.hybrid_service is not None:
            for attempt in range(3):
                try:
                    hybrid_results = self.hybrid_service.search(
                        expanded_query, top_k=FUSED_POOL, filters=retrieval_filters
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
        for resume, hybrid in pairs[:RERANK_POOL]:
            result = self._score_resume(resume, hybrid, filters, query)
            if result is None:
                continue
            scored.append(result)
        scoring_time = time.perf_counter() - scoring_start

        # 4. Strict filtering (optional) before ranking
        if filters.strict:
            scored = [r for r in scored if r.metadata_score == 1.0]

        # Reject candidates from unrelated professions unless no better matches exist.
        # Role Similarity is a 0-1 score derived from the 0-100 recruiter scale.
        if scored:
            strong_role = [r for r in scored if r.score_breakdown.get("role", 0.0) >= 0.30]
            if strong_role:
                scored = strong_role

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
                "query_parse_ms": parse_time * 1000,
                "scoring_ms": scoring_time * 1000,
                "metadata_scoring_ms": self._metadata_scoring_time * 1000,
                "summary_ms": self._ai_summary_time * 1000,
                "latency_ms": (time.perf_counter() - search_start) * 1000,
                "embedding_time_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
                "dense_retrieval_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
                "sparse_retrieval_ms": hybrid_metrics.get("sparse_latency", 0.0) * 1000,
                "fusion_ms": hybrid_metrics.get("fusion_latency", 0.0) * 1000,
                "retrieval_time_ms": (hybrid_metrics.get("sparse_latency", 0.0) + hybrid_metrics.get("fusion_latency", 0.0)) * 1000,
                "rerank_time_ms": 0.0,
                "generation_time_ms": scoring_time * 1000,
            }
            final_results = self._rank_enhanced(scored, top_k)
            _print_retrieval_quality(final_results, query)
            return final_results

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

        # 7. Final ranking: use the recruiter-oriented overall match.
        # The cross-encoder reranker, if enabled, may provide a small bonus.
        for r in rerank_pool:
            final = r.score_breakdown.get("overall", 0.0)
            if r.rerank_score > 0:
                final = round(min(1.0, final + r.rerank_score * 0.05), 4)
            r.final_score = final
            r.score_breakdown["cross_encoder"] = round(r.rerank_score, 4)
            r.score_breakdown["final"] = r.final_score

        rerank_pool.sort(key=lambda x: x.final_score, reverse=True)
        final = rerank_pool[:top_k]
        _print_retrieval_quality(final, query)

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
            "query_parse_ms": parse_time * 1000,
            "scoring_ms": scoring_time * 1000,
            "metadata_scoring_ms": (scoring_time - self._ai_summary_time) * 1000,
            "summary_ms": self._ai_summary_time * 1000,
            "latency_ms": (time.perf_counter() - search_start) * 1000,
            "embedding_time_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
            "dense_retrieval_ms": hybrid_metrics.get("dense_latency", 0.0) * 1000,
            "sparse_retrieval_ms": hybrid_metrics.get("sparse_latency", 0.0) * 1000,
            "fusion_ms": hybrid_metrics.get("fusion_latency", 0.0) * 1000,
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
            meta = r.resume_metadata
            audit = {
                "rank": rank_i,
                "candidate_id": meta.resume_id,
                "candidate_name": get_display_name(meta, r.source_filename, r.resume_text),
                "display_title": meta.role or "",
                "primary_occupation": meta.primary_role or meta.role or "",
                "role_family": meta.role_family or "",
                "industry": meta.role_family or "",
                "experience_years": meta.experience_years,
                "matched_skills": r.matched_skills,
                "matched_experience": r.score_breakdown.get("matched_experience", []),
                "matched_projects": r.matched_projects,
                "matched_education": r.score_breakdown.get("matched_education", []),
                "semantic_score": r.score_breakdown.get("semantic", 0.0),
                "dense_score": r.score_breakdown.get("dense", 0.0),
                "sparse_score": r.score_breakdown.get("sparse", 0.0),
                "cross_encoder_score": r.score_breakdown.get("cross_encoder", 0.0),
                "metadata_score": r.metadata_score,
                "final_score": r.final_score,
                "role_score": r.score_breakdown.get("role", 0.0),
                "skill_score": r.score_breakdown.get("skill", 0.0),
                "industry_score": r.score_breakdown.get("industry", 0.0),
                "experience_score": r.score_breakdown.get("experience", 0.0),
                "education_score": r.score_breakdown.get("education", 0.0),
                "overall_match": r.score_breakdown.get("overall", 0.0),
            }
            logger.info("Ranking audit | %s", json.dumps(audit, ensure_ascii=False, default=str))
            reason = " | ".join(r.explanation) if r.explanation else "No strong match signals"
            print(
                f"[{audit['rank']}] Candidate: {audit['candidate_name']} | "
                f"Primary Role: {audit['primary_occupation']} | "
                f"Role: {audit['role_score']:.2f} | "
                f"Skill: {audit['skill_score']:.2f} | "
                f"Experience: {audit['experience_score']:.2f} | "
                f"Semantic: {audit['semantic_score']:.2f} | "
                f"Final: {audit['final_score']:.2f} | "
                f"Reason: {reason}"
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

        # Use the _score_resume final (already includes role gate + metadata boost).
        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[:top_k]

        # Print score breakdown
        print("\n" + "=" * 70)
        print("🎯 Candidate Score Breakdown (Enhanced Ranking)")
        print("=" * 70)
        for i, r in enumerate(top, start=1):
            meta = r.resume_metadata
            name = get_display_name(meta, r.source_filename, r.resume_text)
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
        retrieved_chunks: list[dict[str, Any]],
        matched_skills: list[str],
    ) -> str:
        """Generate and cache a concise recruiter-friendly summary from retrieved text."""
        m = resume.resume_metadata
        key = m.resume_id
        if key in _PRECOMPUTED_SUMMARIES:
            return _PRECOMPUTED_SUMMARIES[key]

        if key in self._summary_cache:
            return self._summary_cache[key]

        summary_start = time.perf_counter()
        summary = generate_resume_summary(
            resume_text=resume.resume_text or "",
            matched_text=matched_text,
            retrieved_chunks=retrieved_chunks,
            primary_role=m.primary_role or m.role,
            role_family=m.role_family,
            experience_years=m.experience_years,
            education=m.education or [],
            skills=m.skills or [],
            matched_skills=matched_skills,
            summary=resume.summary or m.summary,
            projects=m.projects or [],
            certifications=m.certifications or [],
        )
        self._ai_summary_time += time.perf_counter() - summary_start
        self._summary_cache[key] = summary
        return summary

    def _score_metadata(
        self,
        m: ResumeMetadata,
        filters: SearchFilters,
    ) -> tuple[float, list[str], list[str], list[str]]:
        """Compute the metadata filter score using only precomputed metadata.

        This method never reads resume.resume_text.
        """
        _metadata_start = time.perf_counter()
        score_sum = 0.0
        total_weight = 0.0
        matched_fields: list[str] = []
        edu_strings: list[str] = []
        cert_strings: list[str] = []

        if filters.role:
            total_weight += FIELD_WEIGHTS["role"]
            role_text = (m.primary_role or m.role or "").lower()
            filter_text = (filters.role or "").lower()
            if role_text in filter_text or filter_text in role_text:
                score_sum += FIELD_WEIGHTS["role"]
                matched_fields.append("role")

        if filters.location:
            total_weight += FIELD_WEIGHTS["location"]
            if filters.location.lower() in (m.location or "").lower():
                score_sum += FIELD_WEIGHTS["location"]
                matched_fields.append("location")

        if filters.experience_min is not None or filters.experience_max is not None:
            total_weight += FIELD_WEIGHTS["experience"]
            years = m.experience_years or 0.0
            min_y = filters.experience_min or 0.0
            max_y = filters.experience_max or float("inf")
            if min_y <= years <= max_y:
                score_sum += FIELD_WEIGHTS["experience"]
                matched_fields.append("experience")

        if filters.skills:
            total_weight += FIELD_WEIGHTS["skills"]
            idx = self._resume_index.get(m.resume_id, {})
            resume_skills = idx.get("skills_tokens", _tokenize_skills(m.skills))
            wanted = (self._search_context or {}).get("wanted_skills") or _tokenize_skills(filters.skills)
            matched = resume_skills & wanted
            if matched:
                score_sum += FIELD_WEIGHTS["skills"] * (len(matched) / len(wanted))
                matched_fields.append("skills")

        edu_strings = [_fmt_education(e) for e in (m.education or []) if _fmt_education(e)]
        if filters.education:
            total_weight += FIELD_WEIGHTS["education"]
            if any(filters.education.lower() in s.lower() for s in edu_strings):
                score_sum += FIELD_WEIGHTS["education"]
                matched_fields.append("education")

        cert_strings = [_fmt_certification(c) for c in (m.certifications or []) if _fmt_certification(c)]
        if filters.certifications:
            total_weight += FIELD_WEIGHTS["certifications"]
            if any(filters.certifications.lower() in s.lower() for s in cert_strings):
                score_sum += FIELD_WEIGHTS["certifications"]
                matched_fields.append("certifications")

        if filters.source_dataset:
            total_weight += FIELD_WEIGHTS["source_dataset"]
            if m.source_dataset == filters.source_dataset:
                score_sum += FIELD_WEIGHTS["source_dataset"]
                matched_fields.append("source_dataset")

        metadata_score = score_sum / total_weight if total_weight > 0 else 1.0
        self._metadata_scoring_time += time.perf_counter() - _metadata_start
        return metadata_score, matched_fields, edu_strings, cert_strings

    def _score_resume(
        self,
        resume: ResumeDocument,
        hybrid: Any | None,
        filters: SearchFilters,
        query: str,
    ) -> SearchResult | None:
        """Compute all scores for a single resume using a weighted Overall Match."""
        _score_resume_start = time.perf_counter()
        resume_id = resume.resume_metadata.resume_id
        m = resume.resume_metadata
        ctx = self._search_context or {}
        idx = self._resume_index.get(resume_id, {})
        meta_text = idx.get("search_text", "")
        meta_text_tokens = idx.get("search_text_tokens", set())
        skills_tokens = idx.get("skills_tokens", set())
        raw_terms = ctx.get("raw_terms") or query.lower().split()
        query_terms = ctx.get("query_terms") or {t for t in raw_terms if t.isalnum() and len(t) > 2}
        domain_terms = ctx.get("domain_terms") or {t for t in query_terms if t in QUERY_DOMAINS}
        skill_terms = ctx.get("skill_terms") or (query_terms - domain_terms)

        # --- metadata filter score (precomputed metadata only) ---
        _step = _StepTimer("metadata_scoring", resume_id, extra="pure metadata")
        metadata_score, matched_fields, edu_strings, cert_strings = self._score_metadata(resume.resume_metadata, filters)

        _step.stop()

        # --- vector scores (retrieval) ---
        _step = _StepTimer("vector_scores", resume_id, extra=f"chunks={len(hybrid.matched_chunks) if hybrid else 0}")
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

        _step.stop()

        # --- skill matching weighted by section (work > projects > skills > certs > education) ---
        _step = _StepTimer("skill_scoring", resume_id, extra=f"meta_text={len(meta_text)} chars")
        matched_skills: list[str] = []
        wanted_skills = ctx.get("wanted_skills") or _tokenize_skills(filters.skills) or skill_terms
        skill_match_available = bool(wanted_skills)
        if skill_match_available:
            resume_skills = skills_tokens if skills_tokens else (_tokenize_skills(resume.skills) if resume.skills else set())
            matched_skills = sorted((resume_skills | meta_text_tokens) & wanted_skills)
            # Score skill mentions against precomputed metadata text, never raw resume text.
            skill_score = _section_weighted_term_score(meta_text, wanted_skills)
        else:
            skill_score = 0.0

        skill_display = "N/A" if not skill_match_available else f"{round(skill_score * 100, 2)}%"

        _step.stop()

        # --- role match (occupation-aware similarity; contributes 40% of final) ---
        _step = _StepTimer("role_scoring", resume_id)
        # Primary occupation is extracted from work history; the raw dataset
        # category is intentionally not used as a substitute.
        primary_role = resume.primary_role or ""
        filter_role = ctx.get("filter_role_lower") or (filters.role or "").lower()
        role_text = (primary_role or "").lower()
        occupation_query = ctx.get("occupation_query") or (filters.role or query).lower()

        if filter_role and filter_role in role_text:
            # Explicit filter role is present in the candidate's primary role.
            role_similarity = 100
        else:
            # Deterministic, embedding-free occupation similarity scoring.
            role_similarity = _recruiter_role_score(occupation_query, primary_role)
        role_score = role_similarity / 100.0

        _step.stop()

        # --- industry match (domain terms in resume text, weighted by section) ---
        _step = _StepTimer("industry_scoring", resume_id, extra=f"text={len(meta_text)} chars")
        resume_text = meta_text
        summary_text = idx.get("summary_text", "")
        searchable_text = f"{resume_text} {summary_text}"
        # Weight domain mentions by section: work > projects > skills > certs > education.
        industry_score = _section_weighted_term_score(searchable_text, domain_terms)
        matched_industry_terms = sorted(
            {d for d in domain_terms if d in searchable_text or _stem_term(d) in searchable_text}
        )

        _step.stop()

        # --- experience match (query terms in resume text, weighted by section) ---
        _step = _StepTimer("experience_scoring", resume_id, extra=f"text={len(meta_text)} chars")
        min_y = ctx.get("min_y") if "min_y" in ctx else (filters.experience_min or 0.0)
        max_y = ctx.get("max_y") if "max_y" in ctx else (filters.experience_max or float("inf"))
        if filters.experience_min is not None or filters.experience_max is not None:
            years = idx.get("experience_years", resume.experience_years or 0.0)
            exp_filter_match = min_y <= years <= max_y
        else:
            exp_filter_match = False

        if exp_filter_match:
            experience_score = 1.0
        else:
            experience_score = _section_weighted_term_score(meta_text, query_terms)

        _step.stop()

        # --- project match (query terms in project descriptions) ---
        _step = _StepTimer("project_scoring", resume_id, extra=f"projects={len(resume.projects or [])}")
        project_terms = query_terms | wanted_skills | domain_terms
        project_text = idx.get("project_text", " ".join(resume.projects or []).lower())
        project_score = _section_weighted_term_score(project_text, project_terms)

        _step.stop()

        # --- education match (query terms in education entries) ---
        _step = _StepTimer("education_scoring", resume_id, extra=f"edu_entries={len(edu_strings)}")
        edu_hits = {
            t for t in query_terms
            if any(t in s.lower() for s in edu_strings)
        }
        matched_education_entries = [
            edu
            for edu in edu_strings
            if any(term in edu.lower() for term in query_terms)
            or any(dt in edu.lower() for dt in domain_terms)
        ]
        education_score = (
            len(edu_hits) / len(query_terms) if query_terms and edu_strings else 0.0
        )

        _step.stop()

        # --- location match ---
        _step = _StepTimer("location_scoring", resume_id)
        location_text = (resume.location or "").lower()
        location_score = 1.0 if (
            (resume.location and any(t in location_text for t in query_terms))
            or (filters.location and filters.location.lower() in location_text)
        ) else 0.0

        _step.stop()

        # --- keyword match (query term presence in resume text / matched text) ---
        _step = _StepTimer("keyword_scoring", resume_id)
        keyword_hits = {t for t in query_terms if t in searchable_text or t in (matched_text or "").lower()}
        keyword_score = len(keyword_hits) / len(query_terms) if query_terms else 0.0

        _step.stop()

        # --- semantic similarity from dense retrieval ---
        _step = _StepTimer("semantic_similarity", resume_id)
        semantic_similarity = min(1.0, max(0.0, semantic_score))

        _step.stop()

        # --- Suitability Score (role-aware candidate fit, 0-1) ---
        _step = _StepTimer("suitability_score", resume_id)
        component_scores = {
            "role": role_score,
            "semantic": semantic_similarity,
            "skill": skill_score,
            "experience": experience_score,
            "industry": industry_score,
            "education": education_score,
        }

        # All six components always apply; missing evidence contributes 0.0.
        overall_match = round(
            min(1.0, sum(component_scores[k] * SUITABILITY_SCORE_WEIGHTS[k] for k in SUITABILITY_SCORE_WEIGHTS)),
            4,
        )

        _step.stop()

        # --- occupation gate ---
        _step = _StepTimer("occupation_gate", resume_id)
        # Strongly penalize unrelated/different occupations; "Related" (>=0.6) is accepted.
        if role_score < 0.6:
            overall_match = round(overall_match * (role_score / 0.6), 4)

        # --- metadata boost (adjusts retrieval, never replaces it) ---
        metadata_boost = 0.3 + 0.7 * metadata_score
        overall_match = round(min(1.0, overall_match * metadata_boost), 4)

        _step.stop()

        # --- explainability reasons (uses only real evidence) ---
        _step = _StepTimer("explainability", resume_id)
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

        _step.stop()

        # --- top retrieved chunks for the drawer ---
        _step = _StepTimer("retrieved_chunks", resume_id, extra=f"chunks={len(hybrid.matched_chunks) if hybrid else 0}")
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

        _step.stop()

        # --- derived match sections for the UI ---
        _step = _StepTimer("ui_evidence", resume_id, extra=f"experience={len(resume.resume_metadata.experience or [])}")
        query_terms_lower = {t for t in query_terms}
        matched_projects = [p for p in (resume.projects or []) if any(t in p.lower() for t in query_terms_lower)][:3]
        matched_certifications = [c for c in (resume.certifications or []) if any(t in c.lower() for t in query_terms_lower)][:3]

        # Work history is stored inside the canonical ResumeMetadata.
        work_history = getattr(resume.resume_metadata, "experience", None) or []
        matched_experience: list[str] = []
        for exp in work_history:
            raw_title = (
                exp.get("title") if isinstance(exp, dict) else getattr(exp, "title", None)
            )
            exp_title = (raw_title or "").lower()
            if not exp_title:
                continue
            for t in query_terms_lower:
                if t in exp_title or (filters.role and filters.role.lower() in exp_title) or occupation_query in exp_title:
                    matched_experience.append(str(raw_title))
                    break
        matched_experience = matched_experience[:5]

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

        _step.stop()

        # Resume preview (metadata only)
        _step = _StepTimer("preview_generation", resume_id, extra=f"summary_len={len(m.summary or '')}")
        preview = _PRECOMPUTED_SUMMARIES.get(resume_id) or (m.summary or "")[:300]

        _step.stop()

        # AI summary from retrieved resume content (cached per resume)
        _step = _StepTimer("summary_generation", resume_id, extra="generate_resume_summary")
        ai_summary = self._get_ai_summary(
            resume=resume,
            matched_text=matched_text,
            retrieved_chunks=retrieved_chunks,
            matched_skills=matched_skills,
        )

        _step.stop()

        # --- skill section evidence for explainability ---
        _step = _StepTimer("skill_evidence", resume_id, extra=f"skills={len(matched_skills)} text={len(meta_text)} chars")
        skill_evidence: dict[str, list[str]] = {}
        resume_full_text = meta_text
        if resume_full_text and matched_skills:
            intervals = _get_section_intervals(resume_full_text)
            text_lower = resume_full_text.lower()
            resume_skills_lower = {s.lower().strip() for s in (resume.skills or [])}
            for sk in matched_skills:
                pattern = re.compile(rf"\b{re.escape(sk.lower())}\b")
                found_keys: set[str] = set()
                for match in pattern.finditer(text_lower):
                    pos = match.start()
                    for start, end, _, key in intervals:
                        if start <= pos < end:
                            found_keys.add(key)
                            break
                if not found_keys and sk in resume_skills_lower:
                    found_keys.add("skills")
                sorted_keys = sorted(found_keys, key=lambda k: SECTION_TIER_WEIGHTS.get(k, 0.0), reverse=True)
                skill_evidence[sk] = [SECTION_DISPLAY.get(k, k.title()) for k in sorted_keys]

        _step.stop()

        score_breakdown = {
            "overall": overall_match,
            "suitability_score_0_100": round(overall_match * 100, 2),
            "role": round(role_score, 4),
            "semantic": round(semantic_similarity, 4),
            "skill": round(skill_score, 4),
            "experience": round(experience_score, 4),
            "industry": round(industry_score, 4),
            "education": round(education_score, 4),
            "project": round(project_score, 4),
            "skill_evidence": skill_evidence,
            "location": round(location_score, 4),
            "keyword": round(keyword_score, 4),
            "dense": round(semantic_score, 4),
            "sparse": round(bm25_score, 4),
            "query_role": occupation_query,
            "raw_scores": {k: round(v, 4) for k, v in component_scores.items()},
            "denominator": 1.0,
            "matched_industry": matched_industry_terms,
            "matched_education": matched_education_entries,
            "matched_experience": matched_experience,
            "overall": overall_match,
        }

        _total_ms = (time.perf_counter() - _score_resume_start) * 1000
        print(f"[PROFILE _score_resume] resume_id={resume_id} step=_score_resume_total ms={_total_ms:.2f}")

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


