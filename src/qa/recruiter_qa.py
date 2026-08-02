"""Upgraded Recruiter QA: structured query parsing + cited, no-hallucination answers."""

from __future__ import annotations

import re
import time
from typing import Any

from src.search import SearchFilters, SearchResult, SearchService

# Common technical and business skills for query extraction.
SKILL_KEYWORDS = [
    "python", "sql", "aws", "docker", "kubernetes", "react", "node", "java",
    "machine learning", "ai", "rag", "llm", "postgresql", "mongodb", "git",
    "ci/cd", "flask", "django", "spark", "hadoop", "excel", "power bi",
    "tableau", "tensorflow", "pytorch", "c++", "javascript", "typescript",
    "angular", "vue", "spring", "rest api", "graphql", "microservices",
    "linux", "azure", "gcp", "salesforce", "jira", "html", "css",
    "fastapi", "next.js", "redis", "mongodb", "postgresql", "oracle", "scala",
    "go", "rust", "php", "swift", "kotlin", "c#", "matlab", "sas",
    "spss", "stata", "r", "looker", "qlik", "d3.js", "spark", "airflow",
    "kafka", "rabbitmq", "jenkins", "gitlab", "github", "terraform",
    "ansible", "puppet", "chef", "nginx", "apache", "tomcat", "express",
    "spring boot", "laravel", "ruby on rails", "django", "flask",
]

# Locations the system can recognize.
LOCATION_KEYWORDS = [
    "bangalore", "mumbai", "delhi", "hyderabad", "pune", "chennai", "kolkata",
    "india", "usa", "uk", "remote", "new york", "california", "texas",
    "florida", "london", "toronto", "vancouver", "sydney", "singapore",
    "dubai", "berlin", "amsterdam", "paris", "dublin", "boston",
    "seattle", "san francisco", "chicago", "austin", "denver",
]

# Pre-defined role titles we can recognize.
ROLE_TITLES = [
    "backend engineer", "frontend engineer", "full stack engineer",
    "software engineer", "data scientist", "data analyst", "business analyst",
    "java developer", "python developer", "devops engineer", "cloud engineer",
    "hr manager", "hr specialist", "teacher", "sales representative",
    "project manager", "product manager", "marketing manager",
    "account manager", "customer support specialist", "technical support",
    "nurse", "doctor", "accountant", "financial analyst", "legal counsel",
    "operations manager", "administrative assistant", "office assistant",
    "mechanical engineer", "electrical engineer", "civil engineer",
]

# Education keywords.
EDUCATION_KEYWORDS = [
    "bachelor", "master", "mba", "phd", "b.tech", "m.tech", "b.e.", "m.e.",
    "b.sc", "m.sc", "b.com", "m.com", "b.a.", "m.a.", "diploma",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _kw_match(kw: str, q: str) -> bool:
    """Match a keyword as a whole word (or phrase, if it contains spaces)."""
    kw = kw.strip()
    if not kw:
        return False
    if " " in kw:
        return kw in q
    return re.search(r"\b" + re.escape(kw) + r"\b", q) is not None


class QueryParser:
    """Parse a free-text recruiter question into structured SearchFilters."""

    def parse(self, query: str) -> SearchFilters:
        q = _normalize(query)

        # skills — use word boundaries to avoid matching "ai" in "mumbai" etc.
        skills = sorted({kw for kw in set(SKILL_KEYWORDS) if _kw_match(kw, q)})

        # location
        location = None
        for loc in sorted(LOCATION_KEYWORDS, key=len, reverse=True):
            if _kw_match(loc, q):
                location = loc.title() if loc.lower() not in ("usa", "uk") else loc.upper()
                break

        # experience
        exp_match = re.search(r"(\d+)\+?\s*years?", q)
        experience_min = float(exp_match.group(1)) if exp_match else None

        # role
        role = None
        for title in sorted(ROLE_TITLES, key=len, reverse=True):
            if title in q:
                role = title.title()
                break

        # education
        education = None
        for edu in sorted(set(EDUCATION_KEYWORDS), key=len, reverse=True):
            # Strip periods for education abbreviations; still require a full word.
            clean_q = q.replace(".", "")
            if _kw_match(edu.replace(".", ""), clean_q):
                education = edu
                break

        return SearchFilters(
            role=role,
            location=location,
            experience_min=experience_min,
            skills=skills,
            education=education,
        )


class RecruiterAnswerGenerator:
    """Deterministic, citation-only answer generator for recruiter QA."""

    def generate(self, query: str, results: list[SearchResult]) -> dict[str, Any]:
        start = time.perf_counter()

        if not results:
            answer = "Not found in indexed resumes."
            return {
                "answer": answer,
                "prompt": query,
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(answer.split()),
                "response_time_ms": (time.perf_counter() - start) * 1000,
            }

        lines = [
            f"Based on the indexed resumes, here are the matches for: '{query}'\n"
        ]

        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. **{r.candidate_name or 'Unknown'}** (Resume ID: {r.resume_id})")
            lines.append(f"   - Role: {r.role or 'Not specified'}")
            if r.matched_sections:
                lines.append(f"   - Matched Sections: {', '.join(r.matched_sections)}")
            if r.matched_skills:
                lines.append(f"   - Matched Skills: {', '.join(r.matched_skills)}")
            if r.matched_projects:
                lines.append(f"   - Matched Projects: {', '.join(r.matched_projects)}")
            if r.matched_certifications:
                lines.append(f"   - Matched Certifications: {', '.join(r.matched_certifications)}")
            if r.experience_years is not None and r.experience_years > 0:
                lines.append(f"   - Matched Experience: {r.experience_years:g} years")
            if r.projects:
                lines.append(f"   - Projects: {', '.join(r.projects[:3])}")
            if r.education:
                lines.append(f"   - Matched Education: {', '.join(r.education[:3])}")
            lines.append(f"   - Matched Resume Section: {(r.matched_sections[0] if r.matched_sections else 'unknown')}")
            if r.matched_text:
                lines.append(f"   - Evidence: {r.matched_text[:200]}")
            lines.append(
                f"   - Scores: dense={r.dense_score:.4f}, "
                f"bm25={r.bm25_score:.4f}, rrf={r.rrf_score:.4f}, "
                f"final={r.final_score:.4f}"
            )
            lines.append("")

        answer = "\n".join(lines).strip()

        return {
            "answer": answer,
            "prompt": query,
            "prompt_tokens": len(query.split()),
            "completion_tokens": len(answer.split()),
            "response_time_ms": (time.perf_counter() - start) * 1000,
        }


class RecruiterQA:
    """
    End-to-end recruiter QA with structured query parsing and cited answers.

    Pipeline:
        recruiter question -> QueryParser -> SearchFilters
        -> SearchService (semantic + metadata) -> RecruiterAnswerGenerator
    """

    def __init__(self, search_service: SearchService | None = None):
        self.parser = QueryParser()
        self.search_service = search_service or SearchService(hybrid_service=None)
        self.answer_generator = RecruiterAnswerGenerator()

    def ask(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Answer a recruiter question with cited evidence."""
        t0 = time.perf_counter()

        filters = self.parser.parse(query)
        search_results = self.search_service.search(query, top_k=top_k, filters=filters)
        llm_result = self.answer_generator.generate(query, search_results)

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "query": query,
            "filters": filters.model_dump(),
            "answer": llm_result["answer"],
            "results": [r.model_dump() for r in search_results],
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "response_time_ms": total_ms,
            "trace": {
                "query_parsing_ms": None,
                "search_ms": total_ms - llm_result["response_time_ms"],
                "answer_ms": llm_result["response_time_ms"],
                "total_ms": total_ms,
            },
        }
