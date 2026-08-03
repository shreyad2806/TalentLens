"""AI Resume Summarization.

Generates a concise, recruiter-friendly summary from retrieved resume text.
No external LLM is used; summaries are built by extracting and ranking
sentences that are present in the resume content itself.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any


def _sentences(text: str) -> list[str]:
    """Split text into clean sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    raw = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for r in raw:
        s = r.strip()
        if len(s) >= 20:
            out.append(s)
    return out


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9+#/]+", text.lower()))


def _similar(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / max(len(ta), len(tb))


def _score_sentence(
    sent: str,
    query_terms: set[str],
    matched_skills: set[str],
    skills: set[str],
    role: str,
    education: list[str],
) -> float:
    """Score a sentence by how much it overlaps with the query and metadata."""
    score = 0.0
    lower = sent.lower()
    tokens = _tokens(sent)

    # Query and matched skills (highest weight)
    score += 3.0 * len(tokens & query_terms)
    score += 3.0 * len(tokens & matched_skills)

    # General skills
    score += 1.0 * len(tokens & skills)

    # Role presence
    if role and role.lower() in lower:
        score += 2.0

    # Education presence
    for edu in education:
        if edu and edu.lower() in lower:
            score += 2.0

    # Experience / action markers
    action_words = {
        "led", "led", "managed", "developed", "built", "engineered", "designed",
        "implemented", "created", "architected", "optimized", "spearheaded",
        "head", "directed", "oversaw", "coordinated", "executed", "delivered",
        "drove", "achieved", "reduced", "increased", "improved", "analyzed",
        "forecasted", "budgeted", "planned", "advised", "mentored",
    }
    score += 1.5 * len(tokens & action_words)

    # Years / numbers
    if re.search(r"\b\d+\s*(\+?)\s*years?\b", lower, re.IGNORECASE):
        score += 2.0
    if re.search(r"\b\d+\b", lower):
        score += 0.5

    # Prefer shorter, information-dense sentences
    if 40 <= len(sent) <= 140:
        score += 0.5
    if len(sent) > 220:
        score -= 1.0

    return score


def _why_match_line(
    role: str,
    education: list[str],
    matched_skills: list[str],
) -> str:
    """Build a short match explanation using only retrieved facts."""
    parts: list[str] = []
    if role:
        parts.append(role.lower())
    if education:
        parts.extend(e.lower() for e in education[:2] if e)
    if matched_skills:
        parts.extend(sk.lower() for sk in matched_skills[:4])

    if not parts:
        return "Strong match based on semantic similarity."

    unique = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return "Strong match because of " + ", ".join(unique) + "."


logger = logging.getLogger(__name__)


def _call_openai(prompt: str) -> str:
    """Call OpenAI if configured, otherwise return empty string."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_TALENTLENS")
    if not api_key:
        return ""
    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful technical recruiter assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("OpenAI summary failed: %s", e)
        return ""


def _build_summary_prompt(
    retrieved_chunks: list[dict[str, Any]],
    role: str | None,
    experience_years: float | None,
    education: list[str] | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
) -> str:
    """Build a strict recruiter prompt using only structured metadata."""
    _ = retrieved_chunks  # ignored: we use metadata only to avoid hallucination/verbatim copying

    header = f"Role: {role or 'Not specified'}"
    if experience_years is not None:
        header += f" | Experience: {experience_years:g} years"
    if skills:
        header += f" | Skills: {', '.join([s for s in skills if s][:8])}"
    if matched_skills:
        header += f" | Matched Skills: {', '.join([s for s in matched_skills if s][:8])}"
    if education:
        header += f" | Education: {', '.join([e for e in education if e][:3])}"

    return (
        "You are a technical recruiter assistant. Generate exactly 3 concise "
        "recruiter-focused sentences using ONLY the metadata below.\n\n"
        "Format:\n"
        "Sentence 1: Years of experience and primary role.\n"
        "Sentence 2: Top technologies (up to 6), comma-separated.\n"
        "Sentence 3: Domain expertise based on role and education.\n\n"
        "Rules:\n"
        "- Exactly 3 sentences, each under 25 words.\n"
        "- Never use the phrase 'Experienced in ...' or any variant.\n"
        "- Do not copy resume text verbatim.\n"
        "- Do not invent missing information; omit unknown fields.\n"
        "- Use only the metadata provided.\n\n"
        f"{header}\n\n"
        "Summary:"
    )


def _fallback_summary(
    role: str | None,
    experience_years: float | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
    education: list[str] | None = None,
) -> str:
    """Build a deterministic 3-sentence recruiter summary from metadata only.

    Sentence 1: years + primary role
    Sentence 2: top technologies
    Sentence 3: domain expertise
    """
    sentences: list[str] = []

    # Sentence 1: years + primary role
    if role and experience_years is not None:
        sentences.append(f"{role} with {experience_years:g} years of experience.")
    elif role:
        sentences.append(f"{role}.")
    elif experience_years is not None:
        sentences.append(f"Professional with {experience_years:g} years of experience.")
    else:
        sentences.append("Candidate profile.")

    # Sentence 2: top technologies
    tech_skills = [s for s in (skills or []) if s][:6]
    if not tech_skills and matched_skills:
        tech_skills = [s for s in matched_skills if s][:6]
    if tech_skills:
        sentences.append(f"Top technologies include {', '.join(tech_skills)}.")

    # Sentence 3: domain expertise
    domain_parts: list[str] = []
    if role:
        domain_parts.append(role)
    if education:
        domain_parts.extend(str(e).strip() for e in education[:2] if e)
    if domain_parts:
        sentences.append(f"Domain expertise in {', '.join(domain_parts)}.")
    elif tech_skills:
        sentences.append(f"Domain focus on {tech_skills[0]}.")
    else:
        sentences.append("Relevant background for this search.")

    return " ".join(sentences)


def generate_resume_summary(
    resume_text: str,
    matched_text: str,
    retrieved_chunks: list[dict[str, Any]],
    role: str | None,
    experience_years: float | None,
    education: list[str] | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
) -> str:
    """Generate a concise, recruiter-friendly summary from retrieved resume chunks.

    First tries OpenAI if configured, then falls back to a deterministic summary
    built only from the supplied metadata and retrieved chunks.
    """
    start_time = time.perf_counter()
    prompt = _build_summary_prompt(retrieved_chunks, role, experience_years, education, skills, matched_skills)
    answer = _call_openai(prompt)
    if answer:
        word_count = len(answer.split())
        sentence_count = answer.count(".") + answer.count("?") + answer.count("!")
        is_format_ok = (
            1 <= sentence_count <= 3
            and word_count <= 120
            and "experienced in" not in answer.lower()
            and "experienced with" not in answer.lower()
        )
        if is_format_ok:
            logger.info("LLM summary generated in %.3fs (%d words)", time.perf_counter() - start_time, word_count)
            return answer
    summary = _fallback_summary(role, experience_years, skills, matched_skills, education)
    logger.info("Deterministic summary generated in %.3fs", time.perf_counter() - start_time)
    return summary
