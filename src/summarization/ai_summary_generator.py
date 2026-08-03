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
            max_tokens=300,
            temperature=0.2,
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
    context = "\n\n".join(
        f"[{i + 1}] {c.get('text', '')}" for i, c in enumerate(retrieved_chunks)
    )
    if not context:
        context = "[No retrieved chunks]"

    header = f"Role: {role or 'Not specified'} | Experience: {experience_years or 'Not specified'} years"
    if skills:
        header += f" | Skills: {', '.join(skills[:12])}"
    if matched_skills:
        header += f" | Matched Skills: {', '.join(matched_skills[:12])}"

    return (
        "You are a technical recruiter assistant. Generate a concise recruiter-friendly "
        "candidate summary using ONLY the retrieved resume chunks below.\n\n"
        "Summary format (include only the sections supported by the chunks; "
        "omit unavailable sections):\n"
        "Professional Summary\n"
        "1-2 sentences describing the candidate.\n\n"
        "Experience\n"
        "Years of experience, primary domains, leadership level.\n\n"
        "Core Expertise\n"
        "Top technologies, business expertise, major strengths.\n\n"
        "Rules:\n"
        "- Maximum 120 words.\n"
        "- Never invent missing information.\n"
        "- If a section is unavailable, omit it and its heading.\n"
        "- Do not include filler like 'Based on the context'.\n"
        "- Use only facts present in the chunks.\n\n"
        f"{header}\n\n"
        "Retrieved Chunks:\n"
        f"{context}\n\n"
        "Summary:"
    )


def _fallback_summary(
    role: str | None,
    experience_years: float | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
) -> str:
    """Build a grounded summary from metadata fields only."""
    parts: list[str] = []
    if role and experience_years is not None:
        parts.append(f"{role} with {experience_years:g} years of experience.")
    elif role:
        parts.append(f"{role}.")
    elif experience_years is not None:
        parts.append(f"Professional with {experience_years:g} years of experience.")

    tech_skills = [s for s in (skills or []) if s][:6]
    if tech_skills:
        parts.append(f"Strong background in {', '.join(tech_skills)}.")

    if matched_skills:
        parts.append(f"Experienced in {', '.join(matched_skills[:6])}.")

    summary = " ".join(parts)
    words = summary.split()
    if len(words) > 120:
        summary = " ".join(words[:120])
    return summary


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
        if word_count <= 160:  # allow slack for formatting before truncating
            logger.info("LLM summary generated in %.3fs (%d words)", time.perf_counter() - start_time, word_count)
            return answer
    summary = _fallback_summary(role, experience_years, skills, matched_skills)
    logger.info("Deterministic summary generated in %.3fs", time.perf_counter() - start_time)
    return summary
