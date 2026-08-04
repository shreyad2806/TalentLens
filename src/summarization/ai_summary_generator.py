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
    primary_role: str | None,
    role_family: str | None,
    experience_years: float | None,
    education: list[str] | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
    summary: str | None,
    projects: list[str] | None,
    certifications: list[str] | None,
) -> str:
    """Build a strict recruiter prompt grounded in retrieved resume content."""
    snippets: list[str] = []
    for chunk in (retrieved_chunks or []):
        text = chunk.get("text") or chunk.get("matched_text") or ""
        if text and len(text.strip()) >= 10:
            snippets.append(text.strip()[:350])
        if len(snippets) >= 3:
            break

    header_parts: list[str] = []
    if primary_role or role_family:
        header_parts.append(f"Role: {primary_role or role_family}")
    if experience_years is not None:
        header_parts.append(f"Years: {experience_years:g}")
    if role_family:
        header_parts.append(f"Domain: {role_family}")
    techs = [s for s in (matched_skills or skills or []) if s][:6]
    if techs:
        header_parts.append(f"Technologies: {', '.join(techs)}")
    if projects:
        header_parts.append(f"Projects: {', '.join([p for p in projects if p][:3])}")
    if certifications:
        header_parts.append(f"Certifications: {', '.join([c for c in certifications if c][:3])}")
    if summary:
        clean_summary = re.sub(r"\s+", " ", summary).strip()[:250]
        header_parts.append(f"Profile: {clean_summary}")

    context = "\n\n".join(
        [f"Resume excerpt {i+1}:\n{s}" for i, s in enumerate(snippets)]
    )
    header = " | ".join(header_parts) if header_parts else "Candidate profile"

    return (
        "You are a technical recruiter. Write exactly 3 concise, natural sentences "
        "about this candidate using ONLY the retrieved resume excerpts and metadata below.\n\n"
        "Structure:\n"
        "- Sentence 1: Primary role + years of experience.\n"
        "- Sentence 2: Primary technologies and technical strengths.\n"
        "- Sentence 3: Domain expertise, notable achievements or specialization.\n\n"
        "Rules:\n"
        "- Maximum 3 sentences, each under 35 words.\n"
        "- Do NOT start with 'Experienced in...' or 'Experienced with...'.\n"
        "- Do NOT use the phrases 'Top technologies include...' or 'Domain expertise...'.\n"
        "- Prioritize work experience, projects and skills. Mention education only if it is highly relevant.\n"
        "- Ground every statement in the provided resume content.\n"
        "- If information is missing, produce a shorter summary instead of inventing details.\n"
        "- Use natural recruiter language, not bullet-style labels.\n\n"
        f"{header}\n\n"
        f"{context}\n\n"
        "Summary:"
    )


def _fallback_summary(
    primary_role: str | None,
    role_family: str | None,
    experience_years: float | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
    education: list[str] | None = None,
    summary: str | None = None,
    projects: list[str] | None = None,
    certifications: list[str] | None = None,
) -> str:
    """Build a deterministic 3-sentence recruiter summary from metadata only.

    Structure:
        Sentence 1: role + years of experience
        Sentence 2: primary technologies and technical strengths
        Sentence 3: domain expertise, notable achievements or specialization
    """
    sentences: list[str] = []
    display_role = primary_role or role_family
    key_domain = role_family or display_role

    # Sentence 1: role + years of experience
    if display_role and experience_years is not None and experience_years > 0:
        sentences.append(f"{display_role} with {experience_years:g} years of experience.")
    elif display_role:
        sentences.append(f"{display_role} with relevant experience.")
    elif experience_years is not None and experience_years > 0:
        sentences.append(f"Professional with {experience_years:g} years of experience.")
    else:
        sentences.append("Candidate profile.")

    # Sentence 2: primary technologies and technical strengths
    tech_skills = [s for s in (matched_skills or []) if s][:6]
    if not tech_skills and skills:
        tech_skills = [s for s in skills if s][:6]
    if tech_skills:
        if len(tech_skills) == 1:
            tech_clause = tech_skills[0]
        else:
            tech_clause = ", ".join(tech_skills[:-1]) + f" and {tech_skills[-1]}"
        extras: list[str] = []
        if projects:
            extras.extend([p for p in projects if p][:2])
        if certifications:
            extras.extend([c for c in certifications if c][:1])
        if extras:
            sentences.append(f"Strong background in {tech_clause}, with experience in {' and '.join(extras)}.")
        else:
            sentences.append(f"Strong background in {tech_clause}.")

    # Sentence 3: domain expertise, notable achievements or specialization
    if summary:
        clean = re.sub(r"\s+", " ", summary).strip()
        if clean:
            if len(clean) > 120:
                clean = clean[:117] + "..."
            sentences.append(f"Specialized in {clean}.")
    elif projects:
        projects_list = [p for p in projects if p][:2]
        if projects_list:
            sentences.append(f"Notable projects include {' and '.join(projects_list)}.")
    elif certifications:
        certs_list = [c for c in certifications if c][:2]
        if certs_list:
            sentences.append(f"Certified in {' and '.join(certs_list)}.")
    elif key_domain:
        sentences.append(f"Domain focus: {key_domain}.")
    elif education and education[0]:
        sentences.append(f"Background includes {str(education[0]).strip()}.")
    else:
        sentences.append("Relevant background for this search.")

    final = " ".join(sentences)
    if len(final) > 280:
        final = final[:277] + "..."
    return final


def generate_resume_summary(
    resume_text: str,
    matched_text: str,
    retrieved_chunks: list[dict[str, Any]],
    primary_role: str | None,
    role_family: str | None,
    experience_years: float | None,
    education: list[str] | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
    summary: str | None = None,
    projects: list[str] | None = None,
    certifications: list[str] | None = None,
) -> str:
    """Generate a concise, recruiter-friendly summary from retrieved resume chunks.

    First tries OpenAI if configured, then falls back to a deterministic summary
    built only from the supplied metadata and retrieved chunks.
    """
    start_time = time.perf_counter()
    prompt = _build_summary_prompt(
        retrieved_chunks, primary_role, role_family, experience_years, education, skills, matched_skills,
        summary, projects, certifications,
    )
    answer = _call_openai(prompt)
    if answer:
        word_count = len(answer.split())
        sentence_count = answer.count(".") + answer.count("?") + answer.count("!")
        is_format_ok = (
            1 <= sentence_count <= 3
            and word_count <= 120
            and "experienced in" not in answer.lower()
            and "experienced with" not in answer.lower()
            and "top technologies include" not in answer.lower()
            and "domain expertise" not in answer.lower()
        )
        if is_format_ok:
            logger.info("LLM summary generated in %.3fs (%d words)", time.perf_counter() - start_time, word_count)
            return answer
    summary = _fallback_summary(
        primary_role, role_family, experience_years, skills, matched_skills, education, summary, projects, certifications,
    )
    logger.info("Deterministic summary generated in %.3fs", time.perf_counter() - start_time)
    return summary
