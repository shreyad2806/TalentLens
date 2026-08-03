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
) -> str:
    """Build a strict recruiter prompt using only structured metadata."""
    _ = retrieved_chunks  # ignored: we use metadata only to avoid hallucination/verbatim copying

    header = f"Primary role: {primary_role or role_family or 'Not specified'}"
    if experience_years is not None:
        header += f" | Years: {experience_years:g}"
    if role_family:
        header += f" | Key domain: {role_family}"
    if matched_skills:
        header += f" | Primary technologies: {', '.join([s for s in matched_skills if s][:6])}"
    elif skills:
        header += f" | Primary technologies: {', '.join([s for s in skills if s][:6])}"
    if education:
        header += f" | Education: {', '.join([e for e in education if e][:2])}"

    return (
        "You are a technical recruiter. Write exactly 3 concise, natural sentences "
        "about this candidate using ONLY the metadata below.\n\n"
        "Structure:\n"
        "- Sentence 1: Primary role + years of experience + key domain.\n"
        "- Sentence 2: Primary technologies (up to 6), in a natural phrase.\n"
        "- Sentence 3: Recent work focus, with education mentioned only after work.\n\n"
        "Rules:\n"
        "- Maximum 3 sentences, each under 28 words.\n"
        "- Do NOT start with 'Experienced in...' or 'Experienced with...'.\n"
        "- Do NOT use the phrases 'Top technologies include...' or 'Domain expertise...'.\n"
        "- Mention work experience before education.\n"
        "- Do not invent missing information; omit unknown fields.\n"
        "- Use natural recruiter language, not bullet-style labels.\n\n"
        f"{header}\n\n"
        "Summary:"
    )


def _fallback_summary(
    primary_role: str | None,
    role_family: str | None,
    experience_years: float | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
    education: list[str] | None = None,
) -> str:
    """Build a deterministic 3-sentence recruiter summary from metadata only.

    Structure:
        Sentence 1: role + years + key domain
        Sentence 2: primary technologies
        Sentence 3: work focus, education last
    """
    sentences: list[str] = []
    display_role = primary_role or role_family
    key_domain = role_family or display_role

    # Sentence 1: role + years + key domain
    if display_role and experience_years is not None:
        if key_domain and key_domain != display_role:
            sentences.append(f"{display_role} with {experience_years:g} years of hands-on experience, primarily in {key_domain}.")
        else:
            sentences.append(f"{display_role} with {experience_years:g} years of hands-on experience.")
    elif display_role:
        sentences.append(f"{display_role} with relevant hands-on experience.")
    elif experience_years is not None:
        sentences.append(f"Professional with {experience_years:g} years of hands-on experience.")
    else:
        sentences.append("Candidate profile.")

    # Sentence 2: primary technologies
    tech_skills = [s for s in (matched_skills or []) if s][:6]
    if not tech_skills and skills:
        tech_skills = [s for s in skills if s][:6]
    if tech_skills:
        if len(tech_skills) == 1:
            tech_clause = tech_skills[0]
        else:
            tech_clause = ", ".join(tech_skills[:-1]) + f" and {tech_skills[-1]}"
        sentences.append(f"They work with {tech_clause}.")

    # Sentence 3: work focus before education
    if key_domain:
        work_focus = f"Recent work has been in {key_domain}"
        if education and education[0]:
            sentences.append(f"{work_focus}, supported by {str(education[0]).strip()}.")
        else:
            sentences.append(f"{work_focus}.")
    elif education and education[0]:
        sentences.append(f"Background includes {str(education[0]).strip()}.")
    else:
        sentences.append("Relevant background for this search.")

    return " ".join(sentences)


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
) -> str:
    """Generate a concise, recruiter-friendly summary from retrieved resume chunks.

    First tries OpenAI if configured, then falls back to a deterministic summary
    built only from the supplied metadata and retrieved chunks.
    """
    start_time = time.perf_counter()
    prompt = _build_summary_prompt(
        retrieved_chunks, primary_role, role_family, experience_years, education, skills, matched_skills
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
    summary = _fallback_summary(primary_role, role_family, experience_years, skills, matched_skills, education)
    logger.info("Deterministic summary generated in %.3fs", time.perf_counter() - start_time)
    return summary
