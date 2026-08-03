"""AI Resume Summarization.

Generates a concise, recruiter-friendly summary from retrieved resume text.
No external LLM is used; summaries are built by extracting and ranking
sentences that are present in the resume content itself.
"""

from __future__ import annotations

import re


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


def generate_resume_summary(
    resume_text: str,
    matched_text: str,
    query: str,
    role: str,
    education: list[str] | None,
    skills: list[str] | None,
    matched_skills: list[str] | None,
) -> str:
    """Return at most 3 short paragraphs: Professional, Experience, Key Strengths.

    Only sentences that exist in the retrieved resume content are used.
    """
    source = (resume_text or "").strip()
    if not source and matched_text:
        source = matched_text.strip()
    if not source:
        return "No resume text available for summarization."

    sentences = _sentences(source)
    if not sentences:
        return source[:300]

    # Normalise search terms
    qt = _tokens(query) | set(s.lower() for s in (matched_skills or []))
    ms = set(s.lower() for s in (matched_skills or []))
    sk = set(s.lower() for s in (skills or []))
    edu = [e for e in (education or []) if e and e.strip()]

    # Score and rank sentences
    ranked = [
        (s, _score_sentence(s, qt, ms, sk, role or "", edu))
        for s in sentences
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Pick up to 3 diverse, high-scoring sentences
    selected: list[str] = []
    for s, _ in ranked:
        if len(selected) >= 3:
            break
        if all(_similar(s, existing) < 0.6 for existing in selected):
            selected.append(s)

    # Build three paragraphs
    paragraphs: list[str] = []

    # 1. Professional Summary: role / years / overview
    prof_sents: list[str] = []
    for s in selected:
        if (role and role.lower() in s.lower()) or re.search(r"\b\d+\s*years?\b", s, re.IGNORECASE):
            prof_sents.append(s)
            break
    if not prof_sents and selected:
        prof_sents.append(selected[0])
    paragraphs.append(" ".join(prof_sents[:2]))

    # 2. Experience Summary: action / responsibilities
    exp_sents: list[str] = []
    for s in selected:
        if re.search(
            r"\b(led|managed|developed|built|engineered|designed|implemented|created|architected|optimized|oversaw|coordinated|executed|analyzed|forecasted|budgeted|planned)\b",
            s,
            re.IGNORECASE,
        ):
            exp_sents.append(s)
            break
    if not exp_sents:
        # Pick the next-best non-prof sentence
        for s in selected:
            if s not in prof_sents:
                exp_sents.append(s)
                break
    paragraphs.append(" ".join(exp_sents[:2]))

    # 3. Key Strengths: matched skills + why match
    key_sents: list[str] = []
    for s in selected:
        if any(sk.lower() in s.lower() for sk in (matched_skills or [])):
            key_sents.append(s)
            break
    why = _why_match_line(role or "", edu, matched_skills or [])
    if why and why not in key_sents:
        key_sents.append(why)
    paragraphs.append(" ".join(key_sents[:2]))

    # Remove duplicates while preserving order and trim to 3 paragraphs
    seen: set[str] = set()
    final: list[str] = []
    for p in paragraphs:
        if p and p not in seen:
            seen.add(p)
            final.append(p)

    if not final[0]:
        final[0] = selected[0]

    # Final cap: each paragraph one short sentence or two; never more than 3 paragraphs
    out = "\n\n".join(final[:3])
    return out
