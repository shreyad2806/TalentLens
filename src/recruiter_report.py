"""Deterministic, metadata-grounded recruiter report generator for TalentLens."""

from __future__ import annotations


def _norm(text: str | None) -> str:
    return (text or "").lower().strip()


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if v]
    return [str(v).strip() for v in str(value).split(",") if v.strip()]


def _candidate_skills(candidate: dict) -> set[str]:
    skills: list[str] = []
    for key in ("skills", "top_skills", "primary_skills", "secondary_skills", "matched_skills"):
        skills.extend(_as_list(candidate.get(key)))
    return {_norm(s) for s in skills}


def build_recruiter_report(candidate: dict, query: str, parsed_query: dict | None) -> dict:
    """Build a recruiter report using only the candidate dict and parsed query."""
    parsed_query = parsed_query or {}
    name = (candidate.get("name") or "Candidate").strip()
    role = (candidate.get("role") or candidate.get("primary_role") or "the role").strip()
    years = candidate.get("experience", "")
    experience_years = candidate.get("experience_years") or 0.0
    if not experience_years and years and "year" in str(years).lower():
        try:
            experience_years = float(str(years).split()[0])
        except ValueError:
            experience_years = 0.0

    all_skills = _as_list(candidate.get("skills")) or _as_list(candidate.get("top_skills"))
    matched_skills = _as_list(candidate.get("matched_skills"))
    matched_skills_set = {_norm(s) for s in matched_skills}

    query_skills = _as_list(parsed_query.get("Skills") or parsed_query.get("skills"))
    missing_skills = [s for s in query_skills if _norm(s) and _norm(s) not in _candidate_skills(candidate)]

    # 1. Candidate overview (2-3 sentences, using existing ai_summary if available).
    overview = str(candidate.get("summary") or candidate.get("ai_summary") or "").strip()
    if not overview:
        overview = f"{name} is a {role} with {years} of experience."
    sentences = [s.strip() for s in overview.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    overview = ". ".join(sentences[:3]) + "." if sentences else overview
    if not overview.endswith((".", "!", "?")):
        overview += "."

    # 2. Strengths (from matched signals, no hallucination).
    strengths: list[str] = []
    role_score = float(candidate.get("role_match", 0) or 0)
    skill_score = candidate.get("skill_match", 0)
    skill_score = float(skill_score if isinstance(skill_score, (int, float)) else 0)
    exp_score = float(candidate.get("experience_match", 0) or 0)
    overall = float(candidate.get("overall_match", 0) or 0)

    if role_score >= 80:
        strengths.append(f"Strong {role} profile (role match {role_score:.0f}%)")
    if skill_score >= 80:
        strengths.append(f"High skill alignment ({skill_score:.0f}% match)")
    for sk in matched_skills[:6]:
        if sk:
            strengths.append(f"Demonstrated {sk} experience")
    if experience_years and experience_years >= 5:
        strengths.append(f"{experience_years:g} years of relevant experience")
    if matched_edu := _as_list(candidate.get("matched_education")):
        for e in matched_edu[:2]:
            strengths.append(f"Relevant education in {e}")
    for p in _as_list(candidate.get("projects"))[:2]:
        strengths.append(f"Project experience: {p}")
    if not strengths:
        strengths.append("Relevant profile for this search")

    # 3. Potential gaps (only real missing skills from the query or matched thresholds).
    gaps: list[str] = []
    if missing_skills:
        for ms in missing_skills[:5]:
            gaps.append(f"No explicit {ms} experience")
    if skill_score and skill_score < 50:
        gaps.append("Skill match is below the strong threshold")
    if role_score and role_score < 50:
        gaps.append("Role alignment is below the strong threshold")
    if exp_score and exp_score < 50:
        gaps.append("Experience level is below the strong threshold")

    # 4. Relevant skills (top 10; matched first).
    seen: set[str] = set()
    relevant_skills: list[str] = []
    for sk in matched_skills + all_skills:
        key = _norm(sk)
        if key and key not in seen:
            seen.add(key)
            relevant_skills.append(sk)
    relevant_skills = relevant_skills[:10]

    # 5. Relevant projects (top 3).
    relevant_projects = _as_list(candidate.get("projects"))[:3]

    # 6. Suggested interview questions (personalized from strengths and gaps).
    questions: list[str] = []
    if matched_skills:
        questions.append(f"Describe a time you used {matched_skills[0]} to solve a real problem.")
    if relevant_projects:
        questions.append(f"Explain your role and results on the {relevant_projects[0]} project.")
    for ms in missing_skills[:2]:
        questions.append(f"What is your experience with {ms}?")
    questions.append(f"What is the most complex {role} challenge you have faced?")
    if len(questions) < 5 and matched_skills[1:2]:
        questions.append(f"Walk me through a recent project where you used {matched_skills[1]}.")
    if len(questions) < 5:
        questions.append("How do you keep your technical skills current?")
    questions = questions[:5]

    # 7. Hiring recommendation.
    skill_value = skill_score
    if overall >= 90 and role_score >= 80 and skill_value >= 80:
        recommendation = "Highly Recommended"
    elif overall >= 80 and role_score >= 60 and skill_value >= 60:
        recommendation = "Recommended"
    elif overall >= 60:
        recommendation = "Worth Interviewing"
    elif overall >= 40:
        recommendation = "Borderline"
    else:
        recommendation = "Not Recommended"

    if matched_skills:
        reason = f"{recommendation} based on {overall:.0f}% overall match, strong {role} alignment, and {', '.join(matched_skills[:4])} experience."
    elif overall >= 60:
        reason = f"{recommendation} based on {overall:.0f}% overall match and a {role} profile that fits the search."
    else:
        reason = f"{recommendation} because overall match ({overall:.0f}%) is low and key requirements are missing."

    # 8. Match percentages.
    match = {
        "overall": float(candidate.get("overall_match", 0) or 0),
        "role": float(candidate.get("role_match", 0) or 0),
        "skill": float(skill_score if isinstance(skill_score, (int, float)) else 0),
        "experience": float(candidate.get("experience_match", 0) or 0),
    }

    return {
        "name": name,
        "role": role,
        "overview": overview,
        "strengths": strengths,
        "gaps": gaps,
        "relevant_skills": relevant_skills,
        "missing_skills": missing_skills,
        "relevant_projects": relevant_projects,
        "interview_questions": questions,
        "hiring_recommendation": recommendation,
        "hiring_explanation": reason,
        "match": match,
    }


def format_markdown(report: dict) -> str:
    """Return a professional Markdown version of the report."""
    lines = [
        f"# Recruiter Report: {report['name']}",
        "",
        "## Candidate Overview",
        report["overview"],
        "",
        "## Strengths",
    ]
    for s in report["strengths"]:
        lines.append(f"- {s}")
    lines.extend(["", "## Potential Gaps"])
    if report["gaps"]:
        for g in report["gaps"]:
            lines.append(f"- {g}")
    else:
        lines.append("- No significant gaps identified")
    lines.extend(["", "## Relevant Skills"])
    for sk in report["relevant_skills"]:
        lines.append(f"- {sk}")
    lines.extend(["", "## Missing Skills"])
    if report["missing_skills"]:
        for ms in report["missing_skills"]:
            lines.append(f"- {ms}")
    else:
        lines.append("- None identified")
    lines.extend(["", "## Relevant Projects"])
    for p in report["relevant_projects"]:
        lines.append(f"- {p}")
    lines.extend(["", "## Suggested Interview Questions"])
    for i, q in enumerate(report["interview_questions"], start=1):
        lines.append(f"{i}. {q}")
    lines.extend([
        "",
        "## Hiring Recommendation",
        f"**{report['hiring_recommendation']}**",
        report["hiring_explanation"],
        "",
        "## Overall Match",
        f"- Overall Match: {report['match']['overall']:.0f}%",
        f"- Role Match: {report['match']['role']:.0f}%",
        f"- Skill Match: {report['match']['skill']:.0f}%",
        f"- Experience Match: {report['match']['experience']:.0f}%",
    ])
    return "\n".join(lines)


def format_text(report: dict) -> str:
    """Return a plain-text version of the report."""
    sections = [
        f"Recruiter Report: {report['name']}",
        "",
        "CANDIDATE OVERVIEW",
        report["overview"],
        "",
        "STRENGTHS",
    ]
    for s in report["strengths"]:
        sections.append(f"- {s}")
    sections.extend(["", "POTENTIAL GAPS"])
    if report["gaps"]:
        for g in report["gaps"]:
            sections.append(f"- {g}")
    else:
        sections.append("- No significant gaps identified")
    sections.extend(["", "RELEVANT SKILLS"])
    for sk in report["relevant_skills"]:
        sections.append(f"- {sk}")
    sections.extend(["", "MISSING SKILLS"])
    if report["missing_skills"]:
        for ms in report["missing_skills"]:
            sections.append(f"- {ms}")
    else:
        sections.append("- None identified")
    sections.extend(["", "RELEVANT PROJECTS"])
    for p in report["relevant_projects"]:
        sections.append(f"- {p}")
    sections.extend(["", "SUGGESTED INTERVIEW QUESTIONS"])
    for i, q in enumerate(report["interview_questions"], start=1):
        sections.append(f"{i}. {q}")
    sections.extend([
        "",
        "HIRING RECOMMENDATION",
        report["hiring_recommendation"],
        report["hiring_explanation"],
        "",
        "OVERALL MATCH",
        f"Overall Match: {report['match']['overall']:.0f}%",
        f"Role Match: {report['match']['role']:.0f}%",
        f"Skill Match: {report['match']['skill']:.0f}%",
        f"Experience Match: {report['match']['experience']:.0f}%",
    ])
    return "\n".join(sections)
