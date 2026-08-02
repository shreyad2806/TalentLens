"""Backend Resume Preview Generator for TalentLens.

The frontend should never build resume previews.  This module converts a
ResumeDocument into a concise, recruiter-friendly plain-text preview.
"""

from __future__ import annotations

import re

from src.models import ResumeDocument


class ResumePreviewGenerator:
    """Generate a concise, recruiter-friendly resume preview from a ResumeDocument."""

    MAX_LENGTH = 300
    MAX_SKILLS = 4
    MAX_SUMMARY_WORDS = 25

    def generate(self, resume: ResumeDocument) -> str:
        """Return a plain-text preview with Role, Experience, Skills, and Summary."""
        role = self._format_role(resume.role)
        experience = self._format_experience(resume.experience_years)
        skills = self._format_skills(resume.skills)
        summary = self._format_summary(resume)

        parts: list[str] = []
        if role:
            parts.append(role)
        if experience:
            parts.append(experience)
        if skills:
            parts.append(skills)
        if summary:
            parts.append(summary)

        preview = "\n".join(parts)
        if not preview:
            preview = "No preview available."

        return self._truncate_to(preview, self.MAX_LENGTH)

    def _format_role(self, role: str | None) -> str:
        return (role or "").strip()

    def _format_experience(self, years: float | None) -> str:
        if years is None or years <= 0:
            return ""
        if years == 1.0:
            return "1 year experience"
        return f"{years:g} years experience"

    def _format_skills(self, skills: list[str] | None) -> str:
        if not skills:
            return ""
        clean = [s.strip() for s in skills if s and s.strip()]
        top = clean[: self.MAX_SKILLS]
        return " • ".join(top)

    def _format_summary(self, resume: ResumeDocument) -> str:
        # 1. Prefer the professional summary.
        if resume.summary:
            return self._clean_summary(resume.summary)

        # 2. Try to extract a concise sentence from the latest experience section.
        if resume.resume_text:
            return self._extract_experience_sentence(resume.resume_text)

        # 3. Fall back to the role/experience sentence.
        if resume.role:
            return f"{resume.role} professional."

        return ""

    def _clean_summary(self, text: str) -> str:
        text = self._remove_markdown(text)
        text = re.sub(r"\s+", " ", text).strip()
        # Limit to a reasonable number of words and one sentence if needed.
        words = text.split()
        if len(words) > self.MAX_SUMMARY_WORDS:
            truncated = " ".join(words[: self.MAX_SUMMARY_WORDS])
            # truncate at last full stop or keep the partial words
            if "." in truncated:
                truncated = truncated.rsplit(".", 1)[0] + "."
            else:
                truncated += "..."
            return truncated
        return text

    def _extract_experience_sentence(self, text: str) -> str:
        """Pull a short, meaningful sentence that sounds like experience."""
        text = self._remove_markdown(text)
        # Split on sentence boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            # Look for action-oriented wording that likely describes work.
            if re.search(
                r"\b(developed|built|engineered|designed|implemented|led|managed|created|architected|maintained|optimized|worked|experienced)\b",
                sent,
                re.IGNORECASE,
            ):
                return sent

        # No action sentence found: take the first non-trivial sentence.
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= 20:
                return sent

        return ""

    def _remove_markdown(self, text: str) -> str:
        text = re.sub(r"\*\*|__", "", text)  # bold/italic markers
        text = re.sub(r"#+\s*", "", text)  # headings
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _truncate_to(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text

        # Try to truncate at the last sentence boundary that still fits.
        cut = text.rfind(". ", 0, max_len - 3)
        if cut == -1:
            cut = text.rfind(" ", 0, max_len - 3)
        if cut == -1:
            cut = max_len - 3

        truncated = text[:cut].strip()
        if not truncated.endswith("."):
            truncated += "..."
        return truncated
