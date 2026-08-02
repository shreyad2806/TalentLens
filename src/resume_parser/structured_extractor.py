"""
Structured metadata extractor for TalentLens.

Pipeline:
    Resume text
    -> section detection
    -> regex extraction
    -> NER-style heuristics
    -> normalization
    -> structured metadata dict

This module intentionally does NOT use LLMs so the pipeline remains
deterministic and offline-capable.
"""

import re
from datetime import datetime
from typing import Any

from .normalizer import MetadataNormalizer
from .schema import ResumeDocument
from .section_parser import SectionParser


class StructuredMetadataExtractor:
    """
    Extract and normalize all resume metadata in one pass.
    """

    def __init__(self):
        self.section_parser = SectionParser()
        self.normalizer = MetadataNormalizer()

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def extract(self, text: str, record: dict[str, Any] | None = None) -> ResumeDocument:
        """
        Extract structured, normalized metadata from a resume text blob.

        Args:
            text: Raw resume text.
            record: Optional CSV record with columns like Location, Education, Email, Phone.

        Returns:
            ResumeDocument with normalized fields.
        """
        if not text:
            return ResumeDocument(raw_text=text or "")

        # 1. Normalize whitespace so regex/section detection works on long CSV strings
        text = self._preprocess(text)

        # 2. Section detection
        sections = self.section_parser.detect_sections(text)

        # 3. Extract section contents
        section_texts = self._extract_section_texts(text, sections)

        # 4. Contact from header + regex
        contact = self._extract_contact(text, record)

        # 5. Structured fields
        skills = self._extract_skills(text, section_texts)
        experience, experience_years = self._extract_experience(text, section_texts)
        education = self._extract_education(text, section_texts, record)
        certifications = self._extract_certifications(section_texts)
        projects = self._extract_projects(section_texts)
        location = self._extract_location(text, section_texts, record)
        summary = self._extract_summary(text, section_texts)
        role = self._extract_role(text, section_texts, experience)

        # 6. Build normalized ResumeDocument
        return ResumeDocument(
            name=contact.get("name") or None,
            email=contact.get("email") or None,
            phone=contact.get("phone") or None,
            summary=summary,
            skills=skills,
            experience=experience,
            projects=projects,
            education=education,
            certifications=certifications,
            languages=[],  # Optional: not a primary field
            raw_text=text,
            metadata={
                "total_experience_years": experience_years,
                "location": location,
                "extraction_source": "structured_extractor",
                "extraction_timestamp": datetime.now().isoformat(),
            },
        )

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------
    @staticmethod
    def _preprocess(text: str) -> str:
        if not text:
            return text
        # Insert newlines around known section headings so the section parser works
        # even when the CSV has collapsed whitespace.
        heading_patterns = [
            r"\b(Summary|Professional Summary|Objective|Profile|About Me)\b",
            r"\b(Experience|Work Experience|Professional Experience|Employment History|Work History)\b",
            r"\b(Education|Academic Background|Educational Qualification)\b",
            r"\b(Skills|Technical Skills|Core Competencies|Technologies|Key Skills)\b",
            r"\b(Certifications|Certificates|Professional Certifications)\b",
            r"\b(Projects|Personal Projects|Project Experience)\b",
            r"\b(Languages|Language Proficiency)\b",
            r"\b(Contact|Personal Details|Personal Information)\b",
        ]
        for pattern in heading_patterns:
            text = re.sub(pattern, r"\n\1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_section_texts(self, text: str, sections: dict[str, Any]) -> dict[str, str]:
        """Return a dict of section name -> content string."""
        result = {}
        for name, section in sections.items():
            result[name] = section.content
        # Also attempt to populate via the standalone extractors
        for key in ["summary", "skills", "experience", "education", "projects", "certifications"]:
            if key not in result:
                content = self.section_parser.get_section_content(text, key)
                if content:
                    result[key] = content
        return result

    # -----------------------------------------------------------------------
    # Contact extraction
    # -----------------------------------------------------------------------
    def _extract_contact(self, text: str, record: dict[str, Any] | None) -> dict[str, str | None]:
        contact = {"name": None, "email": None, "phone": None}

        # Email
        email_match = re.search(r"[\w.-]+@[\w.-]+\.\w{2,}", text)
        if email_match:
            contact["email"] = email_match.group()
        elif record and record.get("Email"):
            contact["email"] = str(record.get("Email")).strip()

        # Phone (international and US)
        phone_match = re.search(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{1,3}[-.\s]?\d{5,12}",
            text,
        )
        # Try a simpler broad phone pattern
        if not phone_match:
            for pattern in [
                r"\+?\d{1,3}[-.\s]\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
                r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
                r"\d{10}",
            ]:
                phone_match = re.search(pattern, text)
                if phone_match:
                    break
        if phone_match:
            contact["phone"] = phone_match.group().strip()
        elif record and record.get("Phone"):
            contact["phone"] = str(record.get("Phone")).strip()

        # Name: first substantial non-heading, non-contact line
        contact["name"] = self._extract_candidate_name(text)
        return contact

    def _extract_candidate_name(self, text: str) -> str | None:
        if not text:
            return None
        lines = text.splitlines()
        headings = {h.lower() for h in ["summary", "experience", "education", "skills", "projects", "certifications", "contact", "objective"]}

        for line in lines[:15]:  # Look in the header region
            line = line.strip()
            if not line:
                continue
            # Skip lines that are emails, phones, or URLs
            if re.search(r"[\w.-]+@[\w.-]+\.\w+", line):
                continue
            if re.search(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}|\+?\d{10,15}", line):
                continue
            if re.search(r"linkedin|github|http", line, re.IGNORECASE):
                continue
            # Likely name: 2-4 title-case-ish words, not a heading, < 80 chars
            if len(line) <= 80 and len(line.split()) in (2, 3, 4):
                if line.lower() not in headings and not re.match(r"^(\d|\W)+$", line):
                    # Try to detect if it looks like a name (contains at least two capitalized words)
                    if sum(1 for w in line.split() if w[0].isupper()) >= 2:
                        return line
        return None

    # -----------------------------------------------------------------------
    # Skills extraction
    # -----------------------------------------------------------------------
    def _extract_skills(self, text: str, section_texts: dict[str, str]) -> list[str]:
        candidates: set[str] = set()

        # 1. From explicit skills section (if found)
        skills_text = section_texts.get("skills", "")
        if skills_text:
            # Split by delimiters
            for delimiter in [",", ";", "\n", "•", "-", "*", "|", "/"]:
                if delimiter in skills_text:
                    for part in skills_text.split(delimiter):
                        part = part.strip()
                        if part and len(part) <= 60 and len(part.split()) <= 6:
                            candidates.add(part)

        # 2. From the whole resume using the canonical skill list
        for skill in self.normalizer.CANONICAL_SKILLS:
            pattern = re.compile(r"(?<![\w.])" + re.escape(skill) + r"(?![\w])", re.IGNORECASE)
            if pattern.search(text):
                candidates.add(skill)

        # 3. From metadata/fallback database broad keyword list
        # (extract any token that matches the canonical list)
        normalized = self.normalizer.normalize_skills(list(candidates))

        # Filter noise
        filtered = [s for s in normalized if len(s) > 1 and not s.isdigit()]
        return filtered

    # -----------------------------------------------------------------------
    # Experience extraction
    # -----------------------------------------------------------------------
    def _extract_experience(self, text: str, section_texts: dict[str, str]):
        from .schema import Experience

        experiences: list[Experience] = []
        exp_text = section_texts.get("experience", "")

        # Try to split experience text into entries separated by blank lines or dates
        blocks = re.split(r"\n\s*\n", exp_text or text) if exp_text else []
        if not blocks:
            # Fallback: look for entries with date patterns in the whole text
            blocks = self._split_by_dates(text)

        for block in blocks:
            block = block.strip()
            if len(block) < 20:
                continue

            title, company = self._extract_title_company(block)
            start, end, current = self._extract_dates(block)
            location = self._extract_block_location(block)

            exp = Experience(
                company=company,
                title=title,
                location=location,
                start_date=start,
                end_date=end,
                description=block,
                current=current,
            )
            experiences.append(exp)

        # Total years from text
        years = self.normalizer.normalize_experience_years(text)
        if years is None and experiences:
            # Sum the years across entries
            years = self._sum_experience_years(experiences)

        return experiences, years

    def _split_by_dates(self, text: str) -> list[str]:
        """Split text around common date patterns to identify job blocks."""
        date_pattern = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4})"
        parts = re.split(date_pattern, text)
        blocks = []
        current = []
        for part in parts:
            if re.match(date_pattern, part, re.IGNORECASE):
                if current:
                    blocks.append("".join(current))
                    current = []
                current.append(part)
            else:
                current.append(part)
        if current:
            blocks.append("".join(current))
        return [b.strip() for b in blocks if len(b.strip()) > 20]

    def _extract_title_company(self, block: str) -> tuple[str | None, str | None]:
        lines = block.splitlines()
        title = None
        company = None

        # Common patterns:
        # "Title @ Company"
        # "Title | Company"
        at_match = re.search(r"^(.+?)\s*[@|]\s*(.+?)$", lines[0] if lines else "")
        if at_match:
            title = at_match.group(1).strip()
            company = at_match.group(2).strip()
            return title, company

        # First 2 lines: one is title, one is company
        for line in lines[:3]:
            line = line.strip()
            if not line or len(line) > 80:
                continue
            lower = line.lower()
            if any(w in lower for w in ["engineer", "developer", "manager", "analyst", "director", "architect", "lead", "consultant", "specialist", "intern", "associate"]):
                title = line
            elif any(w in lower for w in ["company", "inc", "corp", "ltd", "llc", "limited", "solutions", "technologies", "services", "group"]):
                company = line

        if not title:
            # First non-date, non-location, short line
            for line in lines[:2]:
                if re.search(r"\d{4}", line):
                    continue
                if len(line.strip().split()) <= 8 and len(line.strip()) <= 60:
                    title = line.strip()
                    break
        return title, company

    def _extract_dates(self, block: str) -> tuple[str | None, str | None, bool]:
        patterns = [
            # Jan 2018 - Dec 2020 / Jan 2018 to Present
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current|now)",
            # 01/2018 - 12/2020
            r"(\d{1,2}/\d{4})\s*[-–to]+\s*(\d{1,2}/\d{4}|present|current|now)",
            # 2018 - 2020
            r"(\d{4})\s*[-–to]+\s*(\d{4}|present|current|now)",
        ]
        for pattern in patterns:
            m = re.search(pattern, block, re.IGNORECASE)
            if m:
                start = m.group(1)
                end = m.group(2)
                current = end.lower() in ("present", "current", "now")
                return start, None if current else end, current
        return None, None, False

    def _extract_block_location(self, block: str) -> str | None:
        m = re.search(r"([A-Z][a-z]+,\s*[A-Z]{2}|[A-Z][a-z]+\s*,\s*[A-Za-z\s]+)", block)
        return m.group(1).strip() if m else None

    def _sum_experience_years(self, experiences: list) -> float | None:
        total = 0.0
        now = datetime.now()
        for exp in experiences:
            if not exp.start_date:
                continue
            start_year = self._year_from_date(exp.start_date)
            if not start_year:
                continue
            if exp.current:
                end_year = now.year
            else:
                end_year = self._year_from_date(exp.end_date) or now.year
            total += max(0.0, end_year - start_year)
        return total if total > 0 else None

    @staticmethod
    def _year_from_date(date_str: str | None) -> int | None:
        if not date_str:
            return None
        m = re.search(r"(\d{4})", date_str)
        if m:
            return int(m.group(1))
        return None

    # -----------------------------------------------------------------------
    # Education extraction
    # -----------------------------------------------------------------------
    def _extract_education(self, text: str, section_texts: dict[str, str], record: dict[str, Any] | None) -> list:
        from .schema import Education

        edu_text = section_texts.get("education", "")
        if not edu_text:
            # Fallback: look for degree keywords in the first 600 chars
            edu_text = text[:1200]

        # If the CSV has an explicit Education column, use it as a primary source
        if record and record.get("Education"):
            raw = str(record.get("Education")).strip()
            if raw:
                # Create a single education entry from the explicit column
                degree = self.normalizer.normalize_education(raw)
                return [Education(degree=degree, institution=None, description=raw)]

        entries = []
        blocks = re.split(r"\n\s*\n", edu_text)
        for block in blocks:
            block = block.strip()
            if len(block) < 10:
                continue
            degree = self.normalizer.normalize_education(block)
            institution = self._extract_institution(block)
            field = self._extract_field_of_study(block)
            entries.append(Education(
                degree=degree,
                institution=institution,
                field_of_study=field,
                description=block,
            ))

        # If nothing found, try a degree keyword scan on the whole text
        if not entries:
            for deg_syn in self.normalizer.DEGREE_NORMALIZATION_ORDER:
                if re.search(re.escape(deg_syn), text, re.IGNORECASE):
                    deg = self.normalizer.DEGREE_SYNONYMS[deg_syn]
                    return [Education(degree=deg, institution=None, description=f"Degree detected: {deg}")]

        return entries

    def _extract_institution(self, block: str) -> str | None:
        for line in block.splitlines()[:3]:
            line = line.strip()
            if re.search(r"\b(university|college|institute|school|academy)\b", line, re.IGNORECASE):
                return line
            if len(line.split()) <= 6 and len(line) <= 80:
                return line
        return None

    def _extract_field_of_study(self, block: str) -> str | None:
        fields = ["computer science", "engineering", "information technology", "business", "arts", "science", "mathematics", "electronics", "mechanical", "electrical", "civil"]
        for field in fields:
            if re.search(r"\b" + re.escape(field) + r"\b", block, re.IGNORECASE):
                return field.title()
        return None

    # -----------------------------------------------------------------------
    # Certifications extraction
    # -----------------------------------------------------------------------
    def _extract_certifications(self, section_texts: dict[str, str]) -> list:
        from .schema import Certification
        cert_text = section_texts.get("certifications", "")
        if not cert_text:
            return []

        certs = []
        for line in cert_text.splitlines():
            line = line.strip()
            if len(line) < 5:
                continue
            if re.match(r"^(certifications?|certificates?|professional)\s*$", line, re.IGNORECASE):
                continue
            # Try to find an issuer
            issuers = ["aws", "google", "microsoft", "oracle", "cisco", "pmp", "scrum", "salesforce", "comptia", "isc"]
            issuer = None
            for i in issuers:
                if i in line.lower():
                    issuer = i.upper() if i in ("aws", "pmp") else i.title()
                    break
            certs.append(Certification(name=line, issuer=issuer))
        return certs

    # -----------------------------------------------------------------------
    # Projects extraction
    # -----------------------------------------------------------------------
    def _extract_projects(self, section_texts: dict[str, str]) -> list:
        from .schema import Project
        proj_text = section_texts.get("projects", "")
        if not proj_text:
            return []

        projects = []
        blocks = re.split(r"\n\s*\n", proj_text)
        for block in blocks:
            block = block.strip()
            if len(block) < 20:
                continue
            lines = block.splitlines()
            name = lines[0].strip() if lines else None
            technologies = []
            for skill in self.normalizer.CANONICAL_SKILLS:
                if re.search(r"(?<![\w.])" + re.escape(skill) + r"(?![\w])", block, re.IGNORECASE):
                    technologies.append(skill)
            projects.append(Project(name=name, description=block, technologies=technologies))
        return projects

    # -----------------------------------------------------------------------
    # Location extraction
    # -----------------------------------------------------------------------
    def _extract_location(self, text: str, section_texts: dict[str, str], record: dict[str, Any] | None) -> str | None:
        # CSV column is most reliable
        if record and record.get("Location"):
            raw = str(record.get("Location")).strip()
            if raw:
                return self.normalizer.normalize_location(raw)

        # Try contact/address section or text
        target = section_texts.get("contact", text)

        # City, State/Country patterns
        patterns = [
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?,\s*[A-Z]{2})\b",
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?,\s*[A-Z][a-zA-Z]+)\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, target)
            if m:
                return self.normalizer.normalize_location(m.group(1))

        # Known city/country keyword search
        for synonym in self.normalizer.LOCATION_NORMALIZATION_ORDER:
            if re.search(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", text, re.IGNORECASE):
                return self.normalizer.LOCATION_SYNONYMS[synonym]

        # Remote
        if re.search(r"\b(remote|work from home|wfh)\b", text, re.IGNORECASE):
            return "Remote"

        return None

    # -----------------------------------------------------------------------
    # Summary extraction
    # -----------------------------------------------------------------------
    def _extract_summary(self, text: str, section_texts: dict[str, str]) -> str | None:
        summary = section_texts.get("summary")
        if summary:
            return self._clean_section_summary(summary)
        # No section: take first non-name, non-contact paragraph as summary
        for line in text.splitlines()[:15]:
            line = line.strip()
            if not line:
                continue
            if re.search(r"[\w.-]+@[\w.-]+\.\w+", line):
                continue
            if re.search(r"\d{10,15}", line):
                continue
            if len(line) > 80 and len(line.split()) >= 10:
                return line
        return None

    @staticmethod
    def _clean_section_summary(text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if re.match(r"^(summary|objective|profile)\s*$", line, re.IGNORECASE):
                continue
            if line:
                cleaned.append(line)
        return " ".join(cleaned[:4]).strip() if cleaned else text.strip()

    # -----------------------------------------------------------------------
    # Role extraction
    # -----------------------------------------------------------------------
    def _extract_role(self, text: str, section_texts: dict[str, str], experiences: list) -> str | None:
        # 1. First experience title
        if experiences:
            for exp in experiences:
                if exp.title:
                    return exp.title

        # 2. Look for a leading role/title line in the header
        for line in text.splitlines()[:15]:
            line = line.strip()
            if not line or len(line) > 80:
                continue
            lower = line.lower()
            if any(w in lower for w in ["engineer", "developer", "manager", "analyst", "director", "architect", "lead", "consultant", "specialist", "intern", "associate", "coordinator", "administrator"]):
                return line

        return None
