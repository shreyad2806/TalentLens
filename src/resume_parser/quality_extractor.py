"""
QualityMetadataExtractor for TalentLens.

Improves extraction quality over the original StructuredMetadataExtractor:
- Rejects section headings as candidate names / roles
- Extracts location only from contact/address or known cities
- Treats standalone calendar years as invalid experience
- Structures education into degree / specialization / university / year
- Extracts only project names and certification names
- Adds per-field confidence and source
- Remains deterministic and LLM-free
"""

import re
from datetime import datetime
from typing import Any

from src.normalization.role_normalizer import RoleNormalizer

from .name_validator import INVALID_CANDIDATE_NAMES, is_valid_candidate_name, normalize_candidate_name
from .normalizer import MetadataNormalizer
from .occupation_extractor import PrimaryOccupationExtractor
from .schema import Certification, Education, Experience, Project, ResumeDocument
from .section_parser import SectionParser


class QualityMetadataExtractor:
    """Extract and normalize resume metadata with quality guardrails."""

    # Stop words for names/roles/locations
    SECTION_STOP: set[str] = {
        "summary", "professional summary", "executive summary", "objective", "profile", "about me",
        "career overview", "highlights", "accomplishments",
        "experience", "relevant experience", "work experience", "professional experience", "employment history", "work history",
        "education", "academic background", "educational qualification",
        "skills", "technical skills", "core competencies", "technologies", "key skills",
        "certifications", "certificates", "professional certifications",
        "projects", "personal projects", "project experience",
        "languages", "language proficiency",
        "contact", "personal details", "personal information",
        "awards", "publications", "interests", "hobbies", "references",
    }

    JOB_TITLE_KEYWORDS = [
        "engineer", "engineering", "developer", "development", "manager", "management",
        "analyst", "analytics", "director", "architect", "lead", "leader",
        "consultant", "specialist", "intern", "associate", "coordinator", "administrator",
        "assistant", "representative", "supervisor", "executive", "officer", "scientist",
        "designer", "programmer", "tester", "support", "recruiter", "accountant",
        "teacher", "nurse", "doctor", "lawyer", "attorney", "physician", "therapist",
        "hr", "benefit", "advocate", "personnel", "customer", "brand", "grade",
        "sales", "marketing", "business", "accounting", "finance", "operations",
        "administration", "project", "healthcare", "medical", "clinical",
    ]

    US_STATES: set[str] = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    }

    LOCATION_STOP_WORDS: set[str] = {
        "management", "executive", "sales", "business", "development",
        "marketing", "engineering", "company", "city", "state", "client",
    }

    def __init__(self):
        self.section_parser = SectionParser()
        self.normalizer = MetadataNormalizer()
        self.known_locations = set(self.normalizer.LOCATION_SYNONYMS.values())

    def extract(self, text: str, record: dict[str, Any] | None = None) -> ResumeDocument:
        if not text:
            return ResumeDocument(raw_text="")

        text = self._preprocess(text)
        sections = self.section_parser.detect_sections(text)
        section_texts = self._collect_section_texts(text, sections)
        record = record or {}

        # Extract each field with confidence/source
        contact = self._extract_contact(text, record)
        candidate_name, name_conf, name_source = self._extract_candidate_name(text, record)
        role, role_conf, role_source = self._extract_role(text, record, section_texts)
        location, loc_conf, loc_source = self._extract_location(text, record, section_texts)
        skills, skill_conf, skill_source, skill_stats = self._extract_skills(text, section_texts)
        experience, experience_years, exp_conf, exp_source = self._extract_experience(text, section_texts)
        education, edu_conf, edu_source = self._extract_education(text, section_texts, record)
        certifications, cert_conf, cert_source = self._extract_certifications(section_texts)
        projects, proj_conf, proj_source = self._extract_projects(section_texts)
        summary, sum_conf, sum_source = self._extract_summary(text, section_texts)

        extraction_stats = {
            "name_found": candidate_name is not None,
            "email_valid": bool(contact.get("email")),
            "phone_valid": bool(contact.get("phone")),
            "linkedin_found": contact.get("linkedin") is not None,
            "raw_skills_count": skill_stats.get("raw_skills_count", 0),
            "normalized_skills_count": skill_stats.get("normalized_skills_count", 0),
        }

        primary_occupation = PrimaryOccupationExtractor.extract(
            parsed_resume={"experience": [e.model_dump() for e in experience] if experience else []},
            category=record.get("Category") if record else None,
            headline=summary,
        )
        extraction_stats["primary_occupation"] = primary_occupation

        return ResumeDocument(
            name=candidate_name,
            email=contact.get("email"),
            phone=contact.get("phone"),
            summary=summary,
            skills=skills,
            experience=experience,
            projects=projects,
            education=education,
            certifications=certifications,
            languages=[],
            raw_text=text,
            metadata={
                "total_experience_years": experience_years,
                "location": location,
                "extraction_source": "quality_extractor",
                "extraction_timestamp": datetime.now().isoformat(),
                "sections_detected": list(sections.keys()),
                "contact": contact,
                "extraction_stats": extraction_stats,
                "primary_occupation": primary_occupation,
                "field_confidence": {
                    "candidate_name": name_conf,
                    "role": role_conf,
                    "location": loc_conf,
                    "skills": skill_conf,
                    "experience": exp_conf,
                    "education": edu_conf,
                    "certifications": cert_conf,
                    "projects": proj_conf,
                    "summary": sum_conf,
                    "email": 1.0 if contact.get("email") else 0.0,
                    "phone": 1.0 if contact.get("phone") else 0.0,
                },
                "field_source": {
                    "candidate_name": name_source,
                    "role": role_source,
                    "location": loc_source,
                    "skills": skill_source,
                    "experience": exp_source,
                    "education": edu_source,
                    "certifications": cert_source,
                    "projects": proj_source,
                    "summary": sum_source,
                    "email": "regex" if contact.get("email") else "none",
                    "phone": "regex" if contact.get("phone") else "none",
                },
            },
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(text: str) -> str:
        if not text:
            return text
        # Normalize line endings and horizontal whitespace, but preserve vertical
        # structure so multi-line entries (experience, education, projects) remain
        # separable.
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        heading_patterns = [
            r"\b(Summary of Qualifications|Professional Summary|Career Overview|Career Summary|Executive Summary|Objective|Profile|About Me|Summary)\b",
            r"\b(Professional Experience|Relevant Experience|Work Experience|Employment History|Work History|Career History|Experience)\b",
            r"\b(Education and Training|Academic Qualifications|Academic Background|Educational Qualification|Training|Education)\b",
            r"\b(Core Competencies|Key Competencies|Areas of Expertise|Technical Skills|Professional Skills|Computer Skills|Skills & Abilities|Key Skills|Competencies|Expertise|Technologies|Skills|Highlights)\b",
            r"\b(Licenses & Certifications|Professional Certifications|Awards & Certifications|Accomplishments|Credentials|Certificates|Licenses|Certification|Certifications)\b",
            r"\b(Personal Projects|Project Experience|Relevant Projects|Academic Projects|Selected Projects|Projects)\b",
            r"\b(Language Proficiency|Languages)\b",
            r"\b(Personal Information|Personal Details|Contact)\b",
        ]
        for pattern in heading_patterns:
            text = re.sub(pattern, r"\n\1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _collect_section_texts(self, text: str, sections: dict[str, Any]) -> dict[str, str]:
        result = {name: sec.content for name, sec in sections.items()}
        for key in ["summary", "skills", "experience", "education", "projects", "certifications", "contact"]:
            if key not in result:
                content = self.section_parser.get_section_content(text, key)
                if content:
                    result[key] = content
        return result

    # ------------------------------------------------------------------
    # Contact
    # ------------------------------------------------------------------
    def _extract_contact(self, text: str, record: dict[str, Any]) -> dict[str, str | None]:
        # Email
        email = None
        m = re.search(r"[\w.-]+@[\w.-]+\.\w{2,}", text)
        if m and self._is_valid_email(m.group()):
            email = m.group()
        elif record.get("Email") and self._is_valid_email(str(record.get("Email"))):
            email = str(record.get("Email")).strip()

        # Phone
        phone = None
        for pattern in [
            r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\d{10,12}",
        ]:
            m = re.search(pattern, text)
            if m and self._is_valid_phone(m.group()):
                phone = m.group().strip()
                break
        if not phone and record.get("Phone") and self._is_valid_phone(str(record.get("Phone"))):
            phone = str(record.get("Phone")).strip()

        # LinkedIn
        linkedin = self._extract_linkedin(text)

        return {"email": email, "phone": phone, "linkedin": linkedin}

    @staticmethod
    def _is_valid_email(email: str | None) -> bool:
        if not email or not isinstance(email, str):
            return False
        email = email.strip()
        # Reject common placeholder/sentinel domains and missing parts
        if not re.match(r"^[\w.-]+@[\w.-]+\.\w{2,}$", email):
            return False
        local, _, domain = email.partition("@")
        if not local or not domain or "." not in domain:
            return False
        # Reject obviously fake/sentinel domains
        bad_domains = {"example.com", "test.com", "email.com", "domain.com", "noemail.com", "unknown"}
        if domain.lower() in bad_domains:
            return False
        return True

    @staticmethod
    def _is_valid_phone(phone: str | None) -> bool:
        if not phone or not isinstance(phone, str):
            return False
        digits = re.sub(r"\D", "", phone)
        # 10-15 digits is the most common valid range
        return 10 <= len(digits) <= 15

    @staticmethod
    def _extract_linkedin(text: str) -> str | None:
        # Full LinkedIn profile URL
        m = re.search(r"https?://(?:www\.)?linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
        if m:
            return m.group()
        # Shorthand: linkedin.com/in/...
        m = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
        if m:
            return "https://" + m.group()
        # "LinkedIn: username" or "linkedin/in/username"
        m = re.search(r"linkedin[:/\s]+in[:/\s]+([\w-]+)", text, re.IGNORECASE)
        if m:
            return f"https://www.linkedin.com/in/{m.group(1)}"
        return None

    # ------------------------------------------------------------------
    # Candidate name
    # ------------------------------------------------------------------
    def _extract_candidate_name(self, text: str, record: dict[str, Any]) -> tuple[str | None, float, str]:
        # 1. Explicit name field from CSV record
        for key in ("Name", "Candidate", "Candidate_Name", "Candidate Name"):
            if record.get(key):
                v = str(record.get(key)).strip()
                if self._looks_like_name(v):
                    return self._clean_name(v), 1.0, f"csv_{key.replace(' ', '_').lower()}"

        # 2. Resume header (first 30 lines)
        lines = text.splitlines()[:30]
        for line in lines:
            line = line.strip()
            if self._looks_like_name(line):
                return self._clean_name(line), 0.9, "header_name"

        # 3. Contact block (first 1500 chars)
        contact_text = text[:1500]
        for line in contact_text.splitlines():
            line = line.strip()
            if self._looks_like_name(line):
                return self._clean_name(line), 0.8, "contact_block_name"

        # 4. Largest / most prominent heading that is not a blacklisted section
        heading_name = self._extract_name_from_largest_heading(text)
        if heading_name:
            return self._clean_name(heading_name), 0.7, "largest_heading"

        # 5. Email signature
        signature_name = self._extract_name_from_signature(text)
        if signature_name:
            return self._clean_name(signature_name), 0.6, "email_signature"

        # 6. Filename
        filename_name = self._extract_name_from_filename(record)
        if filename_name:
            return self._clean_name(filename_name), 0.5, "filename"

        # 7. Resume ID as last resort
        resume_id = str(record.get("ID") or record.get("id") or record.get("Resume_ID") or record.get("resume_id") or "unknown")
        return resume_id, 0.1, "resume_id"

    def _looks_like_name(self, line: str) -> bool:
        """Use the shared, stricter name validator."""
        return bool(line) and is_valid_candidate_name(line)

    def _is_role_like(self, line: str) -> bool:
        low = line.lower()
        # All-caps 2-4 word phrase is almost always a job title in this dataset,
        # but only if it contains a job-title keyword (otherwise it may be a name).
        if line.isupper() and any(k in low for k in self.JOB_TITLE_KEYWORDS) and self._is_valid_role(line):
            return True
        # Otherwise it must contain a job-title keyword
        if self._is_valid_role(line) and any(k in low for k in self.JOB_TITLE_KEYWORDS):
            return True
        return False

    @staticmethod
    def _clean_name(name: str) -> str:
        return normalize_candidate_name(name)

    def _is_blacklisted_heading(self, line: str) -> bool:
        """Return True if the line is a blacklisted section heading."""
        if not line:
            return False
        low = line.lower().strip()
        if low in INVALID_CANDIDATE_NAMES:
            return True
        # Also reject substrings that are obvious section labels
        for heading in INVALID_CANDIDATE_NAMES:
            if len(heading) > 2 and heading in low:
                return True
        return False

    def _extract_name_from_largest_heading(self, text: str) -> str | None:
        """Use the largest (uppercase or title) heading that isn't a section/artifact."""
        candidate = None
        candidate_len = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) > 60 or len(line) < 4:
                continue
            if self._is_blacklisted_heading(line):
                continue
            if not is_valid_candidate_name(line):
                continue
            if len(line) > candidate_len:
                candidate = line
                candidate_len = len(line)
        return normalize_candidate_name(candidate) if candidate else None

    def _extract_name_from_signature(self, text: str) -> str | None:
        """Try to find a name in a closing email signature."""
        # Match after common closing phrases
        pattern = r"(?:Regards|Sincerely|Best regards|Kind regards|Thanks|Thank you),?\s*\n+\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the longest signature candidate
            best = max(matches, key=len)
            if self._looks_like_name(best):
                return best
        return None

    def _extract_name_from_filename(self, record: dict[str, Any]) -> str | None:
        """Try to derive a name from a resume filename."""
        filename = record.get("Filename") or record.get("File") or record.get("Resume_File")
        if not filename:
            return None
        # Remove extension and common separators
        name = re.sub(r"\.[^.]+$", "", str(filename))
        name = re.sub(r"[_-]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if self._looks_like_name(name):
            return name
        return None

    # ------------------------------------------------------------------
    # Role
    # ------------------------------------------------------------------
    def _extract_role(self, text: str, record: dict[str, Any], section_texts: dict[str, str]) -> tuple[str | None, float, str]:
        # 1. CSV Category is the most reliable for this dataset
        category = record.get("Category")
        if category:
            v = category.strip()
            if v and self._is_valid_role(v):
                return RoleNormalizer.normalize(v) or v, 1.0, "csv_category"

        # 2. Header title line before first section heading
        header_lines = text[:1000].splitlines()
        for line in header_lines:
            line = line.strip()
            if self._is_valid_role(line):
                return RoleNormalizer.normalize(line) or line, 0.9, "header_title"

        # 3. Latest experience title (first line of first job block)
        exp_text = section_texts.get("experience", "")
        if exp_text:
            lines = exp_text.splitlines()
            for line in lines:
                line = line.strip()
                if self._is_valid_role(line):
                    return RoleNormalizer.normalize(line) or line, 0.85, "latest_experience_title"

        # 4. Fallback: very first line of the resume
        first = text.splitlines()[0].strip() if text else ""
        if first and self._is_valid_role(first):
            return RoleNormalizer.normalize(first) or first, 0.7, "leading_line"

        return None, 0.0, "none_found"

    def _is_valid_role(self, value: str) -> bool:
        if not value or not isinstance(value, str):
            return False
        v = value.strip()
        if not v or len(v) > 80 or len(v) < 3:
            return False
        low = v.lower()
        # Reject section headings and contact words
        if low in self.SECTION_STOP or any(s in low for s in ["summary", "experience", "education", "contact", "personal information", "personal details", "projects", "skills", "objective", "profile", "career overview"]):
            return False
        # Reject lines that are purely numeric, contain @, or have separators like ":" "|" "/"
        if re.search(r"[\w.-]+@[\w.-]+\.\w+", v):
            return False
        if re.match(r"^\d", v):
            return False
        if re.search(r"[:|/]", v):
            return False
        # Single-word roles must be in the job-title whitelist
        tokens = re.split(r"[\s-]+", v)
        if len(tokens) == 1:
            if tokens[0].lower() not in self.JOB_TITLE_KEYWORDS:
                return False
        return True

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------
    def _extract_location(self, text: str, record: dict[str, Any], section_texts: dict[str, str]) -> tuple[str | None, float, str]:
        # 1. CSV Location, but only if it is a known location
        if record.get("Location"):
            raw = str(record.get("Location")).strip()
            if raw and self._is_known_location(raw):
                return self.normalizer.normalize_location(raw), 1.0, "csv_location"

        # 2. Contact/address section or first 800 chars
        search_text = section_texts.get("contact", text[:800])
        for pattern in [
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Z]{2})\b",
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Za-z\s]{2,30})\b",
        ]:
            for m in re.finditer(pattern, search_text):
                city = m.group(1).strip()
                state = m.group(2).strip()
                if not self._is_valid_location_city_state(city, state):
                    continue
                norm_state = self.normalizer.normalize_location(state) or state.title()
                display = f"{city.title()}, {norm_state}" if city.title() != norm_state else city.title()
                return display, 0.9, "contact_address"

        # 3. Known city keyword search in the whole text
        for synonym in self.normalizer.LOCATION_NORMALIZATION_ORDER:
            if re.search(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", text, re.IGNORECASE):
                return self.normalizer.LOCATION_SYNONYMS[synonym], 0.75, "keyword_search"

        # 4. Remote
        if re.search(r"\b(remote|work from home|wfh)\b", text, re.IGNORECASE):
            return "Remote", 0.7, "remote_keyword"

        return None, 0.0, "none_found"

    def _is_known_location(self, value: str) -> bool:
        if not value:
            return False
        low = value.lower()
        for synonym in self.normalizer.LOCATION_NORMALIZATION_ORDER:
            if synonym in low:
                return True
        return value.title() in self.known_locations

    def _is_valid_location_city_state(self, city: str, state: str) -> bool:
        if not city or not state:
            return False
        low_city = city.lower()
        low_state = state.lower()
        # Reject obvious placeholder or role-derived phrases
        if low_state == "state" or low_city == "city":
            return False
        if any(w in low_city for w in self.LOCATION_STOP_WORDS):
            return False
        if any(w in low_state for w in self.LOCATION_STOP_WORDS):
            return False
        if any(k in low_city for k in self.JOB_TITLE_KEYWORDS):
            return False
        if any(k in low_state for k in self.JOB_TITLE_KEYWORDS):
            return False
        # Accept if the state part normalizes to a known location or is a US state
        norm = self.normalizer.normalize_location(state)
        if norm and norm in self.known_locations:
            return True
        if len(state) == 2 and state.upper() in self.US_STATES:
            return True
        return False

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------
    def _extract_skills(self, text: str, section_texts: dict[str, str]) -> tuple[list[str], float, str, dict[str, int]]:
        candidates: set[str] = set()
        source = "keyword_search"

        skills_text = section_texts.get("skills", "")
        if skills_text:
            for delimiter in [",", ";", "\n", "•", "-", "*", "|", "/"]:
                if delimiter in skills_text:
                    for part in skills_text.split(delimiter):
                        part = part.strip()
                        if part and len(part) <= 60 and len(part.split()) <= 6:
                            candidates.add(part)
            source = "skills_section"

        for skill in self.normalizer.CANONICAL_SKILLS:
            if re.search(r"(?<![\w.])" + re.escape(skill) + r"(?![\w])", text, re.IGNORECASE):
                candidates.add(skill)

        raw_count = len(candidates)
        normalized = self.normalizer.normalize_skills(list(candidates))
        normalized_skills = [s for s in normalized if len(s) > 1 and not s.isdigit()]
        stats = {"raw_skills_count": raw_count, "normalized_skills_count": len(normalized_skills)}
        return normalized_skills, (0.9 if source == "skills_section" else 0.8), source, stats

    # ------------------------------------------------------------------
    # Experience
    # ------------------------------------------------------------------
    def _extract_experience(self, text: str, section_texts: dict[str, str]) -> tuple[list[Experience], float | None, float, str]:
        exp_text = section_texts.get("experience", "")
        experiences: list[Experience] = []

        if exp_text:
            blocks = self._split_experience_blocks(exp_text)
            for block in blocks:
                block = block.strip()
                if len(block) < 20:
                    continue
                exp = self._parse_experience_block(block)
                if exp and (exp.title or exp.start_date):
                    experiences.append(exp)

        # Compute years from explicit text signals only (not raw calendar years)
        years = self._extract_experience_years(text)

        # Fallback: sum duration from parsed job blocks
        if years is None and experiences:
            years = self._sum_experience_years(experiences)

        conf = 0.9 if years is not None and exp_text else (0.7 if years is not None else 0.0)
        source = "experience_section" if exp_text else ("text_regex" if years is not None else "none")
        return experiences, years, conf, source

    def _parse_experience_block(self, block: str) -> Experience | None:
        title, company = self._extract_title_company(block)
        start, end, current = self._extract_dates(block)
        location = self._extract_block_location(block)
        if not title and not start and not company:
            return None
        return Experience(
            company=self.normalizer.normalize_company(company),
            title=title,
            location=location,
            start_date=start,
            end_date=end,
            description=block,
            current=current,
        )

    def _split_experience_blocks(self, exp_text: str) -> list[str]:
        """Split an experience section into individual job blocks."""
        # First try blank-line separation
        blocks = [b.strip() for b in re.split(r"\n\s*\n", exp_text) if b.strip()]
        if len(blocks) > 1:
            return blocks

        # Fallback: split at each date-range boundary
        months = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                  r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?")
        date_pattern = (
            r"(?=\b" + months + r"\s+\d{4}\s*(?:[-–—to]+|through|to)\s*"
            r"(?:" + months + r"\s+\d{4}|present|current|now)\b)"
        )
        return [b.strip() for b in re.split(date_pattern, exp_text, flags=re.IGNORECASE) if b.strip()]

    def _extract_experience_years(self, text: str) -> float | None:
        # Explicit "X years" patterns
        written = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
        }
        for word, num in written.items():
            if re.search(rf"\b{word}\b\s+(?:years?|yrs?)", text, re.IGNORECASE):
                return float(num)

        # 5+ years, over 8 years, more than 3 years, etc.
        patterns = [
            r"(?:over|more than|less than|under|at least|around|about)\s+(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
            r"(?:worked|have|has|with)\s+(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s*experience)?",
            r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\+?\s*(?:months?|mos?)\s*(?:of\s*experience)?",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if m.lastindex == 2 and m.group(2):
                    return float(m.group(2))
                if m.group(1):
                    # If this was a months pattern, convert to years
                    if re.search(r"(?:months?|mos?)", m.group(0), re.IGNORECASE):
                        return round(float(m.group(1)) / 12, 2)
                    return float(m.group(1))
        return None

    def _sum_experience_years(self, experiences: list[Experience]) -> float | None:
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

    def _extract_title_company(self, block: str) -> tuple[str | None, str | None]:
        lines = block.splitlines()
        title = None
        company = None
        for line in lines[:3]:
            line = line.strip()
            if not line:
                continue
            if "@" in line:
                parts = line.split("@")
                if len(parts) == 2:
                    title = parts[0].strip()
                    company = parts[1].strip()
                    break
            if any(k in line.lower() for k in self.JOB_TITLE_KEYWORDS):
                title = line
            elif not company and len(line.split()) <= 4:
                company = line
        if not title and lines:
            for line in lines[:2]:
                line = line.strip()
                if len(line.split()) <= 8 and len(line) <= 70:
                    title = line
                    break
        return title, company

    def _extract_dates(self, block: str) -> tuple[str | None, str | None, bool]:
        patterns = [
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current|now)",
            r"(\d{1,2}/\d{4})\s*[-–to]+\s*(\d{1,2}/\d{4}|present|current|now)",
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
        for pattern in [
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Z]{2})\b",
            r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Za-z\s]{2,30})\b",
        ]:
            for m in re.finditer(pattern, block):
                city = m.group(1).strip()
                state = m.group(2).strip()
                if self._is_valid_location_city_state(city, state):
                    return f"{city.title()}, {self.normalizer.normalize_location(state) or state.title()}"
        for synonym in self.normalizer.LOCATION_NORMALIZATION_ORDER:
            if re.search(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", block, re.IGNORECASE):
                return self.normalizer.LOCATION_SYNONYMS[synonym]
        return None

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------
    def _extract_education(self, text: str, section_texts: dict[str, str], record: dict[str, Any]) -> tuple[list[Education], float, str]:
        # CSV Education column is trusted if present
        if record.get("Education"):
            raw = str(record.get("Education")).strip()
            if raw:
                parsed = self._parse_education_block(raw)
                if parsed:
                    return [parsed], 1.0, "csv_education"

        edu_text = section_texts.get("education", "")
        if not edu_text:
            return [], 0.0, "none_found"

        entries = []
        blocks = re.split(r"\n\s*\n", edu_text)
        for block in blocks:
            block = block.strip()
            if len(block) < 10:
                continue
            parsed = self._parse_education_block(block)
            if parsed:
                entries.append(parsed)

        conf = 0.9 if entries else 0.0
        source = "education_section" if entries else "none"
        return entries, conf, source

    def _parse_education_block(self, block: str) -> Education | None:
        # Match degree synonyms in a punctuation-normalized copy
        clean = re.sub(r"['’\.]", "", block).lower()
        degree = None
        for synonym in self.normalizer.DEGREE_NORMALIZATION_ORDER:
            pattern = r"(?<![\w])" + re.escape(synonym) + r"(?![\w])"
            if re.search(pattern, clean, re.IGNORECASE):
                degree = self.normalizer.DEGREE_SYNONYMS[synonym]
                break

        institution = None
        for line in block.splitlines()[:3]:
            line = line.strip()
            if re.search(r"\b(university|college|institute|school|academy|high school|senior high)\b", line, re.IGNORECASE):
                institution = line
                break

        # Also accept an institution anywhere in the block if not found in first 3 lines
        if not institution:
            m = re.search(
                r"\b([A-Z][\w&.,'\-]*(?:\s+[A-Z][\w&.,'\-]*)*\s*(?:University|College|Institute|School|Academy|High School|Senior High))\b",
                block,
            )
            if m:
                institution = m.group(1).strip()

        fields = ["computer science", "software engineering", "information technology", "business administration",
                  "business", "marketing", "finance", "accounting", "arts", "science", "mathematics", "statistics",
                  "electronics", "mechanical", "electrical", "civil", "law", "political science", "nursing",
                  "psychology", "communications", "human resources", "management"]
        field = None
        for f in fields:
            if re.search(r"\b" + re.escape(f) + r"\b", block, re.IGNORECASE):
                field = f.title()
                break

        end_year = None
        m = re.search(r"(\d{4})\s*[-–to]+\s*(\d{4}|present|current)", block, re.IGNORECASE)
        if m:
            end_str = m.group(2).lower()
            if end_str not in ("present", "current"):
                end_year = m.group(2)
        else:
            m = re.search(r"(?:\b|\D)(\d{4})(?:\b|\D)", block)
            if m:
                y = int(m.group(1))
                if 1980 < y < 2030:
                    end_year = m.group(1)

        if not degree and not institution:
            return None

        return Education(
            degree=degree,
            institution=institution,
            field_of_study=field,
            end_date=end_year,
            description=block,
        )

    # ------------------------------------------------------------------
    # Certifications
    # ------------------------------------------------------------------
    def _extract_certifications(self, section_texts: dict[str, str]) -> tuple[list[Certification], float, str]:
        cert_text = section_texts.get("certifications", "")
        if not cert_text:
            return [], 0.0, "none_found"

        certs = []
        for chunk in re.split(r"[\n•\-*,;|]|\s{2,}", cert_text):
            chunk = chunk.strip(" \t\n\r-•*|")
            if not chunk:
                continue
            if len(chunk) > 180:
                continue
            low = chunk.lower()
            # Drop exact section headings
            if low in self.SECTION_STOP or re.match(
                r"^(certifications?|certificates?|accomplishments|licenses?|credentials|awards)\s*$",
                low,
            ):
                continue
            certs.append(Certification(name=chunk))

        return certs, (0.85 if certs else 0.0), "certifications_section"

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    def _extract_projects(self, section_texts: dict[str, str]) -> tuple[list[Project], float, str]:
        proj_text = section_texts.get("projects", "")
        if not proj_text:
            return [], 0.0, "none_found"

        projects = []
        for chunk in re.split(r"[\n•\-*,;|]|\s{2,}", proj_text):
            chunk = chunk.strip(" \t\n\r-•*|")
            if not chunk:
                continue
            if len(chunk) < 5 or len(chunk) > 160:
                continue
            low = chunk.lower()
            # Drop exact section headings
            if low in self.SECTION_STOP or re.match(
                r"^(projects?|personal projects|academic projects|selected projects)\s*$",
                low,
            ):
                continue
            projects.append(Project(name=chunk))

        return projects, (0.85 if projects else 0.0), "projects_section"

    def _contains_section_word(self, text: str) -> bool:
        low = text.lower()
        return any(s in low for s in self.SECTION_STOP)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _extract_summary(self, text: str, section_texts: dict[str, str]) -> tuple[str | None, float, str]:
        summary = section_texts.get("summary")
        if summary:
            cleaned = self._clean_section_summary(summary)
            if cleaned and len(cleaned) >= 40 and len(cleaned.split()) >= 10:
                return cleaned, 0.9, "summary_section"

        for line in text.splitlines()[:15]:
            line = line.strip()
            if not line or len(line) < 80 or len(line.split()) < 15:
                continue
            if re.search(r"[\w.-]+@[\w.-]+\.\w+", line) or re.search(r"\d{10,15}", line):
                continue
            return line, 0.7, "first_paragraph"

        return None, 0.0, "none_found"

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
        joined = " ".join(cleaned[:4]).strip()
        return joined[:500] if joined else text.strip()[:500]
