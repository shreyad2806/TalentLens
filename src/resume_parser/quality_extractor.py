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
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import ResumeDocument, Experience, Education, Project, Certification
from .section_parser import SectionParser
from .normalizer import MetadataNormalizer


class QualityMetadataExtractor:
    """Extract and normalize resume metadata with quality guardrails."""

    # Stop words for names/roles/locations
    SECTION_STOP: Set[str] = {
        "summary", "professional summary", "objective", "profile", "about me",
        "career overview", "highlights", "accomplishments",
        "experience", "work experience", "professional experience", "employment history", "work history",
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

    US_STATES: Set[str] = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    }

    LOCATION_STOP_WORDS: Set[str] = {
        "management", "executive", "sales", "business", "development",
        "marketing", "engineering", "company", "city", "state", "client",
    }

    def __init__(self):
        self.section_parser = SectionParser()
        self.normalizer = MetadataNormalizer()
        self.known_locations = set(self.normalizer.LOCATION_SYNONYMS.values())

    def extract(self, text: str, record: Optional[Dict[str, Any]] = None) -> ResumeDocument:
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
        skills, skill_conf, skill_source = self._extract_skills(text, section_texts)
        experience, experience_years, exp_conf, exp_source = self._extract_experience(text, section_texts)
        education, edu_conf, edu_source = self._extract_education(text, section_texts, record)
        certifications, cert_conf, cert_source = self._extract_certifications(section_texts)
        projects, proj_conf, proj_source = self._extract_projects(section_texts)
        summary, sum_conf, sum_source = self._extract_summary(text, section_texts)

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

    def _collect_section_texts(self, text: str, sections: Dict[str, Any]) -> Dict[str, str]:
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
    def _extract_contact(self, text: str, record: Dict[str, Any]) -> Dict[str, Optional[str]]:
        email = None
        m = re.search(r"[\w.-]+@[\w.-]+\.\w{2,}", text)
        if m:
            email = m.group()
        elif record.get("Email"):
            email = str(record.get("Email")).strip()

        phone = None
        for pattern in [
            r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\d{10,12}",
        ]:
            m = re.search(pattern, text)
            if m:
                phone = m.group().strip()
                break
        if not phone and record.get("Phone"):
            phone = str(record.get("Phone")).strip()

        return {"email": email, "phone": phone}

    # ------------------------------------------------------------------
    # Candidate name
    # ------------------------------------------------------------------
    def _extract_candidate_name(self, text: str, record: Dict[str, Any]) -> Tuple[Optional[str], float, str]:
        # Prefer CSV name columns if present and valid
        for key in ("Name", "Candidate"):
            if record.get(key):
                v = str(record.get(key)).strip()
                if self._looks_like_name(v):
                    return v, 1.0, f"csv_{key.lower()}"

        lines = text.splitlines()[:30]

        # First, find a line that looks like a name and is not a heading/contact/role
        for line in lines:
            line = line.strip()
            if self._looks_like_name(line):
                return self._clean_name(line), 0.9, "header_name"

        # Search within the contact block (first 1500 chars)
        contact_text = text[:1500]
        for line in contact_text.splitlines():
            line = line.strip()
            if self._looks_like_name(line):
                return self._clean_name(line), 0.8, "contact_block_name"

        return None, 0.0, "none_found"

    def _looks_like_name(self, line: str) -> bool:
        if not line or len(line) > 40 or len(line) < 4:
            return False
        low = line.lower()
        # Reject exact section headings, not arbitrary substrings
        if low in self.SECTION_STOP:
            return False
        tokens = line.split()
        if len(tokens) < 2 or len(tokens) > 4:
            return False
        if re.search(r"\d", line) or re.search(r"[/@:|]", line):
            return False
        # Reject if it looks like a job title
        if self._is_role_like(line):
            return False
        # Names are either title-cased or all-caps
        title_cased = sum(1 for t in tokens if t and t[0].isupper()) >= len(tokens)
        all_caps = all(t.isupper() for t in tokens)
        if not (title_cased or all_caps):
            return False
        # At least one token should be longer than 1 character
        if not any(len(t) > 1 for t in tokens):
            return False
        return True

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
        return re.sub(r"\s+", " ", name).strip()

    # ------------------------------------------------------------------
    # Role
    # ------------------------------------------------------------------
    def _extract_role(self, text: str, record: Dict[str, Any], section_texts: Dict[str, str]) -> Tuple[Optional[str], float, str]:
        # 1. CSV Category is the most reliable for this dataset
        category = record.get("Category")
        if category:
            v = category.strip()
            if v and self._is_valid_role(v):
                return v, 1.0, "csv_category"

        # 2. Header title line before first section heading
        header_lines = text[:1000].splitlines()
        for line in header_lines:
            line = line.strip()
            if self._is_valid_role(line):
                return line, 0.9, "header_title"

        # 3. Latest experience title (first line of first job block)
        exp_text = section_texts.get("experience", "")
        if exp_text:
            lines = exp_text.splitlines()
            for line in lines:
                line = line.strip()
                if self._is_valid_role(line):
                    return line, 0.85, "latest_experience_title"

        # 4. Fallback: very first line of the resume
        first = text.splitlines()[0].strip() if text else ""
        if first and self._is_valid_role(first):
            return first, 0.7, "leading_line"

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
    def _extract_location(self, text: str, record: Dict[str, Any], section_texts: Dict[str, str]) -> Tuple[Optional[str], float, str]:
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
    def _extract_skills(self, text: str, section_texts: Dict[str, str]) -> Tuple[List[str], float, str]:
        candidates: Set[str] = set()
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

        normalized = self.normalizer.normalize_skills(list(candidates))
        return [s for s in normalized if len(s) > 1 and not s.isdigit()], (0.9 if source == "skills_section" else 0.8), source

    # ------------------------------------------------------------------
    # Experience
    # ------------------------------------------------------------------
    def _extract_experience(self, text: str, section_texts: Dict[str, str]) -> Tuple[List[Experience], Optional[float], float, str]:
        exp_text = section_texts.get("experience", "")
        experiences: List[Experience] = []

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

    def _parse_experience_block(self, block: str) -> Optional[Experience]:
        title, company = self._extract_title_company(block)
        start, end, current = self._extract_dates(block)
        location = self._extract_block_location(block)
        if not title and not start and not company:
            return None
        return Experience(
            company=company,
            title=title,
            location=location,
            start_date=start,
            end_date=end,
            description=block,
            current=current,
        )

    def _split_experience_blocks(self, exp_text: str) -> List[str]:
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

    def _extract_experience_years(self, text: str) -> Optional[float]:
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
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if m.lastindex == 2 and m.group(2):
                    return float(m.group(2))
                if m.group(1):
                    return float(m.group(1))
        return None

    def _sum_experience_years(self, experiences: List[Experience]) -> Optional[float]:
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
    def _year_from_date(date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        m = re.search(r"(\d{4})", date_str)
        if m:
            return int(m.group(1))
        return None

    def _extract_title_company(self, block: str) -> Tuple[Optional[str], Optional[str]]:
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

    def _extract_dates(self, block: str) -> Tuple[Optional[str], Optional[str], bool]:
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

    def _extract_block_location(self, block: str) -> Optional[str]:
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
    def _extract_education(self, text: str, section_texts: Dict[str, str], record: Dict[str, Any]) -> Tuple[List[Education], float, str]:
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

    def _parse_education_block(self, block: str) -> Optional[Education]:
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
    def _extract_certifications(self, section_texts: Dict[str, str]) -> Tuple[List[Certification], float, str]:
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
    def _extract_projects(self, section_texts: Dict[str, str]) -> Tuple[List[Project], float, str]:
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
    def _extract_summary(self, text: str, section_texts: Dict[str, str]) -> Tuple[Optional[str], float, str]:
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
