"""Primary occupation extraction from parsed resume work history."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .schema import ResumeDocument


class PrimaryOccupationExtractor:
    """
    Extract a canonical primary occupation from resume work history.

    Selection priority (work-history-first):
        1. Most recent job title
        2. Current job title
        3. Most frequent job title across all jobs
        4. LinkedIn-style headline (if provided)
        5. Resume category

    The extractor returns a normalized primary role, role family, and seniority.
    Education and skills are intentionally ignored unless no work history exists.
    """

    SENIORITY_WORDS = [
        "chief", "ceo", "cto", "cfo", "cio", "coo", "cmo", "vp", "vice president",
        "senior", "sr", "lead", "principal", "staff", "manager", "director",
        "head of", "executive", "president",
        "junior", "jr", "associate", "assistant", "intern",
        "entry level", "entry-level", "mid-level", "midlevel", "mid level",
    ]

    ROLE_FAMILY_PATTERNS: list[tuple[re.Pattern | str, str]] = [
        # Engineering / development
        (re.compile(r"\b(machine learning|ml|deep learning|ai\b|artificial intelligence)\b", re.IGNORECASE), "Machine Learning / AI"),
        (re.compile(r"\b(software|backend|frontend|full[\s\-]?stack|web|mobile|devops|cloud|site reliability|sre)\b", re.IGNORECASE), "Software Engineering"),
        (re.compile(r"\b(data scientist|data science|data analyst|data engineer|analytics|business intelligence|bi developer)\b", re.IGNORECASE), "Data / Analytics"),
        (re.compile(r"\b(research scientist|research engineer|researcher|research assistant)\b", re.IGNORECASE), "Research"),
        (re.compile(r"\b(product manager|product owner|program manager|project manager)\b", re.IGNORECASE), "Product / Program / Project Management"),
        (re.compile(r"\b(consultant|advisory|advisor)\b", re.IGNORECASE), "Consulting"),
        (re.compile(r"\b(finance|financial|accountant|auditor|treasury|banking|investment|trader)\b", re.IGNORECASE), "Finance"),
        (re.compile(r"\b(hardware|electrical|electronics|mechanical|civil|aerospace|biomedical)\b", re.IGNORECASE), "Engineering"),
        (re.compile(r"\b(marketing|brand|content|social media|seo|growth|digital marketing)\b", re.IGNORECASE), "Marketing"),
        (re.compile(r"\b(sales|business development|account manager|account executive)\b", re.IGNORECASE), "Sales"),
        (re.compile(r"\b(human resources|hr|recruiter|talent|people operations)\b", re.IGNORECASE), "Human Resources"),
        (re.compile(r"\b(healthcare|medical|clinical|pharma|biotech|nurse|doctor|physician|therapist)\b", re.IGNORECASE), "Healthcare"),
        (re.compile(r"\b(agriculture|agronomist|farm|farming|crop|soil|horticulture|veterinary)\b", re.IGNORECASE), "Agriculture"),
        (re.compile(r"\b(teacher|professor|educator|instructor|academic|trainer)\b", re.IGNORECASE), "Education"),
        (re.compile(r"\b(lawyer|attorney|legal|counsel|paralegal)\b", re.IGNORECASE), "Legal"),
        (re.compile(r"\b(operations|supply chain|logistics|procurement|warehouse)\b", re.IGNORECASE), "Operations"),
    ]

    TITLE_NORMALIZATIONS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\bsr\.?\b", re.IGNORECASE), "Senior"),
        (re.compile(r"\bjr\.?\b", re.IGNORECASE), "Junior"),
        (re.compile(r"\bml\b", re.IGNORECASE), "Machine Learning"),
        (re.compile(r"\bai\b", re.IGNORECASE), "AI"),
        (re.compile(r"\bdev\b(?=\s|$|\.)", re.IGNORECASE), "Developer"),
        (re.compile(r"\bswe\b", re.IGNORECASE), "Software Engineer"),
        (re.compile(r"\bs\/e\b", re.IGNORECASE), "Software Engineer"),
        (re.compile(r"\bf\/e\b", re.IGNORECASE), "Frontend"),
        (re.compile(r"\bb\/e\b", re.IGNORECASE), "Backend"),
        (re.compile(r"\bfullstack\b", re.IGNORECASE), "Full Stack"),
        (re.compile(r"\bfull stack\b", re.IGNORECASE), "Full Stack"),
        (re.compile(r"\bdata sci\.?\b", re.IGNORECASE), "Data Scientist"),
        (re.compile(r"\bds\b(?=\s|$)", re.IGNORECASE), "Data Scientist"),
        (re.compile(r"\bde\b(?=\s|$)", re.IGNORECASE), "Data Engineer"),
        (re.compile(r"\bda\b(?=\s|$)", re.IGNORECASE), "Data Analyst"),
        (re.compile(r"\bpm\b(?=\s|$)", re.IGNORECASE), "Product Manager"),
    ]

    # Titles that do not fit a strong family map but are clearly engineering-adjacent
    ENGINEERING_KEYWORDS = re.compile(r"\b(engineer|developer|programmer|architect|scientist|analyst)\b", re.IGNORECASE)
    RESEARCH_KEYWORDS = re.compile(r"\b(research|researcher|research assistant|research engineer)\b", re.IGNORECASE)

    @classmethod
    def extract(
        cls,
        parsed_resume: ResumeDocument | dict[str, Any],
        category: str | None = None,
        headline: str | None = None,
    ) -> dict[str, str | None]:
        """Return the primary occupation components for a parsed resume."""
        experiences = cls._get_experiences(parsed_resume)
        source = "none"
        raw_title = None

        if experiences:
            most_recent = cls._most_recent_title(experiences)
            current = cls._current_title(experiences)
            frequent = cls._most_frequent_title(experiences)

            # 1. Most recent job title
            if most_recent:
                raw_title = most_recent
                source = "most_recent_job_title"
            # 2. Current job title (overrides if explicit current marker found)
            if current:
                raw_title = current
                source = "current_job_title"
            # 3. Most frequent job title (fallback)
            if not raw_title and frequent:
                raw_title = frequent
                source = "most_frequent_job_title"

        # 4. LinkedIn-style headline
        if not raw_title and headline:
            raw_title = headline
            source = "headline"

        # 5. Resume category
        if not raw_title and category:
            raw_title = category
            source = "resume_category"

        if not raw_title:
            return {
                "raw_title": None,
                "primary_role": None,
                "role_family": None,
                "seniority": None,
                "source": source,
            }

        primary_role = cls._normalize_title(raw_title)
        seniority = cls._extract_seniority(raw_title)
        role_family = cls._derive_role_family(primary_role, raw_title)

        return {
            "raw_title": raw_title,
            "primary_role": primary_role,
            "role_family": role_family,
            "seniority": seniority,
            "source": source,
        }

    @classmethod
    def _get_experiences(cls, parsed_resume: ResumeDocument | dict[str, Any]) -> list[Any]:
        """Return the list of parsed experience entries."""
        if isinstance(parsed_resume, ResumeDocument):
            return parsed_resume.experience or []
        if isinstance(parsed_resume, dict):
            return parsed_resume.get("experience") or []
        return []

    @classmethod
    def _year_from_date(cls, date_str: str | None) -> int | None:
        """Extract a 4-digit year from a free-form date string."""
        if not date_str:
            return None
        m = re.search(r"(19|20)\d{2}", str(date_str))
        if m:
            return int(m.group(0))
        return None

    @classmethod
    def _most_recent_title(cls, experiences: list[Any]) -> str | None:
        """Select the title from the experience with the most recent end year."""
        scored = []
        for exp in experiences:
            title = getattr(exp, "title", None) or (exp.get("title") if isinstance(exp, dict) else None)
            if not title or not title.strip():
                continue
            end = getattr(exp, "end_date", None) or (exp.get("end_date") if isinstance(exp, dict) else None)
            current = getattr(exp, "current", False) or (exp.get("current") if isinstance(exp, dict) else False)
            end_year = cls._year_from_date(end)
            if current:
                scored.append((title, 9999))
            elif end_year is not None:
                scored.append((title, end_year))
            else:
                scored.append((title, 0))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].strip()

    @classmethod
    def _current_title(cls, experiences: list[Any]) -> str | None:
        """Return the title of the explicitly current position."""
        for exp in experiences:
            current = getattr(exp, "current", False) or (exp.get("current") if isinstance(exp, dict) else False)
            if current:
                title = getattr(exp, "title", None) or (exp.get("title") if isinstance(exp, dict) else None)
                if title and title.strip():
                    return title.strip()
        return None

    @classmethod
    def _most_frequent_title(cls, experiences: list[Any]) -> str | None:
        """Return the most common job title across all experiences."""
        titles = []
        for exp in experiences:
            title = getattr(exp, "title", None) or (exp.get("title") if isinstance(exp, dict) else None)
            if title and title.strip():
                titles.append(title.strip().lower())
        if not titles:
            return None
        return Counter(titles).most_common(1)[0][0].strip().title()

    @classmethod
    def _normalize_title(cls, title: str) -> str:
        """Normalize a raw title into a clean primary role."""
        if not title:
            return ""
        cleaned = re.sub(r"[\|\,;\(\)]", " ", title)
        cleaned = re.sub(r"\b(at|@|with|for|in|the)\b.*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(since|from|to|present|current|now)\b.*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        for pattern, replacement in cls.TITLE_NORMALIZATIONS:
            cleaned = pattern.sub(replacement, cleaned)

        # Canonicalize assistant-level research roles into a real occupation title.
        cleaned = re.sub(r"\bresearch\s+assistant\b", "Research Engineer", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bresearch\s+associate\b", "Research Engineer", cleaned, flags=re.IGNORECASE)

        # Strip leading seniority words so the base role remains
        tokens = cleaned.split()
        while tokens and cls._is_seniority_word(tokens[0]):
            tokens.pop(0)
        normalized = " ".join(tokens)

        # Title-case while preserving acronyms
        normalized = cls._title_case_role(normalized)
        return normalized.strip()

    @classmethod
    def _is_seniority_word(cls, word: str) -> bool:
        """Return True if the word is a seniority/modifier that prefixes a role."""
        low = word.lower().strip(".,;")
        if not low:
            return False
        for sw in cls.SENIORITY_WORDS:
            if low == sw.replace(" ", "") or low == sw:
                return True
            if sw.endswith(low) or low.endswith(sw):
                return True
        return False

    @classmethod
    def _extract_seniority(cls, title: str) -> str | None:
        """Detect and return the highest seniority level present in the title."""
        if not title:
            return None
        low = title.lower()

        # C-level / executive first
        if re.search(r"\b(chief|ceo|cto|cfo|cio|coo|cmo)\b", low):
            return "C-Level"
        if re.search(r"\b(vp|vice president)\b", low):
            return "VP"
        if re.search(r"\b(director)\b", low):
            return "Director"
        if re.search(r"\b(head of)\b", low):
            return "Head"
        if re.search(r"\b(executive)\b", low):
            return "Executive"
        if re.search(r"\b(manager)\b", low):
            return "Manager"
        if re.search(r"\b(principal)\b", low):
            return "Principal"
        if re.search(r"\b(staff)\b", low):
            return "Staff"
        if re.search(r"\b(lead|leading)\b", low):
            return "Lead"
        if re.search(r"\b(senior|sr)\b", low):
            return "Senior"
        if re.search(r"\b(junior|jr)\b", low):
            return "Junior"
        if re.search(r"\b(associate)\b", low):
            return "Associate"
        if re.search(r"\b(assistant)\b", low):
            return "Assistant"
        if re.search(r"\b(intern|internship)\b", low):
            return "Intern"
        if re.search(r"\b(entry[-\s]?level)\b", low):
            return "Entry Level"

        return None

    @classmethod
    def _derive_role_family(cls, primary_role: str, raw_title: str) -> str | None:
        """Map the normalized role to a broader occupational family."""
        text = f"{primary_role} {raw_title}" if raw_title else primary_role
        if not text:
            return None

        for pattern, family in cls.ROLE_FAMILY_PATTERNS:
            if isinstance(pattern, str):
                if pattern.lower() in text.lower():
                    return family
            elif pattern.search(text):
                return family

        # Weak fallbacks based on core keywords
        if cls.RESEARCH_KEYWORDS.search(text):
            return "Research"
        if cls.ENGINEERING_KEYWORDS.search(text):
            return "Engineering"
        return "Other"

    @staticmethod
    def _title_case_role(text: str) -> str:
        """Title-case a role while keeping common acronyms upper-case."""
        if not text:
            return text
        acronyms = {"ml", "ai", "ii", "iii", "iiii", "sql", "api", "aws", "gcp", "ui", "ux", "qa", "devops", "sre"}
        tokens = text.split()
        out = []
        for token in tokens:
            low = token.lower().strip(".,;")
            if low in acronyms:
                out.append(token.upper() if token.isupper() else token.upper() if low in acronyms else token)
            else:
                out.append(token[0].upper() + token[1:].lower() if len(token) > 1 else token.upper())
        return " ".join(out)
