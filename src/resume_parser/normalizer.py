"""
Metadata normalizer for TalentLens.

Provides canonical forms for skills, degrees, locations, and experience values
so downstream ChunkMetadata / EmbeddingRecord / BM25Document objects contain
uniform, predictable values instead of free-text noise.
"""

import re
from typing import Any


class MetadataNormalizer:
    """Normalize extracted metadata fields in one central place."""

    # -----------------------------------------------------------------------
    # Skill normalization
    # -----------------------------------------------------------------------
    SKILL_SYNONYMS: dict[str, str] = {
        # JavaScript ecosystem
        "node": "Node.js",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node js": "Node.js",
        "nodejs.js": "Node.js",
        "reactjs": "React",
        "react.js": "React",
        "react js": "React",
        "vuejs": "Vue",
        "vue.js": "Vue",
        "nextjs": "Next.js",
        "next.js": "Next.js",
        "nuxtjs": "Nuxt.js",
        "nuxt.js": "Nuxt.js",
        "angularjs": "Angular",
        "angular.js": "Angular",
        # Python / ML
        "python3": "Python",
        "python 3": "Python",
        "python2": "Python",
        "python 2": "Python",
        "machine learning": "ML",
        "machine-learning": "ML",
        "deep learning": "DL",
        "deep-learning": "DL",
        "artificial intelligence": "AI",
        "artificial-intelligence": "AI",
        "scikit learn": "Scikit-learn",
        "scikit-learn": "Scikit-learn",
        "sklearn": "Scikit-learn",
        # Databases
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "postgre sql": "PostgreSQL",
        "mssql": "SQL Server",
        "ms sql": "SQL Server",
        "microsoft sql": "SQL Server",
        "sql server": "SQL Server",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "mongo db": "MongoDB",
        "sqlite": "SQLite",
        # Languages / frameworks
        "java": "Java",
        "core java": "Java",
        "javascript": "JavaScript",
        "java script": "JavaScript",
        "typescript": "TypeScript",
        "type script": "TypeScript",
        "c#": "C#",
        "csharp": "C#",
        "c++": "C++",
        "cpp": "C++",
        "golang": "Go",
        "objective-c": "Objective-C",
        "rest api": "REST API",
        "restful api": "REST API",
        "rest apis": "REST API",
        "graphql": "GraphQL",
        "graph ql": "GraphQL",
        "html5": "HTML",
        "html 5": "HTML",
        "css3": "CSS",
        "css 3": "CSS",
        # Cloud / DevOps
        "aws": "AWS",
        "amazon web services": "AWS",
        "gcp": "GCP",
        "google cloud": "GCP",
        "google cloud platform": "GCP",
        "azure": "Azure",
        "microsoft azure": "Azure",
        "docker container": "Docker",
        "k8s": "Kubernetes",
        "kubernetes": "Kubernetes",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "jenkins": "Jenkins",
        "gitlab ci": "GitLab CI",
        "github actions": "GitHub Actions",
        "ci cd": "CI/CD",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        # Tools
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "jira": "Jira",
        "confluence": "Confluence",
        "slack": "Slack",
        "figma": "Figma",
        "trello": "Trello",
        "asana": "Asana",
        "excel": "Excel",
        "power bi": "Power BI",
        "tableau": "Tableau",
        "airflow": "Apache Airflow",
    }

    # Ordering matters: longer, more specific forms are tried before short ones.
    SKILL_NORMALIZATION_ORDER = sorted(SKILL_SYNONYMS.keys(), key=len, reverse=True)

    # -----------------------------------------------------------------------
    # Degree normalization
    # -----------------------------------------------------------------------
    DEGREE_SYNONYMS: dict[str, str] = {
        "bachelor of technology": "B.Tech",
        "bachelor of engineering": "B.E.",
        "bachelor of science": "B.Sc",
        "bachelor of arts": "B.A.",
        "bachelor of commerce": "B.Com",
        "bachelor of computer applications": "BCA",
        "bachelor of business administration": "BBA",
        "bachelor of applied science": "B.A.S.",
        "bachelor degree": "Bachelor",
        "btech": "B.Tech",
        "b.tech": "B.Tech",
        "b.e.": "B.E.",
        "be": "B.E.",
        "b.e": "B.E.",
        "b.sc": "B.Sc",
        "bsc": "B.Sc",
        "b.com": "B.Com",
        "bcom": "B.Com",
        "bca": "BCA",
        "b.a.": "B.A.",
        "associate of arts": "A.A.",
        "associate of science": "A.S.",
        "associate of applied science": "A.A.S.",
        "associate degree": "Associate",
        "associate": "Associate",
        "master of technology": "M.Tech",
        "master of engineering": "M.E.",
        "master of science": "M.Sc",
        "master of arts": "M.A.",
        "master of commerce": "M.Com",
        "master of computer applications": "MCA",
        "master of business administration": "MBA",
        "master degree": "Master",
        "mtech": "M.Tech",
        "m.tech": "M.Tech",
        "m.e.": "M.E.",
        "me": "M.E.",
        "m.e": "M.E.",
        "m.sc": "M.Sc",
        "msc": "M.Sc",
        "m.com": "M.Com",
        "mcom": "M.Com",
        "mca": "MCA",
        "m.a.": "M.A.",
        "high school diploma": "High School Diploma",
        "diploma": "Diploma",
        "ged": "GED",
        "doctor of philosophy": "PhD",
        "doctorate": "PhD",
        "ph.d.": "PhD",
        "ph.d": "PhD",
    }

    DEGREE_NORMALIZATION_ORDER = sorted(DEGREE_SYNONYMS.keys(), key=len, reverse=True)

    # -----------------------------------------------------------------------
    # Location normalization
    # -----------------------------------------------------------------------
    LOCATION_SYNONYMS: dict[str, str] = {
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "mumbai": "Mumbai",
        "bombay": "Mumbai",
        "delhi": "Delhi",
        "new delhi": "New Delhi",
        "hyderabad": "Hyderabad",
        "pune": "Pune",
        "chennai": "Chennai",
        "madras": "Chennai",
        "kolkata": "Kolkata",
        "calcutta": "Kolkata",
        "nyc": "New York",
        "new york city": "New York",
        "ny": "New York",
        "california": "California",
        "ca": "California",
        "texas": "Texas",
        "tx": "Texas",
        "florida": "Florida",
        "fl": "Florida",
        "washington state": "Washington",
        "london": "London",
        "toronto": "Toronto",
        "vancouver": "Vancouver",
        "sydney": "Sydney",
        "melbourne": "Melbourne",
        "berlin": "Berlin",
        "paris": "Paris",
        "tokyo": "Tokyo",
        "singapore": "Singapore",
        "dubai": "Dubai",
        "remote": "Remote",
        "india": "India",
        "usa": "United States",
        "united states": "United States",
        "united states of america": "United States",
        "us": "United States",
        "uk": "UK",
        "united kingdom": "UK",
        "canada": "Canada",
    }

    LOCATION_NORMALIZATION_ORDER = sorted(LOCATION_SYNONYMS.keys(), key=len, reverse=True)

    # -----------------------------------------------------------------------
    # Canonical skill list (used for keyword extraction and normalization)
    # -----------------------------------------------------------------------
    CANONICAL_SKILLS: list[str] = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go",
        "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala", "R", "MATLAB", "Perl",
        "Lua", "React", "Angular", "Vue", "Svelte", "Next.js", "Nuxt.js", "Django",
        "Flask", "Spring", "Express", "FastAPI", "Rails", "Laravel", "ASP.NET",
        "Node.js", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "SQLite", "Oracle", "Cassandra", "DynamoDB", "Firebase", "Supabase",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
        "Jenkins", "GitLab CI", "GitHub Actions", "CI/CD", "Linux", "Ubuntu",
        "Windows", "ML", "DL", "AI", "TensorFlow", "PyTorch", "Keras",
        "Scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop", "Tableau", "Power BI",
        "Excel", "Apache Airflow", "Git", "GitHub", "GitLab", "Jira", "Confluence",
        "Slack", "Figma", "Trello", "Asana", "REST API", "GraphQL", "gRPC",
        "Microservices", "HTML", "CSS",
    ]

    # -----------------------------------------------------------------------
    # Normalization methods
    # -----------------------------------------------------------------------
    @classmethod
    def normalize_skill(cls, skill: str) -> str | None:
        """Return the canonical form of a single skill string."""
        if not skill or not isinstance(skill, str):
            return None
        lower = skill.strip().lower()
        if not lower:
            return None
        # Direct synonym hit
        for synonym in cls.SKILL_NORMALIZATION_ORDER:
            if lower == synonym:
                return cls.SKILL_SYNONYMS[synonym]
        # Substring hit (use word boundaries)
        for synonym in cls.SKILL_NORMALIZATION_ORDER:
            pattern = re.compile(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", re.IGNORECASE)
            if pattern.search(skill):
                return cls.SKILL_SYNONYMS[synonym]
        # No synonym mapping, title-case common lowercase tokens
        return cls._title_case_skill(skill)

    @staticmethod
    def _title_case_skill(skill: str) -> str:
        """Title-case a skill while preserving known acronyms."""
        if not skill:
            return skill
        skill = skill.strip()
        upper_words = {"ai", "ml", "dl", "sql", "api", "aws", "gcp", "css", "html", "rest", "grpc"}
        tokens = skill.split()
        normalized = []
        for t in tokens:
            t = t.strip("-.,;()")
            if not t:
                continue
            if t.lower() in upper_words:
                normalized.append(t.upper())
            else:
                normalized.append(t[0].upper() + t[1:] if len(t) > 1 else t.upper())
        return " ".join(normalized) if normalized else skill

    @classmethod
    def normalize_skills(cls, skills: list[str]) -> list[str]:
        """Normalize and deduplicate a list of skill strings."""
        if not skills:
            return []
        seen: set = set()
        out: list[str] = []
        for skill in skills:
            canonical = cls.normalize_skill(skill)
            if canonical and canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
        return out

    @classmethod
    def normalize_experience_years(cls, value: Any) -> float | None:
        """Convert free-text experience values to a numeric year count."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) if value else None
        text = str(value).strip()
        if not text:
            return None

        # Written numbers
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
        }
        for word, num in word_to_num.items():
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                return float(num)

        # Range pattern: 3-5 years -> take the higher bound
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years?", text, re.IGNORECASE)
        if range_match:
            return float(range_match.group(2))

        # Single number + years / yrs
        single_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
        if single_match:
            return float(single_match.group(1))

        # Number + months / mos -> convert to years
        months_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:months?|mos?)", text, re.IGNORECASE)
        if months_match:
            return round(float(months_match.group(1)) / 12, 2)

        # "Experience: 5"
        exp_match = re.search(r"experience\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if exp_match:
            return float(exp_match.group(1))

        # Plain number
        plain_match = re.search(r"(\d+(?:\.\d+)?)", text)
        if plain_match:
            return float(plain_match.group(1))

        return None

    @classmethod
    def normalize_education(cls, text: str | None) -> str | None:
        """Normalize a free-text education string into canonical degree form."""
        if not text:
            return None
        text = str(text).strip()
        if not text:
            return None
        lower = text.lower()
        for synonym in cls.DEGREE_NORMALIZATION_ORDER:
            pattern = re.compile(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", re.IGNORECASE)
            if pattern.search(text):
                degree = cls.DEGREE_SYNONYMS[synonym]
                return degree
        return text if text else None

    @classmethod
    def normalize_location(cls, text: str | None) -> str | None:
        """Normalize a location string."""
        if not text:
            return None
        text = str(text).strip()
        if not text:
            return None
        lower = text.lower()
        for synonym in cls.LOCATION_NORMALIZATION_ORDER:
            pattern = re.compile(r"(?<![\w.])" + re.escape(synonym) + r"(?![\w])", re.IGNORECASE)
            if pattern.search(text):
                return cls.LOCATION_SYNONYMS[synonym]
        return text[0].upper() + text[1:] if text else None

    @classmethod
    def clean_sentinel(cls, value: Any) -> Any:
        """Replace known placeholder/sentinel values with None or []."""
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [v for v in value if not cls.is_sentinel(v)]
            return cleaned
        if isinstance(value, str):
            if cls.is_sentinel(value):
                return None
            return value
        if isinstance(value, (int, float)):
            return value
        return value

    @classmethod
    def is_sentinel(cls, value: Any) -> bool:
        """Return True if a value is a known placeholder."""
        if value is None:
            return False
        if isinstance(value, str):
            lower = value.lower().strip()
            if not lower:
                return True
            return (
                lower.startswith("no_") or
                "_extracted" in lower or
                lower in ("unknown", "none", "not specified", "not provided", "n/a") or
                lower.startswith("no ")
            )
        return False
