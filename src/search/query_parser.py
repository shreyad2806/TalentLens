"""Query Understanding: extract structured search intent from a natural-language query."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.normalization.skill_normalizer import SkillNormalizer

# Role nouns that terminate a role phrase, e.g. "Finance Manager", "Python Developer".
_ROLE_NOUNS = {
    "manager", "developer", "engineer", "analyst", "scientist", "designer",
    "consultant", "accountant", "architect", "administrator", "specialist",
    "lead", "officer", "executive", "director", "teacher", "nurse", "advocate",
    "professional", "intern", "associate", "coordinator", "supervisor", "head",
}

# Industry keywords -> display industry.
_INDUSTRY_MAP = {
    "finance": "Finance", "financial": "Finance", "banking": "Banking",
    "accounting": "Finance", "insurance": "Insurance", "fintech": "Finance",
    "healthcare": "Healthcare", "medical": "Healthcare", "clinical": "Healthcare",
    "pharma": "Healthcare", "biotech": "Healthcare",
    "construction": "Construction", "civil": "Construction", "architecture": "Construction",
    "marketing": "Marketing", "sales": "Sales", "advertising": "Marketing", "branding": "Marketing",
    "retail": "Retail", "ecommerce": "Retail", "logistics": "Logistics",
    "education": "Education", "teaching": "Education", "academic": "Education",
    "legal": "Legal", "law": "Legal", "hr": "Human Resources", "consulting": "Consulting",
    "manufacturing": "Manufacturing", "production": "Manufacturing", "operations": "Operations",
    "hospitality": "Hospitality", "hotel": "Hospitality", "travel": "Travel", "tourism": "Travel",
    "it": "Information Technology", "software": "Information Technology",
    "technology": "Information Technology", "agriculture": "Agriculture",
    "aviation": "Aviation", "automobile": "Automotive", "automotive": "Automotive",
    "energy": "Energy", "fitness": "Fitness", "media": "Media", "fashion": "Fashion",
}

# Business/domain skills that are valid query skills even though they are
# also industry words (e.g. "Banking experience" -> skill Banking).
_BUSINESS_SKILLS = {
    "banking", "accounting", "budgeting", "forecasting", "auditing", "taxation",
    "payroll", "invoicing", "reconciliation", "compliance", "underwriting",
    "sales", "marketing", "negotiation", "procurement", "logistics",
    "financial analysis", "financial planning", "risk management",
    "project management", "business analysis", "data analysis",
}

# Education keywords -> canonical display.
_EDUCATION_MAP = {
    "mba": "MBA", "bba": "BBA", "b.com": "B.Com", "bcom": "B.Com",
    "m.com": "M.Com", "mcom": "M.Com", "b.tech": "B.Tech", "btech": "B.Tech",
    "m.tech": "M.Tech", "mtech": "M.Tech", "bsc": "B.Sc", "msc": "M.Sc",
    "bachelor": "Bachelor's", "bachelors": "Bachelor's",
    "master": "Master's", "masters": "Master's", "phd": "PhD", "doctorate": "PhD",
    "cpa": "CPA", "ca": "CA", "cfa": "CFA",
}

_KNOWN_LOCATIONS = {
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai",
    "kolkata", "noida", "gurgaon", "india", "remote",
    "new york", "california", "texas", "florida", "washington", "chicago",
    "boston", "seattle", "london", "toronto", "vancouver", "sydney", "singapore",
    "dubai", "usa", "uk",
}

_STOPWORDS = {
    "with", "and", "or", "the", "a", "an", "in", "for", "of", "to",
    "experience", "experienced", "years", "year", "skills", "skill",
    "knowledge", "background", "expertise", "having", "who", "has",
    "candidate", "candidates", "resume", "resumes", "find", "search",
    "looking", "need", "want", "required", "senior", "junior",
}

# Configurable role dictionary: raw lower phrase -> canonical display role.
_ROLE_TEMPLATES = {
    "frontend developer": "Frontend Developer",
    "frontend engineer": "Frontend Engineer",
    "backend developer": "Backend Developer",
    "backend engineer": "Backend Engineer",
    "full stack developer": "Full Stack Developer",
    "full stack engineer": "Full Stack Engineer",
    "fullstack developer": "Full Stack Developer",
    "fullstack engineer": "Full Stack Engineer",
    "mobile developer": "Mobile Developer",
    "mobile engineer": "Mobile Engineer",
    "android developer": "Mobile Developer",
    "ios developer": "iOS Developer",
    "data scientist": "Data Scientist",
    "data analyst": "Data Analyst",
    "software engineer": "Software Engineer",
    "software developer": "Software Developer",
    "web developer": "Web Developer",
    "devops engineer": "DevOps Engineer",
    "machine learning engineer": "Machine Learning Engineer",
    "ml engineer": "Machine Learning Engineer",
    "ai engineer": "AI Engineer",
    "data engineer": "Data Engineer",
    "cloud engineer": "Cloud Engineer",
    "site reliability engineer": "Site Reliability Engineer",
    "sre": "Site Reliability Engineer",
}

# Tokens that are programming languages, frameworks, or platforms and must
# never be merged into the role phrase.
_ROLE_FRAMEWORK_TOKENS = {
    k for k in SkillNormalizer._MAPPING if " " not in k
} | {
    "reactjs", "vuejs", "angularjs", "nodejs", "node", "springboot",
    "spring", "hibernate", "jquery", "bootstrap", "net", ".net", "dotnet",
    "laravel", "rails", "ruby", "php", "golang", "rust", "scala", "perl",
    "mongodb", "postgres", "postgresql", "mysql", "sqlite", "redis",
    "oracle", "cassandra", "dynamodb", "elasticsearch", "kafka", "rabbitmq",
    "graphql", "rest", "soap", "oauth", "jwt", "json", "xml", "yaml",
    "html", "html5", "css", "css3", "sass", "less", "webpack", "babel",
    "eslint", "prettier", "npm", "yarn", "pip", "maven", "gradle", "cmake",
    "jenkins", "gitlab", "github", "bitbucket", "circleci", "travis",
    "terraform", "ansible", "puppet", "chef", "vagrant", "nginx", "apache",
    "tomcat", "jetty", "kubernetes", "docker", "aws", "gcp", "azure",
    "firebase", "heroku", "netlify", "vercel", "digitalocean", "linode",
}


@dataclass
class ParsedQuery:
    """Structured search intent extracted from a natural-language query."""

    raw_query: str = ""
    role: str | None = None
    industry: str | None = None
    skills: list[str] = field(default_factory=list)
    experience_min: float | None = None
    experience_max: float | None = None
    education: str | None = None
    location: str | None = None

    def display_dict(self) -> dict[str, str]:
        """Human-readable mapping for the UI (Not specified when missing)."""
        return {
            "Role": self.role or "Not specified",
            "Industry": self.industry or "Not specified",
            "Skills": ", ".join(self.skills) if self.skills else "Not specified",
            "Experience": (
                f"{self.experience_min:g}+ years" if self.experience_min is not None and self.experience_max is None
                else f"{self.experience_min:g}-{self.experience_max:g} years" if self.experience_min is not None
                else "Not specified"
            ),
            "Education": self.education or "Not specified",
            "Location": self.location or "Not specified",
        }


class QueryParser:
    """Extract structured intent (role, industry, skills, experience, education, location)."""

    def parse(self, query: str) -> ParsedQuery:
        parsed = ParsedQuery(raw_query=query)
        if not query or not query.strip():
            return parsed

        text = query.strip()
        lower = re.sub(r"\s+", " ", text.lower())

        parsed.role = self._extract_role(lower)
        parsed.industry = self._extract_industry(lower, parsed.role)
        parsed.skills = self._extract_skills(lower)
        parsed.experience_min, parsed.experience_max = self._extract_experience(lower)
        parsed.education = self._extract_education(lower)
        parsed.location = self._extract_location(lower)
        return parsed

    def _extract_role(self, lower: str) -> str | None:
        # 1. Explicit role templates take priority.
        for pattern, canonical in sorted(_ROLE_TEMPLATES.items(), key=lambda x: -len(x[0])):
            if re.search(rf"(?<![\w]){re.escape(pattern)}(?![\w])", lower):
                return canonical

        # 2. Fallback: find a role noun and strip leading framework/tech tokens.
        words = lower.split()
        for i, w in enumerate(words):
            token = w.strip(".,")
            if token in _ROLE_NOUNS:
                start = i
                for j in range(i - 1, max(-1, i - 3), -1):
                    prev = words[j].strip(".,")
                    if prev in _STOPWORDS or prev in _EDUCATION_MAP or prev in _ROLE_FRAMEWORK_TOKENS:
                        break
                    start = j
                phrase = " ".join(words[start:i + 1])
                return _ROLE_TEMPLATES.get(phrase, phrase.title())
        return None

    def _extract_industry(self, lower: str, role: str | None) -> str | None:
        # Prefer industry hints inside the role phrase, then the whole query.
        candidates: list[str] = []
        if role:
            candidates.extend(role.lower().split())
        candidates.extend(lower.split())
        for token in candidates:
            token = token.strip(".,")
            if token in _INDUSTRY_MAP:
                return _INDUSTRY_MAP[token]
        return None

    def _extract_skills(self, lower: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()

        # Multi-word taxonomy aliases and business skills first (longest match).
        multi = [k for k in list(SkillNormalizer._MAPPING) + list(_BUSINESS_SKILLS) if " " in k]
        for alias in sorted(multi, key=len, reverse=True):
            if re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", lower):
                norm = SkillNormalizer.normalize(alias) or alias.title()
                if norm.lower() not in seen:
                    seen.add(norm.lower())
                    found.append(norm)

        # Single tokens.
        for token in re.findall(r"[a-zA-Z0-9+#./]+", lower):
            token = token.strip(".,")
            if token in _STOPWORDS or len(token) < 2:
                continue
            if token in SkillNormalizer._MAPPING or token in _BUSINESS_SKILLS:
                norm = SkillNormalizer.normalize(token) or token.title()
                if norm.lower() not in seen:
                    seen.add(norm.lower())
                    found.append(norm)
        return found

    def _extract_experience(self, lower: str) -> tuple[float | None, float | None]:
        m = re.search(r"(\d+)\s*(?:\+|plus)\s*years?", lower)
        if m:
            return float(m.group(1)), None
        m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*years?", lower)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r"(?:at least|minimum|min)\s*(\d+)\s*years?", lower)
        if m:
            return float(m.group(1)), None
        m = re.search(r"(\d+)\s*years?", lower)
        if m:
            return float(m.group(1)), None
        return None, None

    def _extract_education(self, lower: str) -> str | None:
        for token in re.findall(r"[a-zA-Z.]+", lower):
            token = token.strip(".,") if token not in _EDUCATION_MAP else token
            if token in _EDUCATION_MAP:
                return _EDUCATION_MAP[token]
        return None

    def _extract_location(self, lower: str) -> str | None:
        # Multi-word locations first.
        for loc in sorted(_KNOWN_LOCATIONS, key=len, reverse=True):
            if re.search(rf"(?<![\w]){re.escape(loc)}(?![\w])", lower):
                return loc.title()
        return None


_QUERY_PARSER: QueryParser | None = None


def get_query_parser() -> QueryParser:
    """Return the singleton QueryParser, creating it on first call."""
    global _QUERY_PARSER
    if _QUERY_PARSER is None:
        _QUERY_PARSER = QueryParser()
    return _QUERY_PARSER
