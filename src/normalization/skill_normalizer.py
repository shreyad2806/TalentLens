"""Canonical skill name normalizer with taxonomy and noise filtering.

Responsibilities:
1. Map vendor/brand variants to one canonical name (MS Excel -> Excel).
2. Reject soft-skill fluff and sentence fragments (never displayed as skills).
3. Keep only meaningful technical or business skills.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class SkillNormalizer:
    """
    Normalize skill name variants into a canonical taxonomy.

    Examples:
        - "MS Excel" / "Microsoft Excel" / "Advanced Excel" -> "Excel"
        - "NodeJS" -> "Node.js"
        - "ReactJS" -> "React"
        - "Excellent Written Communication" -> rejected (None)
        - "Motivated" -> rejected (None)
    """

    # Alias -> canonical display name.  Keys are lowercase, whitespace-normalized.
    _MAPPING = {
        # Microsoft Office family
        "ms excel": "Excel",
        "msexcel": "Excel",
        "microsoft excel": "Excel",
        "advanced excel": "Excel",
        "excel": "Excel",
        "ms word": "Word",
        "microsoft word": "Word",
        "word": "Word",
        "ms powerpoint": "PowerPoint",
        "microsoft powerpoint": "PowerPoint",
        "power point": "PowerPoint",
        "powerpoint": "PowerPoint",
        "ms office": "Microsoft Office",
        "microsoft office": "Microsoft Office",
        "ms access": "Access",
        "microsoft access": "Access",
        "ms outlook": "Outlook",
        "microsoft outlook": "Outlook",
        "outlook": "Outlook",
        # Programming / frameworks
        "nodejs": "Node.js",
        "node js": "Node.js",
        "node.js": "Node.js",
        "reactjs": "React",
        "react js": "React",
        "react": "React",
        "angularjs": "Angular",
        "angular js": "Angular",
        "vuejs": "Vue",
        "vue js": "Vue",
        "js": "JavaScript",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "ts": "TypeScript",
        "py": "Python",
        "python": "Python",
        "python3": "Python",
        "java": "Java",
        "c sharp": "C#",
        "c#": "C#",
        "cpp": "C++",
        "c plus plus": "C++",
        "c++": "C++",
        "golang": "Go",
        "go lang": "Go",
        "html5": "HTML",
        "html": "HTML",
        "css3": "CSS",
        "css": "CSS",
        # Data / ML
        "pytorch": "PyTorch",
        "torch": "PyTorch",
        "tensor flow": "TensorFlow",
        "tensorflow": "TensorFlow",
        "machine learning": "Machine Learning",
        "ml": "Machine Learning",
        "artificial intelligence": "AI",
        "ai": "AI",
        "deep learning": "Deep Learning",
        "dl": "Deep Learning",
        "natural language processing": "NLP",
        "nlp": "NLP",
        "power bi": "Power BI",
        "powerbi": "Power BI",
        "tableau": "Tableau",
        "sql": "SQL",
        "mysql": "MySQL",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "mongo": "MongoDB",
        "mongodb": "MongoDB",
        # Cloud / DevOps
        "amazon web services": "AWS",
        "aws": "AWS",
        "google cloud platform": "GCP",
        "google cloud": "GCP",
        "gcp": "GCP",
        "microsoft azure": "Azure",
        "azure": "Azure",
        "k8s": "Kubernetes",
        "kubernetes": "Kubernetes",
        "docker": "Docker",
        "ci cd": "CI/CD",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        "git": "Git",
        "github": "GitHub",
        # Enterprise / business
        "sap": "SAP",
        "erp": "ERP",
        "crm": "CRM",
        "salesforce": "Salesforce",
        "quickbooks": "QuickBooks",
        "quick books": "QuickBooks",
        "jira": "Jira",
        "rest api": "REST API",
        "restful api": "REST API",
        "rest apis": "REST API",
        "spring boot": "Spring Boot",
        "springboot": "Spring Boot",
        "spring": "Spring",
    }

    # Fluff tokens: any skill containing one of these words (as a token) is
    # dropped unless it is an exact taxonomy alias.
    _NOISE_TOKENS = {
        "excellent", "written", "oral", "verbal", "communication", "communications",
        "self", "starter", "motivated", "hardworking", "hard", "working",
        "dedicated", "responsible", "reliable", "punctual", "honest", "sincere",
        "passionate", "enthusiastic", "positive", "attitude", "interpersonal",
        "skills", "skill", "ability", "abilities", "proficient", "proficiency",
        "knowledge", "good", "great", "strong", "outstanding", "solid", "proven",
        "quick", "fast", "learner", "adaptable", "flexible", "detail", "oriented",
        "player", "leadership", "organizational", "multitasking", "listener",
        "personality", "confident", "smart", "energetic", "creative",
        "team", "teamwork",
    }

    # Connector / sentence words: their presence marks a sentence fragment,
    # not a skill name (e.g. "Environment Through Effectively Managing").
    _CONNECTOR_TOKENS = {
        "and", "or", "the", "a", "an", "of", "to", "in", "for", "with",
        "by", "on", "at", "through", "via", "from", "into", "as", "that",
        "which", "using", "effectively", "efficiently", "successfully",
    }

    _EXAMPLES = [
        ("MS Excel", "Excel"),
        ("Microsoft Excel", "Excel"),
        ("NodeJS", "Node.js"),
        ("ReactJS", "React"),
        ("Tensor Flow", "TensorFlow"),
        ("Excellent Written Communication", None),
        ("Motivated", None),
        ("Hardworking", None),
        ("Self Starter", None),
        ("Communication Skills", None),
    ]

    @classmethod
    def _key(cls, skill: str) -> str:
        return re.sub(r"[-_\s]+", " ", skill.strip()).lower()

    @classmethod
    def normalize(cls, skill: str | None) -> str | None:
        """Return the canonical skill name, or None if the input is noise."""
        if not skill or not skill.strip():
            return None
        raw = skill.strip()
        key = cls._key(raw)

        # 1. Exact taxonomy hit always wins.
        if key in cls._MAPPING:
            canonical = cls._MAPPING[key]
            logger.info("Skill aliased: %r -> %r", raw, canonical)
            return canonical

        # 2. Reject sentence fragments: too many words or too long.
        words = key.split()
        if len(words) > 4 or len(raw) > 40:
            logger.info("Skill rejected (too long/fragment): %r", raw)
            return None

        # 3. Reject fluff / soft-skill phrases.
        if set(words) & cls._NOISE_TOKENS:
            logger.info("Skill rejected (noise): %r", raw)
            return None

        # 3b. Reject sentence fragments containing connector words.
        if set(words) & cls._CONNECTOR_TOKENS:
            logger.info("Skill rejected (connector): %r", raw)
            return None

        # 4. Reject fragments with sentence punctuation or verbs-like endings.
        if re.search(r"[.!?;:]", raw):
            logger.info("Skill rejected (punctuation): %r", raw)
            return None

        # 5. Keep as-is but title-case for display (preserve acronyms).
        display = " ".join(
            w if (w.isupper() and len(w) > 1) else w.capitalize()
            for w in raw.split()
        )
        logger.info("Skill kept: %r -> %r", raw, display)
        return display

    @classmethod
    def normalize_list(cls, skills: list[str]) -> list[str]:
        """Normalize a skill list: canonical names, noise removed, deduplicated."""
        out: list[str] = []
        seen: set[str] = set()
        for s in skills or []:
            norm = cls.normalize(s)
            if norm is None:
                continue
            k = norm.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(norm)
        return out

    @classmethod
    def examples(cls) -> list[tuple[str, str | None]]:
        return cls._EXAMPLES[:]
