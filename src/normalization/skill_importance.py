"""Skill importance ranking: primary vs secondary vs generic.

Given a candidate's canonical skill list and role family, classify skills into:
- primary: technologies central to the occupation
- secondary: supporting technologies
- generic: non-technical / transferable skills (ignored in dashboard display)

All skills are retained for internal retrieval; this only affects the dashboard view.
"""

from __future__ import annotations

from .skill_normalizer import SkillNormalizer


class SkillImportanceRanker:
    """Rank and partition skills by their centrality to a role family."""

    # Generic / soft / productivity skills that should not be treated as primary.
    # These are still kept in the full skill list for retrieval.
    _GENERIC_RAW = {
        "Microsoft Office", "Word", "PowerPoint", "Outlook", "Access",
        "Google Docs", "Google Sheets", "Gmail", "Teams", "Slack", "Zoom", "Skype",
        "Communication", "Leadership", "Teamwork", "Time Management", "Problem Solving",
        "Critical Thinking", "Interpersonal Skills", "Attention to Detail", "Adaptability",
        "Organization", "Multitasking", "Customer Service", "Public Speaking", "Negotiation",
        "English", "Spanish", "Hindi", "Mandarin", "French", "German", "Italian",
        "Portuguese", "Japanese", "Korean", "Russian", "Arabic", "Bengali", "Tamil",
        "Telugu", "Urdu", "Punjabi", "Marathi", "Gujarati", "Kannada", "Malayalam",
        "Odia", "Assamese", "Management", "Planning", "Scheduling", "Coordination",
    }

    GENERIC_SKILL_KEYS = {SkillNormalizer._key(s) for s in _GENERIC_RAW}

    # Ordered primary technology lists per role family (most central first).
    PRIMARY_BY_FAMILY: dict[str, list[str]] = {
        "Machine Learning / AI": [
            "Python", "Machine Learning", "TensorFlow", "PyTorch", "Deep Learning",
            "NLP", "Computer Vision", "AI", "Scikit-learn", "NumPy", "Pandas",
            "Data Science", "Statistics", "Keras", "Reinforcement Learning",
            "Neural Networks", "Matplotlib", "Seaborn", "XGBoost", "LightGBM",
            "SQL", "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Git", "CI/CD",
        ],
        "Software Engineering": [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
            "Ruby", "PHP", "Swift", "Kotlin", "HTML", "CSS", "React", "Angular",
            "Vue", "Node.js", "Django", "Flask", "Spring", "FastAPI", "SQL",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Docker", "Kubernetes",
            "AWS", "GCP", "Azure", "Git", "CI/CD",
        ],
        "Data / Analytics": [
            "Python", "SQL", "R", "Machine Learning", "Statistics", "Data Analysis",
            "Data Visualization", "Tableau", "Power BI", "Pandas", "NumPy",
            "Matplotlib", "Seaborn", "Excel", "Spark", "Hadoop", "Databricks",
            "Snowflake", "AWS", "GCP", "Azure",
        ],
        "Research": [
            "Python", "R", "MATLAB", "Machine Learning", "Deep Learning", "Statistics",
            "Data Analysis", "LaTeX", "TensorFlow", "PyTorch", "NLP", "Computer Vision",
            "Research", "NumPy", "Pandas", "AWS", "GCP", "Azure",
        ],
        "Product / Program / Project Management": [
            "Product Management", "Agile", "Scrum", "Jira", "Confluence", "Roadmap",
            "OKRs", "Project Management", "Product Strategy", "User Research",
            "A/B Testing", "Data Analysis", "SQL", "Excel",
        ],
        "Consulting": [
            "Consulting", "Strategy", "Data Analysis", "Excel", "Power BI", "Tableau",
            "SQL", "Python", "Project Management", "Stakeholder Management",
        ],
        "Finance": [
            "Excel", "Financial Modeling", "Valuation", "Bloomberg", "SQL", "Python",
            "R", "Tableau", "Power BI", "Accounting", "QuickBooks", "Data Analysis",
            "Statistics",
        ],
        "Marketing": [
            "SEO", "SEM", "Google Analytics", "Google Ads", "Facebook Ads",
            "LinkedIn Ads", "HubSpot", "Mailchimp", "Content Marketing",
            "Social Media Marketing", "Marketing Strategy", "Brand Management",
            "Copywriting",
        ],
        "Sales": [
            "Salesforce", "HubSpot", "CRM", "Sales", "Business Development",
            "Negotiation", "Lead Generation", "Cold Calling",
        ],
        "Human Resources": [
            "Workday", "BambooHR", "Greenhouse", "Recruiting", "Talent Acquisition",
            "Performance Management", "HR", "Onboarding",
        ],
        "Healthcare": [
            "Clinical Research", "Electronic Health Records", "EPIC", "Cerner",
            "Medical Coding", "HL7", "HIPAA", "Data Analysis", "R", "Python",
        ],
        "Agriculture": [
            "Agriculture", "Agronomy", "Farming", "Soil Science", "Crop Management",
            "Precision Agriculture", "GIS", "Drones", "Data Analysis", "Python",
        ],
        "Engineering": [
            "AutoCAD", "SolidWorks", "CATIA", "MATLAB", "ANSYS", "CAD", "CAM",
            "Python", "C++", "Data Analysis", "Project Management",
        ],
    }

    # Build key -> family -> index lookups for stable ordering.
    _PRIMARY_INDEX: dict[str, dict[str, int]] = {}

    @classmethod
    def _init_indices(cls) -> None:
        if cls._PRIMARY_INDEX:
            return
        for family, skills in cls.PRIMARY_BY_FAMILY.items():
            cls._PRIMARY_INDEX[family] = {
                SkillNormalizer._key(s): i for i, s in enumerate(skills)
            }

    @classmethod
    def _skill_key(cls, skill: str) -> str:
        return SkillNormalizer._key(skill)

    @classmethod
    def is_generic(cls, skill: str) -> bool:
        """Return True if this canonical skill is considered generic."""
        return cls._skill_key(skill) in cls.GENERIC_SKILL_KEYS

    @classmethod
    def rank(
        cls,
        skills: list[str],
        role_family: str | None = None,
        primary_role: str | None = None,
    ) -> dict[str, list[str]]:
        """Partition skills into primary, secondary, and generic for the given role.

        If no role family is known, all non-generic skills are treated as primary.
        """
        cls._init_indices()

        primary: list[str] = []
        secondary: list[str] = []
        generic: list[str] = []

        family = (role_family or "").strip() or None
        family_index = cls._PRIMARY_INDEX.get(family) if family else None

        for skill in skills:
            k = cls._skill_key(skill)
            if k in cls.GENERIC_SKILL_KEYS:
                generic.append(skill)
                continue

            if family_index is not None:
                if k in family_index:
                    primary.append(skill)
                else:
                    secondary.append(skill)
            else:
                # No family context: surface all non-generic skills as primary.
                primary.append(skill)

        # Primary skills are ordered by centrality (family list order, then alpha).
        if family_index is not None:
            primary.sort(key=lambda s: family_index.get(cls._skill_key(s), 10_000))
        else:
            primary.sort()
        secondary.sort()
        generic.sort()

        return {
            "primary": primary,
            "secondary": secondary,
            "generic": generic,
            "all_skills": skills,
        }
