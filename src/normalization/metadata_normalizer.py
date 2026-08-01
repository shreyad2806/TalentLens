"""Orchestrates all field normalizers on a ``ResumeDocument``."""

from __future__ import annotations

from typing import List

from src.models import Education, ResumeDocument

from .degree_normalizer import DegreeNormalizer
from .experience_normalizer import ExperienceNormalizer
from .location_normalizer import LocationNormalizer
from .role_normalizer import RoleNormalizer
from .skill_normalizer import SkillNormalizer


class MetadataNormalizer:
    """
    Applies all field normalizers to a ``ResumeDocument``.

    This is the only normalization entry point the downstream pipeline needs.
    It returns a new ``ResumeDocument`` with canonical role, degree, skills,
    location, and experience values.
    """

    @classmethod
    def normalize(cls, doc: ResumeDocument) -> ResumeDocument:
        normalized_education: List[Education] = []
        for edu in doc.education:
            normalized_education.append(
                Education(
                    degree=DegreeNormalizer.normalize(edu.degree),
                    field=edu.field,
                    university=edu.university,
                    graduation_year=edu.graduation_year,
                )
            )

        return doc.model_copy(
            update={
                "role": RoleNormalizer.normalize(doc.role),
                "skills": SkillNormalizer.normalize_list(doc.skills),
                "education": normalized_education,
                "location": LocationNormalizer.normalize(doc.location),
                "experience_years": ExperienceNormalizer.normalize(doc.experience_years),
            }
        )

    @classmethod
    def normalize_all(cls, docs: List[ResumeDocument]) -> List[ResumeDocument]:
        return [cls.normalize(doc) for doc in docs]
