"""Canonical resume document model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel, Field, computed_field

from .resume_metadata import ResumeMetadata


class ResumeDocument(BaseModel):
    """
    Unified resume document.

    Contains only:
      - the candidate's resume text
      - the single canonical ResumeMetadata extracted once by the parser
      - source / quality / timing information

    All candidate-facing fields (name, skills, location, etc.) live inside
    ResumeMetadata and are exposed as computed fields for convenience.
    """

    candidate_id: str = Field(..., description="Stable identifier used for lookup")
    resume_text: str = Field(..., min_length=1, description="Raw resume text")
    resume_metadata: ResumeMetadata = Field(..., description="Single canonical metadata object")
    source_dataset: str = Field(..., description="Dataset / adapter this resume came from")
    metadata_confidence: Dict[str, float] = Field(default_factory=dict)
    metadata_source: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeDocument":
        return cls.model_validate(data)

    @computed_field
    @property
    def candidate_name(self) -> Any:
        return self.resume_metadata.candidate_name

    @computed_field
    @property
    def role(self) -> Any:
        return self.resume_metadata.role

    @computed_field
    @property
    def skills(self) -> Any:
        return self.resume_metadata.skills

    @computed_field
    @property
    def location(self) -> Any:
        return self.resume_metadata.location

    @computed_field
    @property
    def experience_years(self) -> Any:
        return self.resume_metadata.experience_years

    @computed_field
    @property
    def education(self) -> Any:
        return self.resume_metadata.education

    @computed_field
    @property
    def projects(self) -> Any:
        return self.resume_metadata.projects

    @computed_field
    @property
    def certifications(self) -> Any:
        return self.resume_metadata.certifications

    @computed_field
    @property
    def email(self) -> Any:
        return self.resume_metadata.email

    @computed_field
    @property
    def phone(self) -> Any:
        return self.resume_metadata.phone

    @computed_field
    @property
    def summary(self) -> Any:
        return self.resume_metadata.summary
