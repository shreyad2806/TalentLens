"""Adapter for the Kaggle-style Resume.csv corpus."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from src.models import ResumeDocument, ResumeMetadata
from src.resume_parser.parser_service import ParserService

from .base_adapter import BaseDatasetAdapter


class KaggleAdapter(BaseDatasetAdapter):
    """
    Converts the original TalentLens `Resume/Resume.csv` into `ResumeDocument`.

    The source CSV only has `ID`, `Resume_str`, `Resume_html`, and `Category`.
    The parser extracts a single `ResumeMetadata` object once; it is passed
    unchanged into the `ResumeDocument` and then through the rest of the pipeline.
    """

    source_name = "kaggle"

    def __init__(self, source_path: str | None = None) -> None:
        default = Path(__file__).resolve().parents[2] / "Resume" / "Resume.csv"
        super().__init__(source_path or str(default))

    def load(self) -> list[dict[str, Any]]:
        with open(self.source_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def validate(self, record: dict[str, Any]) -> bool:
        """Require an ID and non-empty resume text."""
        if not record.get("ID"):
            return False
        text = record.get("Resume_str", "").strip()
        if not text and record.get("Resume_html"):
            text = self._html_to_text(record.get("Resume_html", ""))
        return bool(text)

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML resume content to plain text when Resume_str is empty."""
        if not html:
            return ""
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def convert(self, record: dict[str, Any]) -> ResumeDocument:
        raw_text = record.get("Resume_str", "").strip()
        if not raw_text and record.get("Resume_html"):
            raw_text = self._html_to_text(record.get("Resume_html", ""))

        parser = ParserService()
        parsed = parser.parse_text(raw_text, record)

        resume_id = str(record["ID"])

        # role: CSV Category is the most reliable signal for this dataset.
        role = record.get("Category") or None
        if not role and parsed.experience:
            role = parsed.experience[0].title

        # Experience years from the parser, with a sensible cap.
        experience_years = None
        if parsed.metadata:
            raw_years = parsed.metadata.get("total_experience_years")
            if raw_years is not None:
                try:
                    experience_years = float(raw_years)
                    if not (0 <= experience_years <= 60):
                        experience_years = None
                except (ValueError, TypeError):
                    experience_years = None

        education = []
        for edu in parsed.education or []:
            parts = [p for p in [edu.degree, edu.field_of_study, edu.institution] if p]
            education.append(" ".join(parts))

        metadata = ResumeMetadata(
            resume_id=resume_id,
            candidate_name=parsed.name,
            role=role,
            skills=parsed.skills or [],
            location=(parsed.metadata or {}).get("location"),
            experience_years=experience_years,
            education=education,
            projects=[p.name for p in (parsed.projects or []) if p.name],
            certifications=[c.name for c in (parsed.certifications or []) if c.name],
            email=parsed.email,
            phone=parsed.phone,
            summary=parsed.summary,
        )

        field_confidence = (parsed.metadata or {}).get("field_confidence", {})
        field_source = (parsed.metadata or {}).get("field_source", {})
        if record.get("Category"):
            field_confidence["role"] = 1.0
            field_source["role"] = "csv_category"

        return ResumeDocument(
            candidate_id=resume_id,
            resume_text=raw_text,
            resume_metadata=metadata,
            source_dataset=self.source_name,
            metadata_confidence=field_confidence,
            metadata_source=field_source,
        )
