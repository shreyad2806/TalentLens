"""Adapter for the synthetic Hugging Face candidate-matching dataset."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models import ResumeDocument, ResumeMetadata

from .base_adapter import BaseDatasetAdapter


def _coerce_list(value: Any) -> List[Any]:
    """Ensure a value is a proper Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [item.strip() for item in s.strip("[]").split(",") if item.strip()]
    return [value]


class SyntheticAdapter(BaseDatasetAdapter):
    """Converts `shreyad2806/candidate-matching-synthetic` resumes into `ResumeDocument`."""

    source_name = "synthetic"

    def __init__(self, source_path: Optional[str] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default = project_root / "data" / "structured" / "default_resumes.parquet"
        if not default.exists():
            default = project_root / "data" / "structured" / "default_resumes.csv"
        super().__init__(source_path or str(default))

    def load(self) -> List[Dict[str, Any]]:
        path = Path(self.source_path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        for col in ["skills", "experience_bullets"]:
            if col in df.columns:
                df[col] = df[col].apply(_coerce_list)

        return df.to_dict(orient="records")

    def validate(self, record: Dict[str, Any]) -> bool:
        return bool(record.get("resume_id")) and bool(record.get("summary", "").strip())

    def convert(self, record: Dict[str, Any]) -> ResumeDocument:
        resume_id = str(record["resume_id"])
        skills = _coerce_list(record.get("skills"))
        bullets = _coerce_list(record.get("experience_bullets"))

        pieces = [
            record.get("summary", "").strip(),
            f"Role: {record.get('role', '')}",
            f"Seniority: {record.get('seniority', '')}",
            f"Industry: {record.get('industry', '')}",
            f"Years of experience: {record.get('years_experience', '')}",
            f"Education: {record.get('education', '')}",
            "Skills: " + ", ".join(skills),
            "Experience:",
        ]
        for b in bullets:
            pieces.append(f"- {b}")

        metadata = ResumeMetadata(
            resume_id=resume_id,
            candidate_name=None,
            role=record.get("role") or None,
            skills=skills,
            location=None,
            experience_years=float(record.get("years_experience", 0)) if record.get("years_experience") else None,
            education=[record.get("education")] if record.get("education") else [],
            projects=[],
            certifications=[],
            email=None,
            phone=None,
            summary=record.get("summary"),
        )

        return ResumeDocument(
            candidate_id=resume_id,
            resume_text="\n\n".join(p for p in pieces if p),
            resume_metadata=metadata,
            source_dataset=self.source_name,
            metadata_confidence={},
            metadata_source={},
        )
