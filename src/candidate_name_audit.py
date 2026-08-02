"""Candidate-name quality audit.

Scans the indexed production dataset, validates every candidate_name,
and exports a CSV report plus a console summary.

Usage:
    python -m src.candidate_name_audit
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResumeDocument
from src.resume_parser.name_validator import is_valid_candidate_name


DATASET_PATH = PROJECT_ROOT / "combined" / "production_dataset.json"
REPORT_PATH = PROJECT_ROOT / "candidate_name_quality_report.csv"


def _load_resumes() -> list[ResumeDocument]:
    """Load the production dataset if it exists."""
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    resumes: list[ResumeDocument] = []
    for raw in data:
        try:
            resumes.append(ResumeDocument.model_validate(raw))
        except Exception:
            continue
    return resumes


def _filename_from_resume(resume: ResumeDocument) -> str:
    """Return the original filename if available."""
    return resume.metadata_source.get("source_filename", "") or resume.source_dataset or ""


def _source_from_resume(resume: ResumeDocument) -> str:
    """Return the extraction source for candidate_name if recorded."""
    return resume.metadata_source.get("candidate_name", "unknown")


def run_audit() -> dict[str, Any]:
    """Run the audit and write the CSV report."""
    resumes = _load_resumes()
    total = len(resumes)
    if total == 0:
        return {"total": 0}

    rows = []
    valid_count = 0
    fallback_count = 0
    filename_fallback_count = 0
    unknown_count = 0
    reasons: Counter = Counter()
    rejected_headings: Counter = Counter()

    for resume in resumes:
        current_name = resume.resume_metadata.candidate_name or ""
        filename = _filename_from_resume(resume)
        source = _source_from_resume(resume)

        valid, reason = is_valid_candidate_name(current_name, reason=True)
        if valid:
            valid_count += 1
        elif current_name == "" or current_name.lower() == "unknown":
            unknown_count += 1
            reasons[reason] += 1
        elif source == "filename":
            filename_fallback_count += 1
            reasons[reason] += 1
        else:
            fallback_count += 1
            reasons[reason] += 1

        if not valid and current_name:
            rejected_headings[current_name] += 1

        rows.append({
            "resume_id": resume.resume_metadata.resume_id,
            "filename": filename,
            "current_name": current_name,
            "detected_source": source,
            "valid": valid,
            "reason_if_invalid": reason if not valid else "",
        })

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    top_rejected = rejected_headings.most_common(10)
    quality_pct = round((valid_count / total) * 100, 2) if total else 0.0

    summary = {
        "total": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "valid_pct": quality_pct,
        "unknown": unknown_count,
        "filename_fallbacks": filename_fallback_count,
        "other_fallbacks": fallback_count,
        "reasons": dict(reasons),
        "top_rejected_headings": top_rejected,
        "report_path": str(REPORT_PATH),
    }

    print("\n=== Candidate Name Quality Audit ===")
    print(f"Total resumes:          {total}")
    print(f"Valid names:            {valid_count} ({quality_pct}%)")
    print(f"Invalid / fallback:     {total - valid_count}")
    print(f"  - Unknown/empty:      {unknown_count}")
    print(f"  - Filename fallback:  {filename_fallback_count}")
    print(f"  - Other fallback:     {fallback_count}")
    print(f"Top rejected values:    {top_rejected}")
    print(f"CSV report written to:  {REPORT_PATH}\n")

    return summary


if __name__ == "__main__":
    run_audit()
