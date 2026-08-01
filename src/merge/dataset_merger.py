"""Multi-dataset merger that assembles resumes into one production corpus."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dataset import AdapterFactory
from src.models import ResumeDocument


def _maybe_add_project_root() -> None:
    """Ensure the project root is on PYTHONPATH for imports."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


class DatasetMerger:
    """
    Merges multiple resume datasets into a single validated ``ResumeDocument`` corpus.

    Pipeline:
        Adapter -> ResumeDocument -> validate -> deduplicate -> production_dataset.json

    The merger preserves ``source_dataset`` and ``metadata_confidence`` so the
    vector-store / QA layers can trace each resume back to its origin.
    """

    def __init__(
        self,
        dataset_configs: List[Tuple[str, Optional[str]]],
        output_path: Optional[str] = None,
    ) -> None:
        _maybe_add_project_root()
        self.dataset_configs = dataset_configs
        self.output_path = Path(output_path or "combined/production_dataset.json")
        self.stats: Dict[str, Any] = {
            "datasets": {},
            "total_loaded": 0,
            "total_invalid": 0,
            "total_duplicates": 0,
            "final_count": 0,
            "output_file": str(self.output_path),
        }

    def _load_one_dataset(
        self, dataset_type: str, source_path: Optional[str]
    ) -> List[ResumeDocument]:
        adapter = AdapterFactory.get_adapter(dataset_type, source_path)
        raw_docs = adapter.convert_all()

        valid_docs: List[ResumeDocument] = []
        invalid = 0
        for doc in raw_docs:
            try:
                # Round-trip through the Pydantic schema to validate every field.
                ResumeDocument.model_validate(doc.model_dump())
                valid_docs.append(doc)
            except Exception:
                invalid += 1

        self.stats["datasets"][dataset_type] = {
            "source_path": source_path or getattr(adapter, "source_path", "default"),
            "loaded": len(raw_docs),
            "valid": len(valid_docs),
            "invalid": invalid,
            "duplicates": 0,
        }
        self.stats["total_loaded"] += len(raw_docs)
        self.stats["total_invalid"] += invalid
        return valid_docs

    def load_all(self) -> List[ResumeDocument]:
        all_docs: List[ResumeDocument] = []
        for dataset_type, source_path in self.dataset_configs:
            docs = self._load_one_dataset(dataset_type, source_path)
            all_docs.extend(docs)
        return all_docs

    def deduplicate(self, docs: List[ResumeDocument]) -> List[ResumeDocument]:
        seen: set[str] = set()
        unique: List[ResumeDocument] = []

        for doc in docs:
            key = f"{doc.source_dataset}:{doc.candidate_id}"
            if key in seen:
                self.stats["total_duplicates"] += 1
                self.stats["datasets"][doc.source_dataset]["duplicates"] += 1
                continue
            seen.add(key)
            unique.append(doc)

        return unique

    def merge(self) -> List[ResumeDocument]:
        docs = self.load_all()
        final = self.deduplicate(docs)
        self.stats["final_count"] = len(final)
        return final

    def save(self, docs: List[ResumeDocument]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in docs], f, indent=2)

    def generate_report(self) -> str:
        lines = [
            "# MERGE_REPORT",
            "",
            f"**Generated at:** {datetime.utcnow().isoformat()}",
            f"**Output file:** `{self.output_path}`",
            "",
            "## Per-dataset statistics",
            "",
            "| Dataset | Loaded | Valid | Invalid | Duplicates Removed |",
            "|---------|--------|-------|---------|--------------------|",
        ]
        for dataset_type, s in self.stats["datasets"].items():
            lines.append(
                f"| `{dataset_type}` | {s['loaded']} | {s['valid']} | {s['invalid']} | {s['duplicates']} |"
            )
        lines.extend(
            [
                "",
                "## Aggregate statistics",
                "",
                f"- **Total loaded:** {self.stats['total_loaded']}",
                f"- **Total invalid (schema validation failed):** {self.stats['total_invalid']}",
                f"- **Total duplicates removed:** {self.stats['total_duplicates']}",
                f"- **Final dataset size:** {self.stats['final_count']}",
                "",
                "## Schema validation",
                "",
                "Every converted ``ResumeDocument`` was round-tripped through Pydantic validation "
                "before deduplication and serialization. Invalid records were dropped and counted above.",
                "",
                "## Notes",
                "",
                "- ``source_dataset`` metadata is preserved for every resume.",
                "- ``metadata_confidence`` and ``metadata_source`` from each adapter are preserved.",
                "- Deduplication is based on ``source_dataset:candidate_id``.",
            ]
        )
        return "\n".join(lines)
