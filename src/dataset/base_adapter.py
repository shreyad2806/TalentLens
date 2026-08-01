"""Abstract base class for all TalentLens dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.models import ResumeDocument


class BaseDatasetAdapter(ABC):
    """
    Abstract adapter that every dataset source must implement.

    The adapter's job is to translate a raw dataset record into a single,
    validated ``ResumeDocument``. Downstream components (parser, chunking,
    embeddings, search) never know which dataset produced the document.
    """

    source_name: str = "base"

    def __init__(self, source_path: Optional[str] = None) -> None:
        self.source_path = source_path

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load the raw dataset into a list of record dictionaries."""
        ...

    @abstractmethod
    def validate(self, record: Dict[str, Any]) -> bool:
        """Return True if the raw record has the minimum required fields."""
        ...

    @abstractmethod
    def convert(self, record: Dict[str, Any]) -> ResumeDocument:
        """Convert a single raw record into a ``ResumeDocument``."""
        ...

    def convert_all(self, records: Optional[List[Dict[str, Any]]] = None) -> List[ResumeDocument]:
        """Load (if not provided), validate, and convert every record."""
        if records is None:
            records = self.load()

        documents: List[ResumeDocument] = []
        for idx, record in enumerate(records):
            if not self.validate(record):
                print(f"[{self.source_name}] Validation failed for record at index {idx}; skipping.")
                continue
            try:
                documents.append(self.convert(record))
            except Exception as exc:
                print(f"[{self.source_name}] Conversion failed for record at index {idx}: {exc}")
        return documents
