"""Adapter placeholder for the EraMatch CV Parsing Benchmark dataset."""

from __future__ import annotations

from typing import Any

from src.models import ResumeDocument

from .base_adapter import BaseDatasetAdapter


class EraMatchAdapter(BaseDatasetAdapter):
    """
    Adapter for the EraMatch CV Parsing Benchmark (JSON, PDF + image) dataset.

    The dataset is not currently downloaded, so this adapter is an interface
    only. When the dataset is available, implement ``load()``, ``validate()``,
    and ``convert()`` to map the rich EraMatch JSON schema to ``ResumeDocument``.
    """

    source_name = "eramatch"

    def __init__(self, source_path: str | None = None) -> None:
        super().__init__(source_path)

    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "EraMatchAdapter.load() is not implemented: dataset not yet downloaded."
        )

    def validate(self, record: dict[str, Any]) -> bool:
        raise NotImplementedError(
            "EraMatchAdapter.validate() is not implemented: dataset not yet downloaded."
        )

    def convert(self, record: dict[str, Any]) -> ResumeDocument:
        raise NotImplementedError(
            "EraMatchAdapter.convert() is not implemented: dataset not yet downloaded."
        )
