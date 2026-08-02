"""Factory for retrieving the right dataset adapter by name."""

from __future__ import annotations

from .base_adapter import BaseDatasetAdapter
from .eramatch_adapter import EraMatchAdapter
from .kaggle_adapter import KaggleAdapter
from .synthetic_adapter import SyntheticAdapter


class AdapterFactory:
    """
    Returns the correct adapter for a given dataset type.

    Adding a new dataset only requires two steps:
      1. Create a new adapter class inheriting from ``BaseDatasetAdapter``.
      2. Register it in ``_adapters`` below.
    """

    _adapters = {
        "kaggle": KaggleAdapter,
        "synthetic": SyntheticAdapter,
        "eramatch": EraMatchAdapter,
    }

    @classmethod
    def get_adapter(cls, dataset_type: str, source_path: str | None = None) -> BaseDatasetAdapter:
        """Return an adapter instance for ``dataset_type``."""
        adapter_class = cls._adapters.get(dataset_type)
        if not adapter_class:
            raise ValueError(
                f"Unknown dataset type: {dataset_type!r}. "
                f"Available: {list(cls._adapters)}"
            )
        return adapter_class(source_path=source_path)

    @classmethod
    def available_adapters(cls) -> list[str]:
        """Return the list of registered dataset types."""
        return list(cls._adapters)
