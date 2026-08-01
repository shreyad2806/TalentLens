"""
Dataset adapters for the TalentLens ingestion pipeline.

Each adapter converts a raw dataset into the unified ``ResumeDocument`` schema.
New datasets are added by creating a new adapter and registering it in the
``AdapterFactory``.
"""

from .adapter_factory import AdapterFactory
from .base_adapter import BaseDatasetAdapter
from .eramatch_adapter import EraMatchAdapter
from .kaggle_adapter import KaggleAdapter
from .synthetic_adapter import SyntheticAdapter

__all__ = [
    "AdapterFactory",
    "BaseDatasetAdapter",
    "EraMatchAdapter",
    "KaggleAdapter",
    "SyntheticAdapter",
]
