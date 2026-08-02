"""Bootstrap status enum for clear startup reporting."""

from enum import Enum


class BootstrapStatus(str, Enum):
    """Possible outcomes of the bootstrap startup process."""

    BUILT_NEW_INDEXES = "BUILT_NEW_INDEXES"
    LOADED_EXISTING_INDEXES = "LOADED_EXISTING_INDEXES"
    REBUILT_CORRUPTED_INDEXES = "REBUILT_CORRUPTED_INDEXES"
    FAILED = "FAILED"
