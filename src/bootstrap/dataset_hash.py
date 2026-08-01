"""
Dataset hash manager for the bootstrap system.

Computes a stable content hash of the discovered resume corpus and stores it
to disk.  On the next startup the cached hash is compared with the current
hash; a mismatch means the dataset changed and the vector / BM25 indexes must
be rebuilt.
"""

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HASH_PATH = Path("data/cache/dataset_hash.json")


class DatasetHashManager:
    """Manages the persistent dataset hash."""

    def __init__(self, hash_path: Optional[str] = None):
        self.hash_path = Path(hash_path) if hash_path else DEFAULT_HASH_PATH

    def _file_signature(self, file_path: Path) -> dict:
        """Return a stable signature for a single file."""
        stat = file_path.stat()
        return {
            "path": str(file_path.resolve()).replace(os.sep, "/"),
            "size": stat.st_size,
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def compute_hash(self, file_paths: list, csv_path: Optional[str] = None) -> str:
        """
        Compute a SHA-256 hash of the dataset.

        Uses file paths, sizes and mtimes for resume files and the raw content
        hash of the CSV, if present.
        """
        records = []
        for path in sorted({str(p) for p in file_paths}):
            p = Path(path)
            if p.exists():
                records.append(self._file_signature(p))

        if csv_path:
            csv = Path(csv_path)
            if csv.exists():
                sha = hashlib.sha256()
                with csv.open("rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha.update(chunk)
                records.append({
                    "path": str(csv.resolve()).replace(os.sep, "/"),
                    "sha256": sha.hexdigest(),
                })

        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load_hash(self) -> Optional[str]:
        """Load the previously saved dataset hash, if any."""
        if not self.hash_path.exists():
            return None
        try:
            data = json.loads(self.hash_path.read_text(encoding="utf-8"))
            return data.get("dataset_hash")
        except Exception as e:
            logger.warning(f"Could not read cached dataset hash: {e}")
            return None

    def save_hash(self, hash_value: str) -> None:
        """Persist the dataset hash to disk."""
        self.hash_path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_path.write_text(
            json.dumps({"dataset_hash": hash_value}, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Remove the cached hash."""
        if self.hash_path.exists():
            self.hash_path.unlink()
