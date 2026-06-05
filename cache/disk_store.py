from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


class DiskArrayStore:
    def __init__(self, cache_root: str | None) -> None:
        self._cache_root = Path(cache_root) if cache_root else None

    @property
    def enabled(self) -> bool:
        return self._cache_root is not None

    def write(self, partition: str, key_hash: str, arr: np.ndarray) -> str:
        if self._cache_root is None:
            raise RuntimeError("Disk store is disabled")
        part_dir = self._cache_root / partition
        part_dir.mkdir(parents=True, exist_ok=True)
        payload = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
        fd, temp_path = tempfile.mkstemp(prefix=f"{key_hash}-", suffix=".npy", dir=part_dir)
        file_path = Path(temp_path)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, payload, allow_pickle=False)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        return str(file_path)

    def read(self, file_path: str, mmap: bool) -> np.ndarray:
        mmap_mode = "r" if mmap else None
        return np.load(file_path, allow_pickle=False, mmap_mode=mmap_mode)

    def delete(self, file_path: str) -> None:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            return
