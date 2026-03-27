from __future__ import annotations

from pathlib import Path

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
        file_path = part_dir / f"{key_hash}.npy"
        payload = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
        np.save(file_path, payload, allow_pickle=False)
        return str(file_path)

    def read(self, file_path: str, mmap: bool) -> np.ndarray:
        mmap_mode = "r" if mmap else None
        return np.load(file_path, allow_pickle=False, mmap_mode=mmap_mode)

    def delete(self, file_path: str) -> None:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            return
