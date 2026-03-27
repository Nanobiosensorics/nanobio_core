from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from .config import CacheConfig


class TempDirManager:
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._active_path: Path | None = None
        self._started = False
        self._memory_only = False
        self._cleanup_registered = False

    @property
    def active_path(self) -> str | None:
        if self._active_path is None:
            return None
        return str(self._active_path)

    @property
    def memory_only(self) -> bool:
        return self._memory_only

    def start(self) -> str | None:
        if self._started:
            return self.active_path

        self._started = True
        try:
            if self._config.cache_root_mode == "explicit_path" and self._config.cache_root_path:
                root = Path(self._config.cache_root_path).expanduser().resolve()
                root.mkdir(parents=True, exist_ok=True)
                self._active_path = Path(tempfile.mkdtemp(prefix="nanobio-cache-", dir=str(root)))
            else:
                self._active_path = Path(tempfile.mkdtemp(prefix="nanobio-cache-"))
        except Exception:
            self._active_path = None
            self._memory_only = True
            return None

        if self._config.cleanup_on_exit and not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
        return str(self._active_path)

    def cleanup(self) -> None:
        if not self._config.cleanup_on_exit:
            return
        path = self._active_path
        self._active_path = None
        if path is None:
            return
        try:
            shutil.rmtree(path, ignore_errors=False)
        except Exception:
            return
