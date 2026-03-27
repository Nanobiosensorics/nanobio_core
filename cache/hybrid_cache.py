from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .cache_metrics import CacheMetrics
from .config import CacheConfig, cache_config_from_env_or_defaults
from .disk_store import DiskArrayStore
from .key_serializer import KeySerializer
from .tempdir_manager import TempDirManager

_SHARED_MANAGER: TempDirManager | None = None
_SHARED_CONFIG: CacheConfig | None = None


def _shared_manager(config: CacheConfig | None) -> tuple[TempDirManager, CacheConfig]:
    if config is not None:
        manager = TempDirManager(config)
        manager.start()
        return manager, config

    global _SHARED_MANAGER
    global _SHARED_CONFIG
    if _SHARED_MANAGER is not None and _SHARED_CONFIG is not None:
        return _SHARED_MANAGER, _SHARED_CONFIG
    cfg = config if config is not None else cache_config_from_env_or_defaults()
    manager = TempDirManager(cfg)
    manager.start()
    _SHARED_MANAGER = manager
    _SHARED_CONFIG = cfg
    return manager, cfg


def cleanup_hybrid_cache_tempdir() -> None:
    global _SHARED_MANAGER
    global _SHARED_CONFIG
    if _SHARED_MANAGER is None:
        return
    _SHARED_MANAGER.cleanup()
    _SHARED_MANAGER = None
    _SHARED_CONFIG = None


@dataclass
class _Entry:
    key_hash: str
    key_repr: str
    normalized_key: Any
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    tier: str
    file_path: str | None
    access_count: int
    last_access_ns: int


class HybridPartitionCache:
    def __init__(
        self,
        partition: str,
        *,
        config: CacheConfig | None = None,
        memory_max_items: int | None = None,
        memory_max_bytes: int | None = None,
        spill_min_bytes: int | None = None,
    ) -> None:
        self.partition = partition
        self._manager, shared_config = _shared_manager(config)
        self._config = shared_config
        self._serializer = KeySerializer()
        self._store = DiskArrayStore(self._manager.active_path)
        self._metrics = CacheMetrics()
        self._index: dict[str, _Entry] = {}
        self._memory_store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._memory_max_items = max(1, int(memory_max_items or self._config.memory_max_items_per_partition))
        self._memory_max_bytes = max(0, int(memory_max_bytes or self._config.memory_max_bytes_per_partition))
        self._spill_min_bytes = max(0, int(spill_min_bytes or self._config.spill_min_bytes))
        self._memory_bytes = 0

    def get(self, key: Any) -> np.ndarray | None:
        key_hash, _key_repr, _normalized = self._serializer.hash_key(key)
        entry = self._index.get(key_hash)
        if entry is None:
            self._metrics.incr("miss")
            return None

        entry.access_count += 1
        entry.last_access_ns = time.time_ns()
        if entry.tier == "MEMORY":
            value = self._memory_store.get(key_hash)
            if value is None:
                self._index.pop(key_hash, None)
                self._metrics.incr("miss")
                return None
            self._memory_store.move_to_end(key_hash)
            self._metrics.incr("memory_hit")
            return value

        if entry.file_path is None:
            self._index.pop(key_hash, None)
            self._metrics.incr("miss")
            return None
        try:
            value = self._store.read(entry.file_path, mmap=self._config.mmap_reads)
            self._apply_readonly(value)
            self._metrics.incr("disk_hit")
            self._metrics.incr("spill_read")
            if value.nbytes < self._spill_min_bytes:
                self._promote_to_memory(key_hash, entry, value)
            return value
        except Exception:
            self._metrics.incr("io_error")
            self._metrics.incr("miss")
            self._delete_entry(key_hash)
            return None

    def put(self, key: Any, value: np.ndarray) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            return value

        key_hash, key_repr, normalized = self._serializer.hash_key(key)
        self._delete_entry(key_hash)
        arr = value
        entry = _Entry(
            key_hash=key_hash,
            key_repr=key_repr,
            normalized_key=normalized,
            dtype=str(arr.dtype),
            shape=tuple(int(v) for v in arr.shape),
            nbytes=int(arr.nbytes),
            tier="MEMORY",
            file_path=None,
            access_count=1,
            last_access_ns=time.time_ns(),
        )

        prefer_memory = arr.nbytes < self._spill_min_bytes
        if prefer_memory and self._can_fit_memory(arr.nbytes):
            self._insert_memory(entry, arr)
            self._enforce_memory_budget()
            self._refresh_gauges()
            return arr

        if self._try_write_disk(entry, arr):
            self._index[key_hash] = entry
            self._refresh_gauges()
            return arr

        if self._can_fit_memory(arr.nbytes):
            self._insert_memory(entry, arr)
            self._enforce_memory_budget()
        else:
            self._metrics.incr("evict")
        self._refresh_gauges()
        return arr

    def get_or_compute(self, key: Any, compute_fn: Callable[[], np.ndarray]) -> np.ndarray:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        return self.put(key, value)

    def invalidate_where(self, predicate_fn: Callable[[_Entry], bool]) -> int:
        removed = 0
        for key_hash in list(self._index.keys()):
            entry = self._index.get(key_hash)
            if entry is None:
                continue
            try:
                match = bool(predicate_fn(entry))
            except Exception:
                continue
            if not match:
                continue
            self._delete_entry(key_hash)
            removed += 1
        self._refresh_gauges()
        return removed

    def clear(self) -> None:
        for key_hash in list(self._index.keys()):
            self._delete_entry(key_hash)
        self._memory_store.clear()
        self._memory_bytes = 0
        self._refresh_gauges()

    def stats_snapshot(self) -> dict[str, int]:
        self._refresh_gauges()
        return self._metrics.snapshot()

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        self.put(key, value)

    def __getitem__(self, key: Any) -> np.ndarray:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def _can_fit_memory(self, nbytes: int) -> bool:
        if nbytes > self._memory_max_bytes:
            return False
        if len(self._memory_store) + 1 > self._memory_max_items:
            return True
        return self._memory_bytes + nbytes <= self._memory_max_bytes

    def _insert_memory(self, entry: _Entry, value: np.ndarray) -> None:
        entry.tier = "MEMORY"
        entry.file_path = None
        self._index[entry.key_hash] = entry
        self._memory_store[entry.key_hash] = value
        self._memory_store.move_to_end(entry.key_hash)
        self._memory_bytes += int(value.nbytes)

    def _try_write_disk(self, entry: _Entry, value: np.ndarray) -> bool:
        try:
            file_path = self._store.write(self.partition, entry.key_hash, value)
        except Exception:
            self._metrics.incr("io_error")
            return False
        entry.tier = "DISK"
        entry.file_path = file_path
        self._metrics.incr("spill_write")
        self._metrics.incr("demotion")
        return True

    def _promote_to_memory(self, key_hash: str, entry: _Entry, value: np.ndarray) -> None:
        if not self._can_fit_memory(int(value.nbytes)):
            return
        self._memory_store[key_hash] = value
        self._memory_store.move_to_end(key_hash)
        self._memory_bytes += int(value.nbytes)
        entry.tier = "MEMORY"
        self._metrics.incr("promotion")
        self._enforce_memory_budget()

    def _enforce_memory_budget(self) -> None:
        while self._memory_store and (
            len(self._memory_store) > self._memory_max_items or self._memory_bytes > self._memory_max_bytes
        ):
            key_hash, value = self._memory_store.popitem(last=False)
            self._memory_bytes -= int(value.nbytes)
            entry = self._index.get(key_hash)
            if entry is None:
                continue
            if self._try_write_disk(entry, value):
                continue
            self._index.pop(key_hash, None)
            self._metrics.incr("evict")

    def _delete_entry(self, key_hash: str) -> None:
        entry = self._index.pop(key_hash, None)
        if entry is None:
            return
        memory_value = self._memory_store.pop(key_hash, None)
        if memory_value is not None:
            self._memory_bytes -= int(memory_value.nbytes)
        if entry.file_path:
            self._store.delete(entry.file_path)

    def _apply_readonly(self, value: np.ndarray) -> None:
        if not self._config.readonly_arrays:
            return
        try:
            value.setflags(write=False)
        except Exception:
            return

    def _refresh_gauges(self) -> None:
        disk_items = 0
        disk_bytes = 0
        for entry in self._index.values():
            if entry.tier == "DISK":
                disk_items += 1
                disk_bytes += int(entry.nbytes)
        self._metrics.set_gauge("memory_items", len(self._memory_store))
        self._metrics.set_gauge("memory_bytes", self._memory_bytes)
        self._metrics.set_gauge("disk_items", disk_items)
        self._metrics.set_gauge("disk_bytes", disk_bytes)
        self._metrics.set_gauge("index_items", len(self._index))
