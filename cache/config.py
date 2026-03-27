from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_mb(name: str, default_mb: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default_mb * 1024 * 1024
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default_mb * 1024 * 1024
    return max(0, value) * 1024 * 1024


@dataclass(frozen=True)
class CacheConfig:
    cache_root_mode: str = "auto_temp"
    cache_root_path: str | None = None
    memory_max_items_per_partition: int = 128
    memory_max_bytes_per_partition: int = 64 * 1024 * 1024
    spill_min_bytes: int = 1 * 1024 * 1024
    mmap_reads: bool = False
    compression: str = "none"
    cleanup_on_exit: bool = True
    readonly_arrays: bool = True


def cache_config_from_env_or_defaults() -> CacheConfig:
    cache_root = os.getenv("NANOBIO_CACHE_DIR")
    return CacheConfig(
        cache_root_mode="explicit_path" if cache_root else "auto_temp",
        cache_root_path=cache_root,
        memory_max_items_per_partition=128,
        memory_max_bytes_per_partition=_env_int_mb("NANOBIO_CACHE_MEM_MB", 64),
        spill_min_bytes=_env_int_mb("NANOBIO_CACHE_SPILL_MIN_MB", 1),
        mmap_reads=_env_bool("NANOBIO_CACHE_MMAP_READS", False),
        compression="none",
        cleanup_on_exit=True,
        readonly_arrays=True,
    )
