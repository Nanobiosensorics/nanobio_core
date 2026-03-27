from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict


class CacheMetrics:
    def __init__(self) -> None:
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._gauges: dict[str, int] = {}

    def incr(self, event_name: str, amount: int = 1) -> None:
        self._counters[event_name] += int(amount)

    def set_gauge(self, name: str, value: int) -> None:
        self._gauges[name] = int(value)

    def snapshot(self) -> dict[str, int]:
        out = dict(self._counters)
        out.update(self._gauges)
        return out
