from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from typing import Any

import numpy as np


class KeySerializer:
    def normalize(self, key: Any) -> Any:
        try:
            return self._normalize_value(key)
        except Exception:
            return {"__repr__": repr(key)}

    def serialize(self, normalized_obj: Any) -> str:
        return json.dumps(normalized_obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def hash_key(self, key: Any) -> tuple[str, str, Any]:
        normalized = self.normalize(key)
        serialized = self.serialize(normalized)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return digest, serialized, normalized

    def _normalize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, tuple):
            return {"__tuple__": [self._normalize_value(item) for item in value]}
        if isinstance(value, list):
            return {"__list__": [self._normalize_value(item) for item in value]}
        if isinstance(value, dict):
            normalized_items: list[tuple[str, Any, Any]] = []
            for k, v in value.items():
                nk = self._normalize_value(k)
                nv = self._normalize_value(v)
                sort_key = self.serialize(nk)
                normalized_items.append((sort_key, nk, nv))
            normalized_items.sort(key=lambda item: item[0])
            return {"__dict__": [[item[1], item[2]] for item in normalized_items]}

        if is_dataclass(value) and not isinstance(value, type):
            return {
                "__dataclass__": value.__class__.__qualname__,
                "fields": [[field.name, self._normalize_value(getattr(value, field.name))] for field in fields(value)],
            }

        if isinstance(value, set):
            members = [self._normalize_value(item) for item in value]
            members.sort(key=self.serialize)
            return {"__set__": members}

        return {"__repr__": repr(value)}
