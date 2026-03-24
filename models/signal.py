from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellSignal:
    raw: np.ndarray
    filtered: np.ndarray
