from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .signal import CellSignal


@dataclass
class CellRecord:
    coordinates: np.ndarray
    signal: Optional[CellSignal] = None
