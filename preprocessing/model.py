"""Arrays and stable acquisition/reconstruction method codes."""

from dataclasses import dataclass
import numpy as np

from ..epic_cardio.pixel_recovery import OBSERVED, TEMPORAL, SPATIAL, STITCHED, MISSING, EXCLUDED

CURATED = 6


@dataclass(frozen=True)
class RawWell:
    values_nm: np.ndarray
    codes: np.ndarray
    time_s: np.ndarray
    phases: tuple
    allowed: np.ndarray
    well: str = ''
    phase_source: str = 'provided'
    cell_mask: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.values_nm)
        if values.ndim != 3 or not len(values) or self.codes.shape != values.shape:
            raise ValueError('Expected nonempty matching time × height × width values and codes')
        if self.allowed.shape != values.shape[1:]:
            raise ValueError('Allowed mask must match spatial shape')
        if len(self.time_s) != len(values) or not np.isfinite(self.time_s).all():
            raise ValueError('Time must match values and be finite')
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError('Time must increase strictly')
        if tuple(sorted(set(self.phases))) != tuple(self.phases):
            raise ValueError('Phase boundaries must be sorted and unique')
        if any(p <= 0 or p >= len(values) for p in self.phases):
            raise ValueError('Phase boundary outside acquisition')


@dataclass
class PreprocessResult:
    values_pm: np.ndarray
    method: np.ndarray
    coverage: np.ndarray
    receipts: dict
    provenance: dict


@dataclass
class StageState:
    raw: RawWell
    values: np.ndarray
    method: np.ndarray
    allowed: np.ndarray
    receipts: dict
    acquired_values: np.ndarray | None = None
    acquired_valid: np.ndarray | None = None

    def copy(self):
        return StageState(self.raw, self.values.copy(), self.method.copy(),
                          self.allowed.copy(), dict(self.receipts), self.acquired_values, self.acquired_valid)
