from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LoadDataResult:
    raw_wells: Dict[str, np.ndarray]
    full_time: np.ndarray
    full_phases: List[int]
    filter_params: Dict[str, List[List[int]]]
    preprocessing_params: Dict[str, Any]
    localization_params: Dict[str, Any]


@dataclass
class PreprocessResult:
    well_data: Dict[str, np.ndarray]
    time: np.ndarray
    phases: List[int]
    filter_ptss: Dict[str, List[List[int]]]
    selected_range: List[Optional[int]]
    active_area_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationResult:
    well_data: Dict[str, tuple]


@dataclass
class ExportResult:
    result_path: str


@dataclass
class PipelineRunResult:
    preprocess: PreprocessResult
    localization: LocalizationResult
