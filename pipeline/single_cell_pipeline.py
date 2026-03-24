from __future__ import annotations

from typing import Any, Dict

from ..epic_cardio import processing
from ..models.result import (
    LoadDataResult,
    LocalizationResult,
    PipelineRunResult,
    PreprocessResult,
)


def load_measurement_data(data_path: str, result_path: str, flip) -> LoadDataResult:
    filter_params, preprocessing_params, localization_params = processing.load_params(result_path)
    raw_wells, full_time, full_phases = processing.load_data(data_path, flip=flip)
    return LoadDataResult(
        raw_wells=raw_wells,
        full_time=full_time,
        full_phases=full_phases,
        filter_params=filter_params,
        preprocessing_params=preprocessing_params,
        localization_params=localization_params,
    )


def preprocess_data(raw_wells, full_time, full_phases, preprocessing_params, filter_ptss) -> PreprocessResult:
    well_data, time, phases, filter_points, selected_range = processing.preprocessing(
        preprocessing_params,
        raw_wells,
        full_time,
        full_phases,
        filter_ptss,
    )
    return PreprocessResult(
        well_data=well_data,
        time=time,
        phases=phases,
        filter_ptss=filter_points,
        selected_range=selected_range,
    )


def localize_data(raw_wells, phases, selected_range, preprocessing_params, localization_params, filter_ptss) -> LocalizationResult:
    return LocalizationResult(
        well_data=processing.localization(
            preprocessing_params,
            localization_params,
            raw_wells,
            phases,
            selected_range,
            filter_ptss,
        )
    )


def save_pipeline_metadata(result_path: str, localized_wells, preprocessing_params, localization_params) -> None:
    processing.save_params(
        result_path,
        localized_wells,
        preprocessing_params,
        localization_params,
    )


def run_pipeline(data: Dict[str, Any], config: Dict[str, Any]) -> PipelineRunResult:
    preprocessed = preprocess_data(
        data["raw_wells"],
        data["full_time"],
        data["full_phases"],
        config["preprocessing_params"],
        data.get("filter_ptss", {}),
    )
    localized = localize_data(
        data["raw_wells"],
        preprocessed.phases,
        preprocessed.selected_range,
        config["preprocessing_params"],
        config["localization_params"],
        preprocessed.filter_ptss,
    )
    return PipelineRunResult(preprocess=preprocessed, localization=localized)
