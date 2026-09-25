# Changes

## 2026-09-24

- Kept fixed preprocessing behavior in core method defaults: invalid-pixel filtering and inter-phase
  alignment always run, frame-jump correction defaults to five passes, and microscope watershed
  segmentation defaults to threshold 160. Evaluator metadata no longer carries these values.
- Added per-pixel interpolation for short Cardio signal artifacts. Bottleneck supplies the temporal
  moving-median baseline, and the configured pm threshold identifies meaningful deviations from it.
  Connected-run detection and NumPy linear interpolation remain phase-local, bounded to five frames
  by default, and report correction summaries for pipeline
  progress. Background correction now rejects masked, non-finite,
  and out-of-range manual reference pixels before aggregation. Individual out-of-range samples are
  zeroed without masking an otherwise usable sensor pixel for its complete trace.
- Added optional per-well progress callbacks to preprocessing and localization pipeline helpers so
  GUI callers can report restoration progress without parsing console output.

## 2026-07-29

- Added Cellpose `<well>_seg.npy` discovery and import. Dictionary payloads use the `masks` field, singleton dimensions are squeezed, and the resolved label mask must be two-dimensional.
- Added foreground-pixel construction, pixel-set merging, and aggregate-signal helpers for evaluator exports.
- Added `build_microscope_cell_image_data(...)` to construct selected-cell microscope crops, aligned EPIC overlays, focused-cell contours, and strategy-pixel contours without depending on Qt.
- Unified microscope strategy-pixel selection for GUI and export crops so `cover`, `max`, and `watershed` contours use the same focused-cell overlap and fallback rules.
- Added clamped overlay-alpha data to microscope single-cell export payloads so callers can reproduce GUI overlay opacity.
