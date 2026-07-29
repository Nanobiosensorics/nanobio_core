# Changes

## 2026-07-29

- Added Cellpose `<well>_seg.npy` discovery and import. Dictionary payloads use the `masks` field, singleton dimensions are squeezed, and the resolved label mask must be two-dimensional.
- Added foreground-pixel construction, pixel-set merging, and aggregate-signal helpers for evaluator exports.
- Added `build_microscope_cell_image_data(...)` to construct selected-cell microscope crops, aligned EPIC overlays, focused-cell contours, and strategy-pixel contours without depending on Qt.
- Unified microscope strategy-pixel selection for GUI and export crops so `cover`, `max`, and `watershed` contours use the same focused-cell overlap and fallback rules.
- Added clamped overlay-alpha data to microscope single-cell export payloads so callers can reproduce GUI overlay opacity.
