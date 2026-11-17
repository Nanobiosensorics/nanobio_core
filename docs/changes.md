# Changes

## 2026-10-14

- Added per-well active-area masks for localization, complete microscope footprints and region growth. Invalid/out-of-frame coordinates are rejected without clipping.
- Added automatic raw-map boundary detection from acquisition frame zero with a two-pixel margin. Pipeline results carry detected filters while preserving stricter manual trims.

## 2026-10-09

- Added fractional microscope-mask projection, retaining label identities and positive-area contributions when multiple cells share a sensor pixel.
- Implemented threshold-bounded watershed partitioning on an 8×8 subpixel grid, with complete-footprint seeds, four-neighbour foreground paths and optional growth limits.
- Corrected boundary projection and pixel-center sampling, removing half-sensor-pixel centroid offsets and clipping outlines to native labels.
- Added weighted signal aggregation using sensor participation fractions; physical IWS multiplies the spatial sum by sensor pixel area. Existing CSVs require re-export.
- Added stable seed-point mapping and deterministic overlap handling; intensity determines foreground rather than shifting cell boundaries.

## 2026-10-03

- Restored valid empty microscope masks as empty candidate sets.
- Enabled compressed TIFF label loading with explicit tifffile/imagecodecs dependencies.

## 2026-10-01

- Added a shared threshold-bounded watershed API that preserves unresolved labels below threshold and measures growth limits from projected footprints.
- Enabled saved preprocessing when importing core through the evaluation repository's `src` package.

## 2026-09-24

- Kept invalid-pixel filtering and phase alignment in core defaults, with five frame-jump correction passes and microscope watershed threshold 160.
- Added phase-local interpolation of short signal artifacts, bounded to five frames by default.
- Validated manual background pixels and isolated invalid samples without discarding complete traces.
- Added per-well preprocessing/localization progress callbacks for GUI callers.

## 2026-07-29

- Added Cellpose mask discovery and validated two-dimensional label import.
- Added foreground/region aggregation helpers and Qt-independent selected-cell microscope crops with aligned sensor overlays.
- Unified GUI/export strategy contours and exposed clamped overlay opacity.
- Added a read-only v1–v5 microscope-project adapter with independent image/mask roots and fingerprint validation.
- Removed unused fixed-resolution microscope pyramid caching; rendering uses native source regions at viewport resolution.
