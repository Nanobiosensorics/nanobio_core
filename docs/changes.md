# Changes

- Fractional watershed now expands full-region seeds through four-neighbour
  threshold-foreground paths using flat marker-controlled watershed. Labels
  cannot take shortcuts across background gaps in a connected region; the
  intensity defines the boundary, not the separation gradient.

- Corrected fractional watershed: partition connected threshold foreground on a
  640×640 grid using complete microscope regions as seeds and direct foreground
  contact paths. Sensor participation is the fraction of its 64 subpixels
  assigned to each cell, including growth beyond the original mask. Unseeded
  foreground remains empty, and the optional growth cap is enforced on this
  partition. The same cached labels drive well overlays and GUI/export outlines.
- Corrected exact mask-area projection at fractional sensor boundaries and
  standardized native crop sampling to pixel centers and sensor-edge indexing.
  IWS exports use `sum(signal × watershed participation)`; physical IWS in
  analysis additionally multiplies by the sensor pixel area. The fine partition
  has an 8×8 sampling precision per sensor pixel; existing CSVs require re-export.

- Added native-mask sensor-pixel participation projection. Each label retains
  its fractional occupied area when several high-resolution cells map to one
  Cardio pixel, and signal aggregation accepts weighted `(x, y, fraction)`
  regions for area-proportional integrated IWS.

- Generalized labeled-mask projection to honor the requested output shape and
  clipped watershed single-cell outlines to the native high-resolution label.

- Replaced foreground-distance watershed basins with deterministic nearest-cover
  partitioning inside each connected threshold component. The complete projected
  cell footprint sets ownership distance; centroid distance breaks footprint ties.
  Signal intensity only defines foreground and does not move cell boundaries.

- Added label-specific sensor coverage projection using positive-area overlap
  with source mask pixels. Unlike a center-sampled label image, these pixel sets
  retain small labels and represent shared coverage without replacing label IDs.

- Watershed uses one centroid-guided seed per overlapping label and the foreground
  distance transform for separation. Optional `seed_points` map stable label IDs to
  projected `(x, y)` coordinates; seed collisions use the nearest available footprint
  pixel. Strict threshold background and optional growth bounds are preserved.
  Microscope mask projection now samples sensor-pixel centers rather than corners,
  eliminating a half-sensor-pixel offset from projected cell centroids.

## 2026-10-03

- Valid empty microscope label TIFFs restore as empty candidate sets, allowing
  nuclei-filtered wells with no eligible cells to remain in saved datasets.
- Label TIFF decoding uses tifffile, including ZSTD-compressed edited masks.
- Core environment dependencies explicitly include tifffile and imagecodecs
  for compressed label-mask restoration.

## 2026-10-01

- Added a shared threshold-bounded watershed result API. The intensity threshold now defines
  authoritative background, microscope labels below threshold remain unresolved instead of being
  forced into the foreground, and optional growth limits are measured from each projected label
  footprint. The legacy watershed entrypoint delegates to the same implementation.
- Use package-relative imports in processing helpers so saved preprocessing can run
  when core is imported through the evaluation repository's `src` package.

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

- Added a read-only v1–v5 microscope project adapter with independent image/mask roots, active/channel/composite selection and fingerprint validation. Synchronized package-relative processing imports across embedded checkouts.

- Removed unused fixed-resolution microscope pyramid caching and its public exports. Current consumers render native source regions at viewport resolution.
