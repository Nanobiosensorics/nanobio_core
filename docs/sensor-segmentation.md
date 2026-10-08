# Sensor segmentation

`epic_cardio.processing.subpixel_watershed` projects native microscope labels onto
an 8× grid by default and uses full-region markers to grow through four-neighbour
threshold foreground. Disconnected unseeded regions remain background. Optional
growth distance is in sensor-pixel units; callers choose whether to limit growth.
Sensor values are repeated without interpolating new signal values.

Returns a fine label image and per-label `(x, y, fraction)` rows. Fractions use the
whole sensor-pixel area as denominator. Consumers sum `WS × fraction` and multiply
by physical sensor area once to obtain spatial IWS. A fixed maximum-projection
support must be reused at every frame when matching the aligned evaluation workflow.

`image_fitting.microscope.project_mask_pixel_participation` measures native-label
coverage with fractional boundary areas. `project_mask_to_epic` scales to the
requested output shape with pixel-center alignment. Package-relative imports allow
both `nanobio_core` and an embedded `src.nanobio_core` import.
