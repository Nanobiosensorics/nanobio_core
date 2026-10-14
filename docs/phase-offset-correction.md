# Pixel phase continuity

Tracker: [nanobio_core #7](https://github.com/Nanobiosensorics/nanobio_core/issues/7).
This is the preparatory phase-offset commit for `feature/hela-nk-composites`;
classified composite support remains future work.

`epic_cardio.processing` applies `correct_phase_offsets` after phase image
registration, background correction and optional jump repair, before localization
or cell metrics. `drift_correction.phase_offset_correction` defaults to true when
missing. Explicit false preserves previous processing. The helper adds one
constant pixel map per later phase, accumulating from corrected prior endpoints;
it preserves signed within-phase increments and skips nonfinite endpoints.
This treats all unobserved gap changes as bias and can remove biological responses.
Cell-specific or subgroup-specific metric offsets are not part of this operation.

Run the core arithmetic/finite-endpoint tests from this repository with
`python -m pytest -q tests/test_phase_offset_correction.py`. The Cardio app covers
processing order, enable/disable behavior, parameter persistence and GUI wiring.

Recovered cubes can have an unobserved phase endpoint. Previously this skipped
phase correction, then became an artificial zero-to-signal step after finite-cube
rendering. With phase correction enabled, pixels missing either boundary endpoint
are now excluded from quantitative support for the complete trace. Valid pixels
retain the existing constant-offset correction and within-phase increments. No
missing endpoint is invented and no acquisition gap is interpolated. The source
bundle retains missingness/provenance. Explicitly disabling phase correction keeps
the prior missing-data behavior.
# Reviewed returning pulses

`epic_cardio.confirmed_pulse.repair_confirmed_pulse` supports an explicitly
reviewed within-phase disturbance. It checks original clean windows, independent
pre/post trend agreement and neighboring excursions before interpolating the
complete interval using actual timestamps. It never repairs across phase gaps,
uses reconstructed anchors, alters excluded border pixels or converts a returning
pulse into a persistent offset. It is opt-in; automatic short-gap limits remain
unchanged. July project source receipts identify the reviewed interval and method.
