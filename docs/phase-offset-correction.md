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
