# Well annotations

`annotations.py` provides a dependency-free version 1 parser for explicit well treatments:
`Noncoated, HeLa -> SAL1(1.5 uM)` or `Gelatin, Preo -> HeLa-EV -> -`.
Arrows separate phases; plus joins simultaneous additions; a lone minus records no change.
Earlier materials remain unless an explicit `Wash(DMEM)` or `Replace(DMEM)` occurs.

The JSON-serializable payload records coating, ordered phase actions, recipient/effector
cell additions, EV donor and quantities with units/basis. RGD percentage is mixture fraction;
unknown coating is distinct from noncoated. Omitted quantities remain unknown. Supported
material vocabulary is explicit and rejects arbitrary legacy group text.

Use `build_plate_annotations(assignments, phase_binding=..., source=...)` after reading a
plate map, then `validate_plate_annotations` on reload. Bindings must match displayed phases
and strictly increase. They map to recorded phase indices; no acquisition times are inferred.
The evaluator may retain legacy manual labels separately. Structured-only consumers must
require this payload instead of parsing legacy labels. Add new materials explicitly to the
shared vocabulary with tests before accepting them in a production plate map.
