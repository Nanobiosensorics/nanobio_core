# Phase-local pixel recovery

`epic_cardio.pixel_recovery.recover_pixels(values, original_valid, time, phases,
allowed, config)` accepts nm wavelengths, original acquisition validity, actual
timestamps and the allowed pixel domain. It returns recovered values, per-sample
method/radius/donor maps and step receipts. Excluded data are unchanged.

Short bracketed oscillations are interpolated within a phase.
Short returning pulses are checked before persistent-step fitting, using original
observations on both sides and their noise. This prevents a pulse's falling edge
from introducing a false stitched plateau when median noise masks part of the pulse.
Before stitching, an earlier six-frame baseline (12–18 frames before the edge)
must also support a persistent displacement. Returning to that baseline rejects
the stitch, including pulses longer than the temporal interpolation limit. Those
longer pulses are not automatically interpolated. Insufficient earlier history
leaves the candidate unconfirmed.
Persistence also requires six valid later samples, 6–12 frames after the edge.
The earlier trend's extrapolation uncertainty is included when deciding whether
the later signal returns; a noisy rising baseline cannot justify a stitched plateau.
Temporal endpoint compatibility includes measured flank noise and requires the
return sample to agree with the following same-phase samples.
All temporal brackets and stitching support stay inside the same acquisition
phase. Between-phase gaps are left to the dedicated phase-offset correction,
which runs once; pixel recovery does not interpolate or stitch them.
Outlier medians and noise estimates ignore invalid acquisition samples. When
fewer than three usable samples support a window, its recorded nonzero samples
are retained without an outlier verdict; zero-majority windows cannot erase them.
Terminal loss is filled only where original measured anchors enclose the target component, using
expanding search radii 1/2/4/8 and a maximum 81-pixel component. The harmonic
solution is computed jointly as a cached linear system equivalent to converged
iterative neighbor filling; farther enclosing donors use linear interpolation.
Imputed pixels never become independent donors. Unsupported/ambiguous gaps stay
NaN. Nonzero steps need persistent displacement and agreement from at least two
nearby original traces before stitching; that consensus is explicitly recorded
as an assumption. True biological jumps cannot be classified from speed alone.

`correct_interphase_well_shifts(..., return_translations=True)` optionally
returns the integer translations so acquisition validity follows registration.
Default return behavior is unchanged. Registered sources can be replayed through
`preprocessing.pixel_recovery_source`; source hashes, phase boundaries and
registration settings are checked. Localization also checks the allowed domain.
The source path performs background/phase correction once and bypasses legacy
cross dilation and destructive clipping. Missing WS reference pixels remain
unavailable; unresolved samples render as zero only for legacy consumers, with
their source masks retained.

Spatial estimates are unsuitable as unqualified measured peak/IWS values.
Separate reconstruction fractions and original-only metrics in downstream use.
Scoping read core README, existing correction code/docs and the evaluation
recovery plan. Tests cover numerical recovery, border/phase boundaries,
unsupported gaps, source invalidation and both processing entry points.
