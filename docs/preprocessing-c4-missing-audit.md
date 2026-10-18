# 0724 result1 C4 missingness audit

Read-only inputs: `/home/balint/data/nanobio/cardio-fluidfm-data/additional/20260724/result1`
project parameters and the verifier's selection-compatible C4 raw acquisition cache
(`20260724/_cache/C4-raw-b63741aabaa79bc6.npz`). The verifier's `v2_cached_raw` checks
project flip, selection and phases before sharing result/result1/result2 acquisition
arrays, and reapplies result1's allowed mask. Selected acquisition range is [232,1170),
938 frames, phase starts 487 and 788. Registration and saved background picks are enabled.
Comparator: reviewed `default` with explicit `curation.method='per_pixel'`, not the
live low-rank default (86,383 missing). Both profiles use reviewed [233,239) curation.

| Sample partition | Before | After |
| --- | ---: | ---: |
| Reviewed per-pixel default MISSING | 75,101 | 75,101 |
| default_v2 MISSING | 83,790 | 76,890 |
| Missing only in v2 | 15,338 | 8,460 |
| Missing only in reviewed default | 6,649 | 6,671 |
| Net v2 excess | 8,689 | 1,789 |

| v2 missing reason | Total before | Total after | Newly missing vs reviewed, before | Newly missing vs reviewed, after |
| --- | ---: | ---: | ---: | ---: |
| unexplained_event, frames [233,239) | 11,022 | 4,122 | 10,995 | 4,117 |
| no_anchor, frames [233,239) | 594 | 594 | 257 | 257 |
| unresolved pixel labels, outside event | 72,174 | 72,174 | 4,086 | 4,086 |

The regional-step protection is not responsible: C4 has no qualified step receipts.
The main cause was rejecting an entire event pixel if any pulse observation was
unavailable, despite complete clean anchors. Of 1,837 rejected event pixels,
1,308 have partial pulse evidence (1,305 have five valid frames, two have four,
one has three). Missing pulse observations are concentrated at frame 234.
The other 529 pixels have complete but poorly explained pulse shapes.

The new partial-event criterion in `curation.repair_low_rank(..., replace_interval=True)`
restores 1,150 pixels × six frames = **6,900 samples**. It requires quiet complete
anchors, drift below the configured 75 pm threshold, identifiable basis coefficients
from more observations than basis dimensions, and ≥80% explained observed energy.
This uses the shared returning acquisition event without accepting a biological ramp
or fitting unknown pulse samples. Existing default/legacy/observed_only preprocessing
arrays retain their frozen byte comparisons.

The remaining 687 rejected event pixels comprise 529 complete but unexplained traces
and 158 partial traces below 80% explained observed energy (none failed the anchor,
sample-count or rank checks). Together they account for 4,122
missing samples; 99 pixels without complete anchors account for another 594.
These remain missing rather than forcing repairs. Newly missing unresolved samples
are 2,098 in [0,487), 1,525 in [487,788), and 463 in [788,938), across 114 pixels.
The new rule leaves these labels unchanged.

The diagnostic comparison identified 3,042 contiguous pixel/frame difference runs
in both directions of disagreement. The new rule changes only the reviewed pulse
interval. Generated diagnostic exports are not retained in the repository.

An independent full result1 automatic plate pass, with no curation, now yields
[233,239) in all 12 wells, versus four exact local detections before consensus.
All 12 final pulse residual audits have zero finite pixels above 75 pm.
