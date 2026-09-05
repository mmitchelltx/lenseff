# Compute budget

Measured on this repository at the Phase 3 gate, before the grid sweep was
designed. Single core, `configs/roman_gbtds_demo.yaml`, 84 cycles spanning the
whole `(q, s)` grid, Python 3.11 / MulensModel 3.11.

| stage | cost | notes |
| --- | --- | --- |
| event setup | 253 ms | once per event, not per injection |
| injection | 43 ms | binary magnification + anomaly window |
| detection | 123 ms | the multi-start PSPL refit dominates |
| **full cycle** | **166 ms** | 190 ms at the 90th percentile |

## What that buys

| config | injections | CPU-hours | 4 cores | 8 cores | 64 cores |
| --- | ---: | ---: | ---: | ---: | ---: |
| `roman_gbtds_demo.yaml` | 2,752 | 0.13 | 1.9 min | 1.0 min | — |
| `roman_gbtds_full.yaml` | 3,070,400 | 141 | 35 h | 18 h | 2.2 h |

The demo config has roughly a factor of five of headroom against its ten-minute
target on a laptop, which is where the grid resolution should be spent.

The production config is a real but tractable overnight run on a workstation,
or a couple of hours on a cluster node. **Check this table against your own
hardware before launching it** — one `lenseff` cycle is dominated by
MulensModel and by SciPy, so it tracks single-core floating-point performance
closely.

## Where the time goes, and the levers

Measured per-call costs that set everything above:

| operation | cost per point | for 8,641 points |
| --- | ---: | ---: |
| PSPL point-source magnification | 0.10 µs | 0.83 ms |
| binary point-source magnification | 1.77 µs | 15.3 ms |
| finite-source binary (`VBBL`) | 106 µs | — |
| linear flux solve + chi-square | — | 0.07 ms |

So one refit objective evaluation costs about 0.9 ms, and the refit's cost is
essentially *(number of optimiser evaluations) × 0.9 ms*.

Levers, in the order worth pulling:

1. **`detection.refit.n_refine`** (default 4). Linear in the refit cost. The
   cheap chi-square screen over `n_starts` candidates means robustness comes
   from widening `n_starts`, which costs one evaluation each, not from raising
   `n_refine`.
2. **`injection.analysis_window_t_E`** (default 3). Every objective evaluation
   is linear in the number of points retained. Narrowing it saves time but
   weakens the baseline-flux constraint.
3. **`injection.finite_source_radii`** (default 20). Only 5 ms of the cycle at
   the default, and lowering it costs accuracy at the caustic crossing —
   a poor trade.
4. **Grid resolution.** Last resort. Grid resolution is what a referee looks
   at, and the per-injection cost is where the savings actually are.

Reproduce this table with:

```bash
python examples/timing.py
```
