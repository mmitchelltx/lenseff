# Compute budget

**Status: to be measured at the Phase 3 / Phase 4 gate.**

Before any production grid is launched, one full injection-recovery cycle must
be timed: generate the light curve, inject the planet, run the multi-start PSPL
refit, evaluate the criteria. Everything about the grid design follows from
that one number.

| quantity | value |
| --- | --- |
| time per injection-recovery cycle | *(to be measured)* |
| `configs/roman_gbtds_demo.yaml` | 2,752 injections |
| `configs/roman_gbtds_full.yaml` | 3,070,400 injections |

Scaling of the production config:

| ms / cycle | serial CPU-hours | wall-clock on 8 cores |
| --- | --- | --- |
| 20 | 17 | 2.1 h |
| 50 | 43 | 5.3 h |
| 200 | 171 | 21 h |
| 500 | 426 | 53 h |

The demo config must finish in under ten minutes on a laptop, which caps the
per-cycle cost at roughly 200 ms on a single core, or about 25 ms with eight
workers. If the measured cost is higher, the lever is the refit: `n_starts`
dominates, and the number of magnification evaluations per fit iteration is
next. Reducing the grid is the last resort, because grid resolution is what a
referee will look at.
