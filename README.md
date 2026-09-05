# lenseff

Detection efficiency for planetary microlensing surveys, by injection-recovery.

`lenseff` injects binary-lens (planetary) perturbations into single-lens light
curves on a real observing calendar, runs a detection pipeline over them, and
aggregates the recoveries into an efficiency surface in the (log *q*, log *s*)
plane with binomial uncertainties. The target application is the Nancy Grace
Roman Space Telescope's Galactic Bulge Time Domain Survey; nothing in the
library is Roman-specific, and surveys are described entirely by configuration.

## The methodological commitment

The detection statistic is

```
Δχ² = χ²(best free PSPL refit) − χ²(binary-lens model)
```

where the refit lets **every** PSPL parameter float — `t_0`, `u_0`, `t_E`, and
the fluxes `f_s`, `f_b`. A free single-lens fit partially reabsorbs a planetary
anomaly, so comparing against the *injected* PSPL parameters instead of a refit
substantially overestimates efficiency. That is the most common error in this
class of code, and avoiding it is the reason this package exists.

## Status

| phase | scope | state |
| --- | --- | --- |
| 0 | repo, packaging, CI, config schema + validation, provenance, RNG | **done** |
| 1 | `survey.py`, `events.py` — Roman GBTDS calendar and error model | **done** |
| 2 | `inject.py` — planetary perturbation via MulensModel | **done** |
| 3 | `detect.py` — multi-start PSPL refit and detection criteria | not started |
| 4 | `grid.py` — parallel sweep, checkpointing, resume | not started |
| 5 | `efficiency.py`, `plotting.py` — Wilson intervals, contour maps | not started |
| 6 | CLI `run`, docs, tutorial notebook | not started |

`lenseff run` is wired up in Phase 6; `validate` and `show` work today.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. All magnification calculations are delegated to
[MulensModel](https://github.com/rpoleski/MulensModel); `lenseff` never
implements a lens equation, a magnification, or a finite-source integral of its
own.

## Quickstart

```bash
lenseff validate configs/roman_gbtds_demo.yaml   # check a config before spending compute
lenseff show configs/roman_gbtds_demo.yaml       # print the resolved config and its hash
lenseff lightcurve configs/roman_gbtds_demo.yaml -o lc.png    # one event on the real calendar
lenseff run configs/roman_gbtds_demo.yaml        # Phase 6
```

### Anomaly morphologies

`python examples/anomaly_gallery.py` renders the textbook cases and is the
visual check that the injection is doing what it claims:

![anomaly gallery](docs/figures/anomaly_gallery.png)

Top left, a **major-image** perturbation (`s > 1`): a single sharp positive
spike. Top right, a **minor-image** perturbation (`s < 1`): the characteristic
demagnification dip immediately before the caustic spike. Bottom left, a
**central-caustic** perturbation on a high-magnification event, sitting on the
peak itself. Bottom right, `q = 1e-5`, short and weak — the regime where an
under-fitted PSPL refit would most badly inflate the efficiency.

## Detection criteria

Configurable, with these defaults (`detection:` in the config):

| criterion | key | default |
| --- | --- | --- |
| χ² improvement over the free PSPL refit | `delta_chi2_min` | 160 |
| consecutive points deviating from the refit | `consecutive_points` | 3 |
| per-point significance of that run | `point_sigma` | 3.0 |
| run must share a sign | `require_same_sign` | true |
| deviation must fall inside an observing season | `require_in_season` | true |

The thresholds are scientific judgements, not implementation details; see
[`docs/detection-criteria.md`](docs/detection-criteria.md) for what each one
buys and what it costs.

## Reproducibility

* **Config-driven.** Nothing scientific is hard-coded. Unknown keys are an
  error, not a silently ignored setting.
* **Addressed RNG.** Every random draw comes from a generator addressed by a
  stable key path (`("injection", cell, event, alpha)`), never from a shared
  sequential stream, so results do not depend on worker count or task order.
* **Provenance in the output.** Each parquet file embeds the resolved
  configuration, its SHA-256, the seed, the git commit, and the versions of
  every dependency that can move a number.

The contract: same config hash plus same seed ⟹ bit-identical data columns.

## Documentation

* [`docs/configuration.md`](docs/configuration.md) — every configuration key.
* [`docs/detection-criteria.md`](docs/detection-criteria.md) — criteria and rationale.
* [`docs/compute-budget.md`](docs/compute-budget.md) — cost per injection and grid sizing.

## License

MIT.
