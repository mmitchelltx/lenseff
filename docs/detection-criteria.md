# Detection criteria

These are scientific judgements, not implementation details. A referee will
interrogate them, so each one is stated here with what it buys, what it costs,
and what still has to be decided.

Everything below is configurable under `detection:`; the defaults are the ones
shipped in `configs/roman_gbtds_demo.yaml`.

## 1. Δχ² > 160 against a free PSPL refit

```
Δχ² = χ²(best free PSPL refit) − χ²(binary-lens model)
```

**The refit is the whole point.** The PSPL refit lets `t_0`, `u_0`, `t_E`,
`f_s` and `f_b` all float, so it absorbs whatever part of the planetary
signal a single-lens model can mimic — a shifted peak, a slightly different
`t_E`, a different blend fraction. Comparing against the *injected* PSPL
parameters instead measures the anomaly's raw amplitude rather than its
detectability, and inflates efficiency, badly at low `q` and near `s = 1`
where the anomaly is broadest.

Why 160 and not, say, 100 or 500:

* Under Gaussian noise the number is enormous. With a handful of extra
  parameters, χ² improvements of 160 have a Gaussian false-alarm probability
  far below one per survey, even after the ~10⁵–10⁶ trials of a real search.
  So **160 is not set by Gaussian statistics** — it is set by systematics.
* The real false-positive population comes from low-level correlated noise:
  detector effects, imperfect image subtraction, unmodelled stellar
  variability, parallax and binary-source degeneracies. A threshold in the
  low hundreds is the working compromise the field has converged on for
  requiring that a candidate be robust to those.
* Raising the threshold buys purity and costs completeness, mostly at low
  `q` where the anomaly is weakest — which is exactly the regime the
  efficiency surface is used to interpret. The sensitivity of the published
  surface to this choice should be quantified by rerunning at 100 and 300;
  that is cheap once the grid exists (`delta_chi2_min` is a config key, and
  the per-injection Δχ² is stored, so it can be re-thresholded without
  recomputing).

**To verify against the literature before publication:** the threshold and
statistic definitions in Penny et al. (2019) for Roman, and in the KMTNet and
OGLE detection-efficiency papers. Cite what you adopt and say where you
differ.

## 2. At least 3 consecutive points deviating by > 3σ

Applied to the residuals **from the PSPL refit**, not from the injected model.

* Guards against a single hot pixel or cosmic ray driving the entire Δχ².
* Guards against a Δχ² accumulated diffusely over thousands of points, which
  is the signature of a mis-estimated `t_E` or an error-bar problem rather
  than a localised anomaly.
* Encodes the physical expectation that a planetary anomaly is *resolved* by
  the cadence. At Roman's cadence this is a weak requirement; at ground-based
  cadence it is a strong one, which is why it is configurable.

### What it actually rejects, measured

This criterion is not decorative. Two concrete cases from the test suite:

**A diffuse deviation that passes the chi-square threshold.** A configuration
approximating OGLE-2005-BLG-390Lb (`q = 7.6e-5`, `s = 1.61`) at a trajectory
angle that *misses* the planetary caustic still accumulates `Δχ² = 615` — well
above 160 — from 349 points each deviating by about 1σ over three days. No
single point reaches 3σ. In real photometry a 1σ trend lasting three days is
indistinguishable from correlated noise, and the run criterion is the only
thing that rejects it. (`test_a_diffuse_low_amplitude_deviation_is_rejected`)

**In the demo run, it is the binding criterion.** At `q = 1e-2`, 58% of
injections pass the Δχ² threshold but only 36% pass the run requirement, and
the split is clean:

| Δχ² | injections | median run length | detected |
| --- | ---: | ---: | ---: |
| 160 – 1,000 | 90 | 1 | 1% |
| 1,000 – 10,000 | 88 | 3 | 55% |
| > 10,000 | 159 | 244 | 100% |

Everything in the first row is chi-square accumulated thinly across thousands
of points; everything in the last is a resolved anomaly. The criterion
separates them almost perfectly.

Two supporting knobs:

* `require_same_sign` (default true): the run must be all-positive or
  all-negative residuals. Noise assembles alternating-sign runs far more
  easily than a real perturbation does.
* `max_gap_within_run_days` (default 0.25): two points count as consecutive
  only if they are closer together than this. Without it, three points
  spanning a downlink gap or a season boundary would count as a "run".

## 3. The deviation must fall inside an observing season

An anomaly in a season gap is not a marginal detection; it is not a detection
at all. Making this explicit rather than relying on "there are no data there
anyway" matters because the Δχ² criterion alone can be satisfied by the
*wings* of a perturbation whose core is unobserved, and because it makes the
season structure a first-class, testable part of the pipeline — one of the six
validation tests places the anomaly in a gap and requires exactly zero
recovery.

## Decisions (settled 2026-09-05)

### (a) χ²(binary) is evaluated at the injected parameters

`Δχ² = χ²(free PSPL refit) − χ²(binary at the injected q, s, α, ρ, with f_s and
f_b refit linearly)`. This is the standard injection-recovery formulation and
it costs one binary-model evaluation per injection instead of a full planet
search. The alternative — fitting (q, s, α) — buys a genuine false-alarm rate
at 10–100× the compute, which would put a production Roman grid out of reach
of the hardware this package targets.

**Consequence, and it is a feature, not a gap.** At `q = 0` the binary model
*is* the injected PSPL model, and the free refit can only fit at least as well
as the truth, so

```
Δχ² ≤ 0    identically, for every planet-free injection
```

The Δχ² criterion therefore has a false-positive rate of exactly zero *by
construction*. That is a sharper, checkable statement than "some small rate",
and the validation suite asserts the inequality directly on every `q = 0`
control rather than estimating a rate that is not there.

The false-positive rate that genuinely exists belongs to criterion 2, the run
of consecutive deviant points, which noise *can* produce. For independent
Gaussian residuals with per-point threshold `k` sigma and a required run of `r`
same-sign points, the expected number of runs in `N` points is

```
E[runs] ≈ 2 (N − r + 1) p^r (1 − p),    p = Φ(−k)
```

which for the defaults (`k = 3`, `r = 3`) and Roman's ~5 × 10⁴ points per
light curve gives ~2 × 10⁻⁴ per event. The validation suite checks the
measured rate against this closed form at high statistics on synthetic noise,
where the rate is actually measurable, instead of on a few hundred light
curves where it is not.

So the original validation requirement splits into two stronger tests:

1. `Δχ² ≤ 0` for every planet-free injection (exact, not statistical), and the
   combined criteria fire on essentially none of them.
2. The consecutive-points false-alarm rate matches the run-statistics
   prediction to within Poisson error.

### (b) No parallax or lens orbital motion in the refit for v1

Including microlensing parallax in the PSPL refit is more conservative — it
gives the single-lens model more freedom to absorb long anomalies — and
roughly triples the refit cost. Recorded as a known limitation: efficiency for
anomalies with timescales approaching `t_E` is an upper bound.

### (c) Error bars are not renormalised

The simulated noise is Gaussian by construction with known σ, so rescaling to
χ²/dof = 1 is a no-op. When events come from a supplied catalog the user is
responsible for the uncertainties they hand in. Revisit if catalog mode is
used on real photometry.

### (d) Both the global and the windowed Δχ² are recorded

Every injection stores `delta_chi2` (all points) and `delta_chi2_window`
(points inside the anomaly window). The windowed statistic is less sensitive
to baseline systematics; storing both costs nothing and lets the surface be
re-thresholded on either without recomputing the grid.
