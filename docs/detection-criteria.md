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

## Open questions for Phase 3

**(a) How is χ²(binary) evaluated — at the injected parameters, or fitted?**

This changes what the statistic means and it changes the answer to the first
validation test.

* *Injected parameters* (the usual choice in efficiency work, and the cheap
  one): Δχ² measures how much of a *known* signal a single-lens model cannot
  absorb. At `q = 0` the binary model is exactly the injected PSPL model, the
  refit can only do better than the truth, so Δχ² ≤ 0 identically and the
  false-positive rate from the Δχ² criterion is **zero by construction**. The
  consecutive-points criterion still has a non-zero rate, since noise can
  produce three consecutive 3σ residuals.
* *Fitted* (a planet *search*, seeded at or around the injected parameters, or
  over a grid of planet models): the statistic acquires the extra freedom that
  makes a genuine, measurable false-alarm rate — and costs one to two orders
  of magnitude more compute per injection.

The stated validation test "`q = 0` yields efficiency consistent with the
false-positive rate implied by the χ² threshold, not zero and not something
large" only has a non-trivial answer under the second formulation, or if the
`q = 0` control is scored on the consecutive-points criterion alone. **Decide
this before Phase 3 is written**, because it sets the compute budget for the
whole project.

**(b) Should the refit be allowed higher-order single-lens terms?**
Microlensing parallax and lens orbital motion can mimic or absorb long
anomalies. Including parallax in the refit is more conservative and more
expensive. Recommend: not in v1, but record the decision.

**(c) Error-bar renormalisation.** Real pipelines rescale uncertainties so
that the single-lens fit gives χ²/dof = 1, which directly rescales Δχ². With
simulated Gaussian noise this is a no-op; with a supplied catalog it is not.

**(d) Is Δχ² the right statistic at all, or should the criteria be evaluated
on a per-anomaly basis** (Δχ² restricted to the anomaly window)? The windowed
statistic is less sensitive to baseline systematics and is what some
ground-based analyses use. Both can be recorded per injection at negligible
extra cost, which is the recommended hedge.
