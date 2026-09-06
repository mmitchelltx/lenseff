---
title: 'lenseff: detection efficiency for planetary microlensing surveys by injection-recovery'
tags:
  - Python
  - astronomy
  - exoplanets
  - gravitational microlensing
  - Nancy Grace Roman Space Telescope
  - detection efficiency
# TODO before submission: replace the placeholder ORCID and affiliation.
# JOSS's editorial bot validates ORCIDs, and 0000-0000-0000-0000 will be
# rejected. See docs/releasing.md.
authors:
  - name: Marcus Mitchell
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 6 September 2026
bibliography: paper.bib
---

# Summary

A planetary microlensing detection is only interpretable alongside the
efficiency of the search that found it: the fraction of planets of a given mass
ratio $q$ and projected separation $s$ that the pipeline would have recovered
had they been there. Occurrence rates, mass-ratio functions, and survey yield
forecasts are all detections divided by efficiency [@gaudi2000; @suzuki2016],
so the efficiency surface carries as much of the scientific result as the
detections do.

`lenseff` computes that surface by injection-recovery. It generates single-lens
light curves on a survey's real observing calendar, injects binary-lens
perturbations across a grid in $(\log q, \log s)$ while marginalising over the
source trajectory angle and over the event population, runs a configurable
detection pipeline over each one, and aggregates the recoveries into an
efficiency surface with binomial confidence intervals. All magnification
calculations are delegated to `MulensModel` [@poleski2019], which in turn uses
`VBBinaryLensing` for finite-source binary magnification [@bozza2018];
`lenseff` implements no lens equation of its own.

The package is written for the Nancy Grace Roman Space Telescope's Galactic
Bulge Time Domain Survey [@penny2019] and ships a preset for it, but nothing in
the library is Roman-specific: cadence, season windows, photometric error
model, and saturation limits are all configuration.

# Statement of need

The detection statistic in this kind of calculation is the $\chi^2$ difference
between the binary-lens model and a point-source point-lens (PSPL) model. The
subtlety, and the reason a shared implementation is worth having, is that the
PSPL model must be **refit freely** to the light curve that contains the
planet, letting $t_0$, $u_0$, $t_E$ and both fluxes float. A free single-lens
fit partially reabsorbs a planetary anomaly by shifting the peak, stretching
the timescale, and rebalancing the blend. Comparing the binary model against
the *injected* PSPL parameters instead measures the anomaly's raw amplitude
rather than its detectability.

The error this introduces is not uniform, which is what makes it dangerous.
Measured within `lenseff`'s own test suite, a strong anomaly ($q = 10^{-2}$,
$s = 1.3$) is inflated by a factor of 1.07 when the refit is skipped, while a
marginal one ($q = 10^{-3}$, $s = 0.6$) is inflated by a factor of 3.9. The
distortion is largest exactly in the regime that sets the position of the
efficiency contour, and therefore exactly where it propagates into a derived
occurrence rate.

Efficiency calculations of this kind are usually written per-project and not
released, so the choices that matter — the $\chi^2$ threshold, whether a run of
consecutive significant residuals is required, how season gaps are handled, and
whether the single-lens model is refit at all — are difficult to compare
between published surfaces. `lenseff` makes each of them an explicit
configuration key with a documented default, records every criterion per
injection so a published surface can be re-thresholded without recomputing the
grid, and embeds the resolved configuration, its SHA-256 hash, the seed, the
git commit and every dependency version in the output files.

# Implementation

Three design choices are worth naming.

**Reproducibility is structural, not incidental.** No random draw comes from a
running sequence. Each is addressed by a stable key path through
`numpy.random.SeedSequence` spawn keys — trajectory angles from
`(seed, "alpha", cell, event)`, photometric noise from
`(seed, "photometric_noise", event, realisation)` — so reordering the work
cannot reorder the randomness. A sweep therefore produces identical output on
one worker or on sixty-four, and an interrupted run resumes to the same table.
This is asserted by tests, not assumed. Because the configuration hash
deliberately excludes scheduling settings, a checkpointed run can be resumed on
a machine with a different core count.

**The refit is over-provisioned on purpose.** The $\chi^2$ surface of a PSPL
fit to an anomalous light curve is multimodal, and under-fitting inflates
efficiency. The fluxes are profiled out analytically, so only three parameters
reach the optimiser; fitting proceeds in $(t_0, \ln u_0, \ln t_E)$; many
candidate starts are screened by a single $\chi^2$ evaluation each and the best
few are refined; and one start comes from a PSPL fit to the data outside the
anomaly window, which is usually in the basin of the deepest minimum.

**Expensive magnification is used only where it is needed.** Finite-source
binary magnification costs roughly a thousand times more per point than the
point-source point-lens formula. `lenseff` obtains both the source trajectory
and the caustic curve from `MulensModel` and switches to the accurate method
only where the source approaches a caustic, which brings a full
injection-recovery cycle to about 170 ms on one core.

# Validation

Six tests encode behaviour that follows from microlensing physics or from the
statistics of the criteria, and gate continuous integration:

1. **Planet-free controls.** At $q = 0$ the binary model *is* the injected PSPL
   model, so a free refit can only fit at least as well as the truth and
   $\Delta\chi^2 \le 0$ exactly. The $\chi^2$ criterion therefore has a
   false-positive rate of zero by construction. The rate that genuinely exists
   belongs to the consecutive-points criterion, and it is checked against its
   closed form for Gaussian residuals.
2. Efficiency increases monotonically with $q$ at fixed $s$.
3. Efficiency peaks in the lensing zone near $s = 1$ and falls off toward close
   and wide separations [@mao1991].
4. Halving every photometric uncertainty increases efficiency.
5. An anomaly falling inside a season gap is never recovered, while the same
   planet on the same event in season is.
6. A configuration approximating OGLE-2005-BLG-390Lb [@beaulieu2006] is
   recovered.

Per-cell uncertainties are Wilson score intervals [@wilson1927], which remain
inside $[0, 1]$ and remain non-degenerate at zero and at complete recovery —
the regimes that dominate the edges of an efficiency surface, where a Gaussian
error bar is simply wrong.

# Availability

`lenseff` requires Python 3.11 or later and builds on NumPy [@numpy2020],
SciPy [@scipy2020], Matplotlib [@matplotlib2007] and Astropy [@astropy2022] in
addition to `MulensModel`. It is released under the MIT licence at
<https://github.com/mmitchelltx/lenseff>, with a tutorial notebook, a fully
documented configuration reference, and a demonstration configuration that
produces an efficiency surface and contour map in under ten minutes on a
laptop.

# References
