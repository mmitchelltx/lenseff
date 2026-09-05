r"""The free PSPL refit, the detection statistic, and the detection criteria.

This is the scientific core of the package.  The statistic is

.. math::

    \Delta\chi^2 = \chi^2(\text{best free PSPL refit}) - \chi^2(\text{binary})

where the refit lets ``t_0``, ``u_0``, ``t_E`` and both fluxes float.  A free
single-lens model partially reabsorbs a planetary anomaly -- it shifts the
peak, stretches the timescale and rebalances the blend -- so comparing the
binary model against the *injected* PSPL parameters instead of against a refit
substantially overestimates efficiency.  Everything below exists to make that
refit find the true global minimum.

How the refit is made robust
----------------------------

The chi-square surface is multimodal, so a single optimisation from a single
starting point is not trustworthy.  The refit therefore:

* **profiles out the fluxes.**  At fixed ``(t_0, u_0, t_E)`` the model is
  linear in ``f_s`` and ``f_b``, so they are solved exactly rather than
  searched.  Only three parameters are ever passed to the optimiser, which
  makes the surface far better behaved and each evaluation cheaper.
* **fits in log space.**  ``ln u_0`` and ``ln t_E`` keep both parameters
  positive and give the simplex a scale-free step.
* **screens many starts cheaply, then refines a few.**  Chi-square is
  evaluated at every candidate start -- which costs one magnification
  evaluation each -- and only the best ``refit.n_refine`` are handed to the
  optimiser.
* **includes an anomaly-masked start.**  A PSPL fit to the data *outside* the
  anomaly window is usually in the basin of the deepest minimum, because the
  anomaly is exactly what drags a single-lens fit away from it.
* **includes the injected truth as a start.**  This uses knowledge a real
  pipeline would not have, and it is deliberately conservative: any start that
  helps the refit reach a lower chi-square can only *reduce* Delta chi-square
  and therefore the measured efficiency.

Finite source
-------------
When the injected baseline needs finite-source magnification -- which happens
when ``u_0`` is comparable to ``rho`` -- the refit uses it too, with ``rho``
held at the event's value.  If it did not, the refit family would not contain
the injected planet-free model, and a ``q = 0`` control could show a spurious
positive Delta chi-square.  The validation suite asserts the exact inequality
``Delta chi2 <= 0`` on every control injection, which is what makes this
correct rather than merely plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.optimize import least_squares, minimize

from lenseff.config import Config, RefitConfig
from lenseff.events import Event, LightCurve
from lenseff.inject import Injection
from lenseff.survey import Survey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    import MulensModel as mm

__all__ = [
    "DetectionResult",
    "FluxSolution",
    "PSPLFit",
    "detect",
    "longest_significant_run",
    "refit_pspl",
    "solve_fluxes",
]

#: Half-width of the finite-source method window, in source radii.
_FINITE_SOURCE_RADII: Final[float] = 10.0

#: Chi-square returned for a parameter set the magnification code cannot
#: evaluate.  Large enough to be rejected, finite so the optimiser can still
#: step away from it.
_PENALTY_CHI2: Final[float] = 1e30

#: A refit that needs finite-source magnification is detected by comparing the
#: point-source and finite-source baselines: if they differ by more than this
#: many sigma at any point, the refit must model the source size too.
_FINITE_SOURCE_TRIGGER_SIGMA: Final[float] = 0.01


@dataclass(frozen=True, slots=True)
class FluxSolution:
    """The exact linear solution for the source and blend fluxes.

    Attributes:
        f_source: Fitted source flux.
        f_blend: Fitted blend flux.
        chi2: Chi-square at that solution.
    """

    f_source: float
    f_blend: float
    chi2: float


def solve_fluxes(
    magnification: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    free_blending: bool = True,
    require_positive_source: bool = True,
) -> FluxSolution:
    """Solve exactly for the fluxes at fixed non-linear parameters.

    The model ``f = f_s A + f_b`` is linear in the fluxes, so the weighted
    least-squares solution is a 2x2 system rather than a search.

    Args:
        magnification: Model magnification at each measurement.
        flux: Measured flux.
        flux_err: Measurement uncertainties.
        free_blending: Fit ``f_b``; when false the blend is held at zero.
        require_positive_source: Clamp a negative source flux to zero, which
            is the constrained optimum, and refit the blend alone.  Negative
            *blend* flux is left free, as is standard in microlensing.

    Returns:
        The flux solution and its chi-square.
    """
    weight = 1.0 / flux_err**2
    s_aa = float(np.sum(magnification * magnification * weight))
    s_af = float(np.sum(magnification * flux * weight))
    if not free_blending:
        f_source = s_af / s_aa if s_aa > 0.0 else 0.0
        f_source = max(f_source, 0.0) if require_positive_source else f_source
        return _flux_solution(f_source, 0.0, magnification, flux, flux_err)

    s_a1 = float(np.sum(magnification * weight))
    s_11 = float(np.sum(weight))
    s_1f = float(np.sum(flux * weight))
    determinant = s_aa * s_11 - s_a1 * s_a1
    if determinant <= 0.0:  # pragma: no cover - degenerate magnification curve
        return _flux_solution(0.0, s_1f / s_11, magnification, flux, flux_err)
    f_source = (s_af * s_11 - s_1f * s_a1) / determinant
    f_blend = (s_1f * s_aa - s_af * s_a1) / determinant
    if require_positive_source and f_source < 0.0:
        return _flux_solution(0.0, s_1f / s_11, magnification, flux, flux_err)
    return _flux_solution(f_source, f_blend, magnification, flux, flux_err)


def _flux_solution(
    f_source: float,
    f_blend: float,
    magnification: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> FluxSolution:
    """Package a flux pair with the chi-square it produces."""
    residual = (flux - f_source * magnification - f_blend) / flux_err
    return FluxSolution(f_source, f_blend, float(np.dot(residual, residual)))


@dataclass(frozen=True, slots=True)
class PSPLFit:
    """The outcome of a free PSPL refit.

    Attributes:
        t_0: Fitted time of peak.
        u_0: Fitted impact parameter.
        t_E: Fitted Einstein crossing time.
        f_source: Fitted source flux.
        f_blend: Fitted blend flux.
        chi2: Chi-square of the fit.
        n_starts: Candidate starting points screened.
        n_refined: Starting points handed to the optimiser.
        converged: Whether at least one optimisation reported success.
    """

    t_0: float
    u_0: float
    t_E: float
    f_source: float
    f_blend: float
    chi2: float
    n_starts: int
    n_refined: int
    converged: bool

    @property
    def parameters(self) -> dict[str, float]:
        """The fitted non-linear parameters."""
        return {"t_0": self.t_0, "u_0": self.u_0, "t_E": self.t_E}


class _PSPLObjective:
    """Chi-square of a free PSPL model, with the fluxes profiled out.

    A single ``MulensModel.Model`` is created and its parameters mutated in
    place, which avoids rebuilding the object thousands of times per fit.

    Args:
        light_curve: The data to fit.
        refit: Refit configuration.
        rho: Source radius to model, or ``None`` for a point source.
    """

    def __init__(self, light_curve: LightCurve, refit: RefitConfig, rho: float | None) -> None:
        import MulensModel as mm

        self._times = light_curve.times
        self._flux = light_curve.flux
        self._flux_err = light_curve.flux_err
        self._refit = refit
        self._rho = rho
        event = light_curve.event
        parameters = {"t_0": event.t_0, "u_0": max(event.u_0, 1e-6), "t_E": event.t_E}
        if rho is not None:
            parameters["rho"] = rho
        self._model: mm.Model = mm.Model(parameters)
        self.n_evaluations = 0

    def magnification(self, t_0: float, u_0: float, t_E: float) -> np.ndarray:
        """Return the PSPL magnification for one parameter set.

        Raises:
            ValueError: If the parameters are unusable, which happens when an
                unbounded optimiser walks into ``t_E -> 0`` or ``u_0 -> 0``.
                Callers turn this into a finite penalty rather than a crash.
        """
        self.n_evaluations += 1
        if not (
            np.isfinite(t_0) and np.isfinite(u_0) and np.isfinite(t_E) and u_0 > 0.0 and t_E > 1e-3
        ):
            raise ValueError(f"unusable PSPL parameters t_0={t_0}, u_0={u_0}, t_E={t_E}")
        parameters = self._model.parameters
        parameters.t_0 = t_0
        parameters.u_0 = u_0
        parameters.t_E = t_E
        if self._rho is not None:
            half = _FINITE_SOURCE_RADII * self._rho * t_E
            self._model.set_magnification_methods(
                [t_0 - half, "finite_source_uniform_Gould94", t_0 + half]
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            magnification = np.asarray(self._model.get_magnification(self._times), dtype=float)
        if not np.all(np.isfinite(magnification)):
            raise ValueError("magnification is not finite for these PSPL parameters")
        return magnification

    def solve(self, vector: np.ndarray) -> FluxSolution:
        """Return the flux solution and chi-square at a parameter vector."""
        t_0, log_u_0, log_t_E = vector
        try:
            magnification = self.magnification(
                float(t_0), float(np.exp(log_u_0)), float(np.exp(log_t_E))
            )
        except ValueError:
            return FluxSolution(0.0, 0.0, _PENALTY_CHI2)
        return solve_fluxes(
            magnification,
            self._flux,
            self._flux_err,
            free_blending=self._refit.free_blending,
            require_positive_source=self._refit.require_positive_source,
        )

    def chi2(self, vector: np.ndarray) -> float:
        """Return chi-square at a parameter vector."""
        return self.solve(vector).chi2

    def residuals(self, vector: np.ndarray) -> np.ndarray:
        """Return the weighted residual vector at a parameter vector."""
        t_0, log_u_0, log_t_E = vector
        try:
            magnification = self.magnification(
                float(t_0), float(np.exp(log_u_0)), float(np.exp(log_t_E))
            )
        except ValueError:
            return np.full(self._flux.size, np.sqrt(_PENALTY_CHI2 / self._flux.size))
        solution = solve_fluxes(
            magnification,
            self._flux,
            self._flux_err,
            free_blending=self._refit.free_blending,
            require_positive_source=self._refit.require_positive_source,
        )
        model = solution.f_source * magnification + solution.f_blend
        return (self._flux - model) / self._flux_err


def _to_vector(t_0: float, u_0: float, t_E: float) -> np.ndarray:
    """Pack parameters into the optimiser's log-space vector."""
    return np.array([t_0, np.log(max(u_0, 1e-8)), np.log(max(t_E, 1e-3))], dtype=float)


def _from_vector(vector: np.ndarray) -> tuple[float, float, float]:
    """Unpack the optimiser's vector into physical parameters."""
    return float(vector[0]), float(np.exp(vector[1])), float(np.exp(vector[2]))


def _candidate_starts(event: Event, refit: RefitConfig) -> Iterator[np.ndarray]:
    """Yield candidate starting points around the injected parameters."""
    yield _to_vector(event.t_0, event.u_0, event.t_E)
    offsets = (0.0, -refit.t_0_jitter_t_E, refit.t_0_jitter_t_E)
    for offset in offsets:
        for u_factor in refit.u_0_factors:
            for t_factor in refit.t_E_factors:
                if offset == 0.0 and u_factor == 1.0 and t_factor == 1.0:
                    continue
                yield _to_vector(
                    event.t_0 + offset * event.t_E,
                    event.u_0 * u_factor,
                    event.t_E * t_factor,
                )


def _bounds(event: Event, refit: RefitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return generous physical bounds on the fit vector.

    The refit is a single-lens fit to data that is nearly single-lens, so its
    minimum lies close to the injected parameters.  The bounds are wide enough
    never to clip that minimum, and exist only to stop an optimiser walking
    into ``t_E -> 0``, where the magnification is undefined.
    """
    lower = np.array(
        [
            event.t_0 - refit.t_0_bound_t_E * event.t_E,
            np.log(refit.u_0_bound_min),
            np.log(event.t_E / refit.t_E_bound_factor),
        ]
    )
    upper = np.array(
        [
            event.t_0 + refit.t_0_bound_t_E * event.t_E,
            np.log(refit.u_0_bound_max),
            np.log(event.t_E * refit.t_E_bound_factor),
        ]
    )
    return lower, upper


def _optimise(
    objective: _PSPLObjective,
    start: np.ndarray,
    refit: RefitConfig,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, bool]:
    """Run one optimisation from one starting point, inside the bounds."""
    lower, upper = bounds
    start = np.clip(start, lower, upper)
    if refit.method == "least_squares":
        result = least_squares(
            objective.residuals,
            start,
            bounds=(lower, upper),
            method="trf",
            xtol=refit.tolerance,
            ftol=refit.tolerance,
            gtol=refit.tolerance,
            max_nfev=refit.max_iterations,
        )
        return np.asarray(result.x, dtype=float), bool(result.success)
    if refit.method == "L-BFGS-B":
        result = minimize(
            objective.chi2,
            start,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": refit.max_iterations},
        )
        return np.asarray(result.x, dtype=float), bool(result.success)
    options = (
        {"maxiter": refit.max_iterations, "xatol": refit.tolerance, "fatol": refit.tolerance}
        if refit.method == "Nelder-Mead"
        else {"maxiter": refit.max_iterations}
    )
    result = minimize(objective.chi2, start, method=refit.method, options=options)
    return np.clip(np.asarray(result.x, dtype=float), lower, upper), bool(result.success)


def _needs_finite_source(injection: Injection, config: Config) -> float | None:
    """Return the source radius the refit must model, or ``None``.

    The refit family has to contain the injected planet-free model, otherwise a
    control injection can show a spurious positive Delta chi-square.  The test
    is empirical: compare the point-source and finite-source baselines on this
    light curve and see whether the difference is measurable.
    """
    from lenseff.events import pspl_magnification

    if not config.injection.finite_source:
        return None
    event = injection.event
    light_curve = injection.light_curve
    point = pspl_magnification(event, light_curve.times, finite_source=False)
    finite = injection.pspl_magnification
    difference = np.abs(finite - point) * light_curve.f_source / light_curve.flux_err
    return event.rho if float(difference.max()) > _FINITE_SOURCE_TRIGGER_SIGMA else None


def refit_pspl(injection: Injection, config: Config) -> PSPLFit:
    """Refit a free PSPL model to a light curve containing a planet.

    Args:
        injection: The injected light curve.
        config: The run configuration.

    Returns:
        The best fit found across all starting points.
    """
    refit = config.detection.refit
    light_curve = injection.light_curve
    rho = _needs_finite_source(injection, config)
    objective = _PSPLObjective(light_curve, refit, rho)

    bounds = _bounds(light_curve.event, refit)
    starts = list(_candidate_starts(light_curve.event, refit))[: refit.n_starts]
    if refit.mask_anomaly_start and injection.anomaly.exists:
        masked = _anomaly_masked_start(injection, config, rho, bounds)
        if masked is not None:
            starts.insert(0, masked)

    scored = sorted(((objective.chi2(start), index) for index, start in enumerate(starts)))
    best_vector = starts[scored[0][1]]
    best_chi2 = scored[0][0]
    converged = False
    n_refined = min(refit.n_refine, len(scored))
    for _, index in scored[:n_refined]:
        vector, success = _optimise(objective, starts[index], refit, bounds)
        chi2 = objective.chi2(vector)
        converged = converged or success
        if chi2 < best_chi2:
            best_chi2, best_vector = chi2, vector

    solution = objective.solve(best_vector)
    t_0, u_0, t_E = _from_vector(best_vector)
    return PSPLFit(
        t_0=t_0,
        u_0=u_0,
        t_E=t_E,
        f_source=solution.f_source,
        f_blend=solution.f_blend,
        chi2=solution.chi2,
        n_starts=len(starts),
        n_refined=n_refined,
        converged=converged,
    )


def _anomaly_masked_start(
    injection: Injection,
    config: Config,
    rho: float | None,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    """Fit PSPL to the data outside the anomaly and return it as a start."""
    light_curve = injection.light_curve
    outside = ~injection.anomaly.contains(light_curve.times)
    if int(outside.sum()) < 50:
        return None
    masked = LightCurve(
        times=light_curve.times[outside],
        flux=light_curve.flux[outside],
        flux_err=light_curve.flux_err[outside],
        season_index=light_curve.season_index[outside],
        f_source=light_curve.f_source,
        f_blend=light_curve.f_blend,
        magnification=light_curve.magnification[outside],
        event=light_curve.event,
    )
    refit = config.detection.refit
    objective = _PSPLObjective(masked, refit, rho)
    event = light_curve.event
    vector, _ = _optimise(objective, _to_vector(event.t_0, event.u_0, event.t_E), refit, bounds)
    return vector


def longest_significant_run(
    residual: np.ndarray,
    times: np.ndarray,
    *,
    sigma: float,
    same_sign: bool,
    max_gap_days: float,
) -> tuple[int, int, int]:
    """Find the longest run of consecutive significant residuals.

    "Consecutive" means adjacent in the time series *and* separated by no more
    than ``max_gap_days``, so a run can never be assembled across a downlink
    gap or a season boundary.

    Args:
        residual: Residuals in units of the uncertainty.
        times: Measurement times, sorted.
        sigma: Per-point significance threshold.
        same_sign: Require every point of the run to deviate the same way.
        max_gap_days: Maximum spacing between consecutive points of a run.

    Returns:
        The run length, and the first and last index of the longest run.
        ``(0, -1, -1)`` when no point is significant.
    """
    significant = np.abs(residual) > sigma
    if not significant.any():
        return 0, -1, -1
    best_length, best_start, best_end = 0, -1, -1
    length, start = 0, 0
    for i in range(residual.size):
        if not significant[i]:
            length = 0
            continue
        if length > 0:
            broken = times[i] - times[i - 1] > max_gap_days
            if same_sign and not broken:
                broken = np.sign(residual[i]) != np.sign(residual[i - 1])
            if broken:
                length = 0
        if length == 0:
            start = i
        length += 1
        if length > best_length:
            best_length, best_start, best_end = length, start, i
    return best_length, best_start, best_end


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything scored for one injection.

    Every criterion is recorded separately so that the efficiency surface can
    be re-thresholded from the stored table without recomputing the grid.

    Attributes:
        delta_chi2: ``chi2(refit) - chi2(binary)`` over all points.
        delta_chi2_window: The same, restricted to the anomaly window.
        chi2_refit: Chi-square of the best free PSPL refit.
        chi2_binary: Chi-square of the binary model at the injected
            parameters, with its fluxes refit.
        run_length: Longest run of consecutive significant residuals from the
            refit.
        run_t_start: Start time of that run.
        run_t_end: End time of that run.
        run_in_season: Whether the run lies inside an observing season.
        n_points_in_anomaly: Measurements inside the anomaly window.
        fit: The PSPL refit.
        criteria: Pass/fail of each configured criterion.
        detected: Whether every criterion passed.
    """

    delta_chi2: float
    delta_chi2_window: float
    chi2_refit: float
    chi2_binary: float
    run_length: int
    run_t_start: float
    run_t_end: float
    run_in_season: bool
    n_points_in_anomaly: int
    fit: PSPLFit
    criteria: dict[str, bool] = field(default_factory=dict)
    detected: bool = False

    def as_record(self) -> dict[str, float | int | bool]:
        """Return a flat record for the output table."""
        record: dict[str, float | int | bool] = {
            "delta_chi2": self.delta_chi2,
            "delta_chi2_window": self.delta_chi2_window,
            "chi2_refit": self.chi2_refit,
            "chi2_binary": self.chi2_binary,
            "run_length": self.run_length,
            "run_in_season": self.run_in_season,
            "n_points_in_anomaly": self.n_points_in_anomaly,
            "fit_t_0": self.fit.t_0,
            "fit_u_0": self.fit.u_0,
            "fit_t_E": self.fit.t_E,
            "fit_f_source": self.fit.f_source,
            "fit_f_blend": self.fit.f_blend,
            "fit_converged": self.fit.converged,
            "detected": self.detected,
        }
        record.update({f"criterion_{name}": value for name, value in self.criteria.items()})
        return record


def detect(injection: Injection, survey: Survey, config: Config) -> DetectionResult:
    """Score one injection against the configured detection criteria.

    Args:
        injection: The injected light curve.
        survey: The survey, used to test whether a deviation lies in a season.
        config: The run configuration.

    Returns:
        The detection result, with every criterion recorded separately.
    """
    detection = config.detection
    light_curve = injection.light_curve

    fit = refit_pspl(injection, config)
    binary = solve_fluxes(
        injection.binary_magnification,
        light_curve.flux,
        light_curve.flux_err,
        free_blending=detection.refit.free_blending,
        require_positive_source=detection.refit.require_positive_source,
    )

    refit_magnification = _refit_magnification(injection, config, fit)
    refit_model = fit.f_source * refit_magnification + fit.f_blend
    binary_model = binary.f_source * injection.binary_magnification + binary.f_blend
    residual = (light_curve.flux - refit_model) / light_curve.flux_err
    binary_residual = (light_curve.flux - binary_model) / light_curve.flux_err

    delta_chi2 = fit.chi2 - binary.chi2
    in_window = injection.anomaly.contains(light_curve.times)
    delta_chi2_window = float(
        np.sum(residual[in_window] ** 2) - np.sum(binary_residual[in_window] ** 2)
    )

    run_length, first, last = longest_significant_run(
        residual,
        light_curve.times,
        sigma=detection.point_sigma,
        same_sign=detection.require_same_sign,
        max_gap_days=detection.max_gap_within_run_days,
    )
    if run_length > 0:
        run_t_start = float(light_curve.times[first])
        run_t_end = float(light_curve.times[last])
        run_in_season = bool(survey.in_season(light_curve.times[first : last + 1]).all())
    else:
        run_t_start = run_t_end = float("nan")
        run_in_season = False

    n_in_anomaly = int(in_window.sum())
    criteria = {
        "delta_chi2": delta_chi2 >= detection.delta_chi2_min,
        "consecutive_points": run_length >= detection.consecutive_points,
        "points_in_anomaly": n_in_anomaly >= detection.min_points_in_anomaly,
    }
    if detection.require_in_season:
        criteria["in_season"] = run_in_season
    return DetectionResult(
        delta_chi2=float(delta_chi2),
        delta_chi2_window=delta_chi2_window,
        chi2_refit=float(fit.chi2),
        chi2_binary=float(binary.chi2),
        run_length=run_length,
        run_t_start=run_t_start,
        run_t_end=run_t_end,
        run_in_season=run_in_season,
        n_points_in_anomaly=n_in_anomaly,
        fit=fit,
        criteria=criteria,
        detected=all(criteria.values()),
    )


def _refit_magnification(injection: Injection, config: Config, fit: PSPLFit) -> np.ndarray:
    """Evaluate the fitted PSPL model at the measurement times."""
    objective = _PSPLObjective(
        injection.light_curve, config.detection.refit, _needs_finite_source(injection, config)
    )
    return objective.magnification(fit.t_0, fit.u_0, fit.t_E)
