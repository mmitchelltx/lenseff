"""Injection of a binary-lens (planetary) perturbation into a PSPL event.

Every magnification is computed by ``MulensModel``; nothing here implements a
lens equation, a magnification formula or a finite-source integral.

Two things make this module more than a wrapper.

**Where the expensive method is used.**  Finite-source binary magnification
(``VBBL``) costs roughly a thousand times more per point than the point-source
point-lens formula, so it is applied only where it matters: the source
trajectory and the caustic curve are both obtained from ``MulensModel``, and
the accurate method is switched on wherever the source centre comes within
``injection.finite_source_radii`` source radii of the caustic.  Everywhere else
in the analysis window the binary point-source solution is used, which is exact
for a point source and about sixty times cheaper.

**Where the anomaly is.**  The anomaly window is located on a dense grid in
time, independently of the observing cadence, so an anomaly that falls entirely
inside a season gap is still found -- and can then be correctly reported as
unobservable.

Shared noise
------------
Every planet injected into a given event reuses that event's standard-normal
draws, rescaled by the uncertainties of the perturbed light curve.  Differences
in Δχ² across the grid are therefore driven by the planet and not by the noise,
which sharply reduces the variance of the efficiency surface at no cost in
correctness.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from lenseff.config import Config
from lenseff.events import Event, LightCurve, pspl_magnification
from lenseff.rng import generator
from lenseff.survey import Survey

if TYPE_CHECKING:  # pragma: no cover - typing only
    import MulensModel as mm

__all__ = [
    "AnomalyWindow",
    "EventSetup",
    "Injection",
    "Planet",
    "analysis_half_width",
    "binary_magnification",
    "caustic_extent",
    "inject_planet",
    "locate_anomaly",
]

#: Extra margin, in Einstein radii, added when the analysis window is widened
#: to cover a planetary caustic that lies far from the host.
_CAUSTIC_MARGIN: Final[float] = 1.0

#: MulensModel emits this when ``rho`` is set but a stretch of the light curve
#: uses a point-source method.  That is exactly the intended configuration
#: here: away from the caustics the finite-source correction is orders of
#: magnitude below the photometric precision.
_FINITE_SOURCE_WARNING: Final[str] = "A finite source parameter"


@dataclass(frozen=True, slots=True)
class Planet:
    """A planetary companion to the lens.

    Attributes:
        q: Planet-to-host mass ratio.  ``0`` means "no planet", which is the
            control case and is handled without ever building a binary model.
        s: Projected separation in Einstein radii.
        alpha_deg: Angle of the source trajectory relative to the binary axis,
            in degrees, in the MulensModel convention.
    """

    q: float
    s: float
    alpha_deg: float

    @property
    def is_control(self) -> bool:
        """Whether this is a planet-free control injection."""
        return self.q <= 0.0

    def as_record(self) -> dict[str, float]:
        """Return a flat record for the output table."""
        return {"q": self.q, "s": self.s, "alpha_deg": self.alpha_deg}


@dataclass(frozen=True, slots=True)
class AnomalyWindow:
    """Where in time the planet perturbs the light curve.

    Attributes:
        exists: Whether any deviation above the threshold was found.
        t_start: First time at which the flux difference between the binary
            and baseline PSPL models exceeds ``injection.anomaly_sigma``
            photometric sigma.
        t_end: Last such time.
        t_peak: Time of maximum deviation.
        peak_deviation: Maximum deviation, in units of the photometric sigma.
    """

    exists: bool
    t_start: float
    t_end: float
    t_peak: float
    peak_deviation: float

    @property
    def duration_days(self) -> float:
        """Length of the anomaly window."""
        return self.t_end - self.t_start if self.exists else 0.0

    def contains(self, times: np.ndarray) -> np.ndarray:
        """Return a mask of times lying inside the window."""
        if not self.exists:
            return np.zeros(np.shape(times), dtype=bool)
        return (times >= self.t_start) & (times <= self.t_end)


@dataclass(frozen=True, eq=False)
class EventSetup:
    """Per-event quantities that do not depend on the injected planet.

    Building this once per event and reusing it across the whole ``(q, s,
    alpha)`` grid keeps the noise realisation fixed and avoids recomputing the
    baseline PSPL curve tens of thousands of times.

    Attributes:
        event: The baseline event.
        times: Every survey visit time.
        season_index: Season of each visit.
        pspl_magnification: Baseline PSPL magnification at each visit.
        unit_noise: Standard-normal draw for each visit.
        f_source: True source flux.
        f_blend: True blend flux.
    """

    event: Event
    times: np.ndarray
    season_index: np.ndarray
    pspl_magnification: np.ndarray
    unit_noise: np.ndarray
    f_source: float
    f_blend: float

    @classmethod
    def build(
        cls, event: Event, survey: Survey, config: Config, realisation: int = 0
    ) -> EventSetup:
        """Precompute the planet-independent quantities for one event.

        Args:
            event: The baseline event.
            survey: The realised calendar.
            config: The run configuration.
            realisation: Selects the noise draw.  Planet injections all use
                realisation ``0`` so that the whole grid shares one noise
                realisation per event; the ``q = 0`` controls use a distinct
                realisation each, since otherwise every control on a given
                event would be the same light curve and could not measure a
                false-positive rate.
        """
        magnification = pspl_magnification(
            event, survey.times, finite_source=config.injection.finite_source
        )
        rng = generator(config.run.seed, "photometric_noise", event.index, realisation)
        f_source = float(survey.flux_from_mag(event.source_mag))
        return cls(
            event=event,
            times=survey.times,
            season_index=survey.season_index,
            pspl_magnification=magnification,
            unit_noise=rng.normal(0.0, 1.0, size=survey.times.size),
            f_source=f_source,
            f_blend=f_source * event.blend_ratio,
        )


@dataclass(frozen=True, eq=False)
class Injection:
    """One injected light curve and the truth needed to score it.

    Attributes:
        planet: The injected planet.
        light_curve: The perturbed, noisy light curve, restricted to the
            analysis window and with saturated points removed.
        binary_magnification: Injected binary magnification at those times.
        pspl_magnification: Baseline PSPL magnification at the same times, for
            diagnostics only -- the detection statistic must use a *refit*.
        anomaly: The anomaly window.
        n_finite_source_points: How many points used the expensive method.
    """

    planet: Planet
    light_curve: LightCurve
    binary_magnification: np.ndarray
    pspl_magnification: np.ndarray
    anomaly: AnomalyWindow
    n_finite_source_points: int

    @property
    def event(self) -> Event:
        """The underlying baseline event."""
        return self.light_curve.event

    def points_in_anomaly(self) -> int:
        """Number of measurements inside the anomaly window."""
        return int(self.anomaly.contains(self.light_curve.times).sum())


def caustic_extent(planet: Planet, n_points: int) -> tuple[np.ndarray, float]:
    """Return the caustic point cloud and its maximum distance from the origin.

    Args:
        planet: The planet, which must not be a control.
        n_points: Number of points sampled along the caustic curve.

    Returns:
        A ``(N, 2)`` array of caustic positions in the centre-of-mass frame,
        and the largest distance of any of them from the origin.
    """
    import MulensModel as mm

    x, y = mm.CausticsBinary(q=planet.q, s=planet.s).get_caustics(n_points=n_points)
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return points, float(np.hypot(points[:, 0], points[:, 1]).max())


def analysis_half_width(event: Event, planet: Planet, config: Config) -> float:
    """Return the half-width in days of the data window kept for this injection.

    The window is centred on ``t_0`` and is at least
    ``injection.analysis_window_t_E`` Einstein times wide, but is widened when
    the caustic lies further out -- which it does for very close and very wide
    separations, where the planetary caustic sits near ``|s - 1/s|`` Einstein
    radii from the host and would otherwise fall outside the window entirely.

    Args:
        event: The baseline event.
        planet: The injected planet.
        config: The run configuration.

    Returns:
        The half-width in days.
    """
    base = config.injection.analysis_window_t_E
    if planet.is_control:
        return base * event.t_E
    _, extent = caustic_extent(planet, config.injection.n_caustic_points)
    return max(base, extent + _CAUSTIC_MARGIN) * event.t_E


def _binary_parameters(event: Event, planet: Planet, *, finite_source: bool) -> dict[str, float]:
    """Return the MulensModel parameter dictionary for the perturbed model."""
    params = {
        "t_0": event.t_0,
        "u_0": event.u_0,
        "t_E": event.t_E,
        "q": planet.q,
        "s": planet.s,
        "alpha": planet.alpha_deg,
    }
    if finite_source:
        params["rho"] = event.rho
    return params


def _finite_source_intervals(
    times: np.ndarray,
    parameters: mm.ModelParameters,
    caustics: np.ndarray,
    radius: float,
) -> list[tuple[float, float]]:
    """Return time intervals where the source is close to a caustic.

    Args:
        times: Times to classify.
        parameters: MulensModel parameters, used for the source trajectory.
        caustics: Caustic point cloud.
        radius: Proximity threshold in Einstein radii.

    Returns:
        Merged ``(start, end)`` intervals, in increasing time order.
    """
    import MulensModel as mm
    from scipy.spatial import cKDTree

    trajectory = mm.Trajectory(times, parameters)
    distance, _ = cKDTree(caustics).query(np.column_stack([trajectory.x, trajectory.y]))
    close = distance < radius
    if not close.any():
        return []
    edges = np.flatnonzero(np.diff(close.astype(np.int8)))
    starts = [0] if close[0] else []
    ends: list[int] = []
    for edge in edges:
        if close[edge + 1]:
            starts.append(int(edge) + 1)
        else:
            ends.append(int(edge))
    if close[-1]:
        ends.append(times.size - 1)
    spacing = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    return [
        (float(times[a] - spacing), float(times[b] + spacing))
        for a, b in zip(starts, ends, strict=True)
    ]


def binary_magnification(
    event: Event, planet: Planet, times: np.ndarray, config: Config
) -> tuple[np.ndarray, int]:
    """Return the binary-lens magnification at the given times.

    Args:
        event: The baseline event.
        planet: The injected planet.
        times: Times to evaluate, which must be sorted.
        config: The run configuration.

    Returns:
        The magnification, and the number of points evaluated with the
        finite-source method.
    """
    import MulensModel as mm

    injection = config.injection
    if planet.is_control:
        return pspl_magnification(event, times, finite_source=injection.finite_source), 0
    if times.size == 0:
        return np.empty(0, dtype=float), 0

    parameters = _binary_parameters(event, planet, finite_source=injection.finite_source)
    model = mm.Model(parameters)
    n_finite = 0
    if injection.finite_source:
        caustics, _ = caustic_extent(planet, injection.n_caustic_points)
        intervals = _finite_source_intervals(
            times,
            mm.ModelParameters(parameters),
            caustics,
            injection.finite_source_radii * event.rho,
        )
        if intervals:
            methods: list[float | str] = []
            for start, end in intervals:
                if methods:
                    methods.extend([injection.point_source_method])
                methods.extend([start, injection.binary_method, end])
            model.set_magnification_methods(methods)
            n_finite = int(sum(int(((times >= a) & (times <= b)).sum()) for a, b in intervals))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FINITE_SOURCE_WARNING, category=UserWarning)
        magnification = np.asarray(model.get_magnification(times), dtype=float)
    return magnification, n_finite


def locate_anomaly(event: Event, planet: Planet, survey: Survey, config: Config) -> AnomalyWindow:
    """Locate the anomaly on a dense time grid, ignoring the observing cadence.

    The window is defined by detectability: the times at which the flux
    difference between the perturbed and baseline models exceeds
    ``injection.anomaly_sigma`` photometric sigma.  Because the grid is
    independent of the cadence, an anomaly falling entirely inside a season gap
    is still located, and is then correctly reported as carrying no data.

    Args:
        event: The baseline event.
        planet: The injected planet.
        survey: The survey, for the flux scale and the error model.
        config: The run configuration.

    Returns:
        The anomaly window.  ``exists`` is false for a control injection or
        when the deviation never reaches the threshold.
    """
    empty = AnomalyWindow(False, float("nan"), float("nan"), float("nan"), 0.0)
    if planet.is_control:
        return empty
    half = analysis_half_width(event, planet, config)
    grid = np.linspace(
        event.t_0 - half, event.t_0 + half, config.injection.anomaly_grid_points, dtype=float
    )
    binary, _ = binary_magnification(event, planet, grid, config)
    single = pspl_magnification(event, grid, finite_source=config.injection.finite_source)
    f_source = float(survey.flux_from_mag(event.source_mag))
    f_blend = f_source * event.blend_ratio
    binary_flux = f_source * binary + f_blend
    single_flux = f_source * single + f_blend
    deviation = np.abs(binary_flux - single_flux) / np.asarray(survey.flux_uncertainty(binary_flux))
    above = deviation > config.injection.anomaly_sigma
    if not above.any():
        return empty
    inside = np.flatnonzero(above)
    peak = int(np.argmax(deviation))
    return AnomalyWindow(
        exists=True,
        t_start=float(grid[inside[0]]),
        t_end=float(grid[inside[-1]]),
        t_peak=float(grid[peak]),
        peak_deviation=float(deviation[peak]),
    )


def inject_planet(setup: EventSetup, planet: Planet, survey: Survey, config: Config) -> Injection:
    """Inject a planetary perturbation into an event and re-observe it.

    The light curve is restricted to the analysis window, the binary
    magnification replaces the PSPL one, uncertainties are recomputed from the
    perturbed model flux, saturated points are dropped, and the event's own
    standard-normal draws are rescaled by the new uncertainties.

    Args:
        setup: The precomputed per-event quantities.
        planet: The planet to inject.  ``q = 0`` produces the planet-free
            control through exactly the same code path.
        survey: The realised calendar and error model.
        config: The run configuration.

    Returns:
        The injection, ready to be scored by :mod:`lenseff.detect`.
    """
    event = setup.event
    half = analysis_half_width(event, planet, config)
    inside = np.flatnonzero(np.abs(setup.times - event.t_0) <= half)
    times = setup.times[inside]
    binary, n_finite = binary_magnification(event, planet, times, config)
    model_flux = setup.f_source * binary + setup.f_blend
    keep = ~survey.is_saturated(model_flux)
    selected = inside[keep]
    times = times[keep]
    binary = binary[keep]
    model_flux = model_flux[keep]
    flux_err = np.asarray(survey.flux_uncertainty(model_flux))
    flux = model_flux + setup.unit_noise[selected] * flux_err
    light_curve = LightCurve(
        times=times,
        flux=flux,
        flux_err=flux_err,
        season_index=setup.season_index[selected],
        f_source=setup.f_source,
        f_blend=setup.f_blend,
        magnification=binary,
        event=event,
    )
    return Injection(
        planet=planet,
        light_curve=light_curve,
        binary_magnification=binary,
        pspl_magnification=setup.pspl_magnification[selected],
        anomaly=locate_anomaly(event, planet, survey, config),
        n_finite_source_points=n_finite,
    )
