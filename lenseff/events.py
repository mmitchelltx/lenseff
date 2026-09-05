"""Baseline PSPL event sampling and light-curve simulation.

Events are either drawn from the population model in ``events.distributions``
or read from a supplied catalog.  Either way an :class:`Event` is the complete
description of a planet-free microlensing event, and
:func:`simulate_light_curve` renders it onto a :class:`~lenseff.survey.Survey`
calendar with noise.

All magnification is computed by ``MulensModel``; this module never implements
a magnification formula of its own.

Determinism
-----------
Event ``i`` is drawn from the ``events`` stream at index ``i`` of the
*candidate* sequence, and its noise from the ``photometric_noise`` stream at
its own index.  Increasing ``n_events`` therefore extends the sample without
changing the events already in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from lenseff.config import Config, DistributionSpec
from lenseff.rng import generator
from lenseff.survey import Survey

if TYPE_CHECKING:  # pragma: no cover - typing only
    import MulensModel as mm

__all__ = [
    "Event",
    "LightCurve",
    "load_catalog",
    "pspl_model",
    "sample_distribution",
    "sample_events",
    "simulate_light_curve",
]

#: Maximum candidate draws per accepted event before giving up.
MAX_REJECTION_ATTEMPTS: Final[int] = 10_000

#: Finite-source magnification is evaluated within this many source radii of
#: the peak; outside, the point-source formula is accurate to well below the
#: photometric precision of any realistic survey.
_FINITE_SOURCE_RADII: Final[float] = 10.0


@dataclass(frozen=True, slots=True)
class Event:
    """A planet-free (point-source point-lens) microlensing event.

    Attributes:
        index: Position in the sampled event list.
        t_0: HJD of peak magnification.
        u_0: Impact parameter in Einstein radii.
        t_E: Einstein radius crossing time in days.
        source_mag: Unmagnified source magnitude.
        blend_ratio: Blend flux divided by source flux, ``f_b / f_s``.
        rho: Source angular radius in Einstein radii.
    """

    index: int
    t_0: float
    u_0: float
    t_E: float
    source_mag: float
    blend_ratio: float
    rho: float

    def pspl_parameters(self) -> dict[str, float]:
        """Return the MulensModel parameter dictionary for this event."""
        return {"t_0": self.t_0, "u_0": self.u_0, "t_E": self.t_E, "rho": self.rho}

    def as_record(self) -> dict[str, float | int]:
        """Return a flat record for the output table."""
        return {
            "event_index": self.index,
            "event_t_0": self.t_0,
            "event_u_0": self.u_0,
            "event_t_E": self.t_E,
            "event_source_mag": self.source_mag,
            "event_blend_ratio": self.blend_ratio,
            "event_rho": self.rho,
        }


@dataclass(frozen=True, eq=False)
class LightCurve:
    """A simulated light curve in flux units.

    Attributes:
        times: HJD of each measurement.
        flux: Observed flux (electrons per second), noise included.
        flux_err: 1-sigma uncertainty on each measurement.
        season_index: Season each measurement belongs to.
        f_source: True unmagnified source flux.
        f_blend: True blend flux.
        magnification: The noiseless magnification actually used, retained for
            diagnostics.  Detection code must never read it.
        event: The event this light curve was generated from.
    """

    times: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    season_index: np.ndarray
    f_source: float
    f_blend: float
    magnification: np.ndarray
    event: Event

    @property
    def n_points(self) -> int:
        """Number of measurements."""
        return int(self.times.size)

    def model_flux(self, magnification: np.ndarray, f_source: float, f_blend: float) -> np.ndarray:
        """Return model flux for a magnification curve and a flux pair."""
        return f_source * magnification + f_blend

    def chi2(self, model_flux: np.ndarray) -> float:
        """Return the chi-square of a model flux curve against the data."""
        return float(np.sum(((self.flux - model_flux) / self.flux_err) ** 2))

    def with_flux(self, flux: np.ndarray, magnification: np.ndarray) -> LightCurve:
        """Return a copy carrying different flux values (used by injection)."""
        return LightCurve(
            times=self.times,
            flux=flux,
            flux_err=self.flux_err,
            season_index=self.season_index,
            f_source=self.f_source,
            f_blend=self.f_blend,
            magnification=magnification,
            event=self.event,
        )


def sample_distribution(
    spec: DistributionSpec, rng: np.random.Generator, size: int | None = None
) -> np.ndarray | float:
    """Draw from a validated :class:`~lenseff.config.DistributionSpec`.

    Truncation bounds are applied by resampling, which is exact but requires
    the bounds to admit a reasonable acceptance rate.

    Args:
        spec: The distribution specification.
        rng: Generator to draw from.
        size: Number of draws; ``None`` returns a scalar.

    Returns:
        The draw(s).

    Raises:
        RuntimeError: If truncation rejects too many consecutive draws.
    """
    n = 1 if size is None else size
    out = np.empty(n, dtype=float)
    filled = 0
    attempts = 0
    while filled < n:
        attempts += 1
        if attempts > MAX_REJECTION_ATTEMPTS:
            raise RuntimeError(
                f"truncation bounds on a {spec.dist!r} distribution reject almost every "
                f"draw; check truncate_min/truncate_max"
            )
        candidate = _draw_untruncated(spec, rng, n - filled)
        if spec.lower is not None:
            candidate = candidate[candidate >= spec.lower]
        if spec.upper is not None:
            candidate = candidate[candidate <= spec.upper]
        take = min(candidate.size, n - filled)
        out[filled : filled + take] = candidate[:take]
        filled += take
    return out if size is not None else float(out[0])


def _draw_untruncated(spec: DistributionSpec, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ``n`` values from a distribution family, ignoring truncation."""
    p = spec.params
    if spec.dist == "fixed":
        return np.full(n, p["value"])
    if spec.dist == "uniform":
        return rng.uniform(p["min"], p["max"], size=n)
    if spec.dist == "loguniform":
        return np.exp(rng.uniform(np.log(p["min"]), np.log(p["max"]), size=n))
    if spec.dist == "normal":
        return rng.normal(p["mean"], p["std"], size=n)
    if spec.dist == "lognormal":
        return p["median"] * np.exp(rng.normal(0.0, p["sigma_ln"], size=n))
    if spec.dist == "powerlaw":
        # inverse-CDF sampling of dN/dx ∝ x**slope on [min, max]
        slope, lo, hi = p["slope"], p["min"], p["max"]
        u = rng.uniform(0.0, 1.0, size=n)
        if np.isclose(slope, -1.0):
            return lo * (hi / lo) ** u
        power = slope + 1.0
        return (lo**power + u * (hi**power - lo**power)) ** (1.0 / power)
    raise ValueError(f"unhandled distribution family {spec.dist!r}")  # pragma: no cover


def _sample_t_0(rng: np.random.Generator, survey: Survey) -> float:
    """Draw ``t_0`` uniformly over the union of the season windows."""
    windows = survey.seasons.windows
    lengths = np.array([end - start for start, end in windows])
    which = rng.choice(len(windows), p=lengths / lengths.sum())
    start, end = windows[which]
    return float(rng.uniform(start, end))


def _draw_candidate(config: Config, survey: Survey, attempt: int, index: int) -> Event | None:
    """Draw one candidate event and apply the acceptance cuts.

    Returns:
        The event, or ``None`` if it fails a cut.
    """
    rng = generator(config.run.seed, "events", attempt)
    dists = config.events.distributions
    t_0 = (
        float(sample_distribution(dists["t_0"], rng))
        if "t_0" in dists
        else _sample_t_0(rng, survey)
    )
    u_0 = float(sample_distribution(dists["u_0"], rng))
    t_E = float(sample_distribution(dists["t_E"], rng))
    source_mag = float(sample_distribution(dists["source_mag"], rng))
    blend_ratio = float(sample_distribution(dists["blend_ratio"], rng))
    rho = float(sample_distribution(dists["rho"], rng))
    event = Event(
        index=index,
        t_0=t_0,
        u_0=u_0,
        t_E=t_E,
        source_mag=source_mag,
        blend_ratio=blend_ratio,
        rho=rho,
    )
    return event if _passes_cuts(event, config, survey) else None


def _passes_cuts(event: Event, config: Config, survey: Survey) -> bool:
    """Apply the ``events`` acceptance cuts to a candidate."""
    phot = survey.photometry
    baseline_mag = event.source_mag - 2.5 * np.log10(1.0 + event.blend_ratio)
    if baseline_mag > phot.faint_limit_mag or baseline_mag < phot.saturation_mag:
        return False
    if config.events.require_peak_in_season and not bool(survey.in_season(event.t_0)[0]):
        return False
    half_width = config.events.peak_window_t_E * event.t_E
    return survey.points_in_window(event.t_0, half_width) >= config.events.min_points_near_peak


def sample_events(config: Config, survey: Survey) -> tuple[Event, ...]:
    """Draw the baseline event population.

    Args:
        config: The run configuration.
        survey: The realised observing calendar, used by the acceptance cuts.

    Returns:
        Exactly ``config.events.n_events`` accepted events.

    Raises:
        RuntimeError: If the acceptance cuts reject nearly every candidate.
    """
    if config.events.source == "catalog":
        return load_catalog(config, survey)
    accepted: list[Event] = []
    attempt = 0
    budget = MAX_REJECTION_ATTEMPTS * max(config.events.n_events, 1)
    while len(accepted) < config.events.n_events:
        if attempt >= budget:
            raise RuntimeError(
                f"only {len(accepted)} of {config.events.n_events} events passed the cuts "
                f"in {attempt} draws; loosen events.min_points_near_peak, the magnitude "
                f"limits, or the t_E prior"
            )
        event = _draw_candidate(config, survey, attempt, len(accepted))
        attempt += 1
        if event is not None:
            accepted.append(event)
    return tuple(accepted)


def load_catalog(config: Config, survey: Survey) -> tuple[Event, ...]:
    """Read baseline events from a parquet or CSV catalog.

    The catalog must provide the columns ``t_0``, ``u_0``, ``t_E``,
    ``source_mag``, ``blend_ratio`` and ``rho``.  The same acceptance cuts are
    applied as in population mode, so a catalog and a population run are
    directly comparable.

    Args:
        config: The run configuration.
        survey: The realised observing calendar.

    Returns:
        Up to ``config.events.n_events`` events, in catalog order.

    Raises:
        ValueError: If the file type is unsupported or columns are missing.
    """
    import pandas as pd

    path = config.events.catalog_path
    if path is None:  # pragma: no cover - guarded by config validation
        raise ValueError("events.catalog_path is required in catalog mode")
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported catalog format {path.suffix!r}; use .parquet or .csv")
    required = ["t_0", "u_0", "t_E", "source_mag", "blend_ratio", "rho"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"catalog {path} is missing column(s) {missing}")
    events: list[Event] = []
    for row in frame.itertuples(index=False):
        event = Event(
            index=len(events),
            t_0=float(row.t_0),
            u_0=float(row.u_0),
            t_E=float(row.t_E),
            source_mag=float(row.source_mag),
            blend_ratio=float(row.blend_ratio),
            rho=float(row.rho),
        )
        if _passes_cuts(event, config, survey):
            events.append(event)
        if len(events) == config.events.n_events:
            break
    return tuple(events)


def pspl_model(event: Event, *, finite_source: bool) -> mm.Model:
    """Build the MulensModel PSPL model for an event.

    Args:
        event: The event.
        finite_source: Whether to evaluate finite-source magnification near
            the peak.  Only matters when ``u_0`` is comparable to ``rho``.

    Returns:
        A configured ``MulensModel.Model``.
    """
    import MulensModel as mm

    if not finite_source:
        return mm.Model({"t_0": event.t_0, "u_0": event.u_0, "t_E": event.t_E})
    model = mm.Model(event.pspl_parameters())
    half = _FINITE_SOURCE_RADII * event.rho * event.t_E
    model.set_magnification_methods(
        [event.t_0 - half, "finite_source_uniform_Gould94", event.t_0 + half]
    )
    return model


def pspl_magnification(event: Event, times: np.ndarray, *, finite_source: bool) -> np.ndarray:
    """Return the PSPL magnification of an event at the given times."""
    return np.asarray(pspl_model(event, finite_source=finite_source).get_magnification(times))


def simulate_light_curve(
    event: Event,
    survey: Survey,
    config: Config,
    *,
    add_noise: bool = True,
    realisation: int = 0,
) -> LightCurve:
    """Render an event onto the survey calendar, with noise.

    Saturated measurements are dropped, which is what a real pipeline does and
    which matters for bright, highly magnified events.

    Uncertainties are computed from the *noiseless* model flux rather than
    from the realised flux.  Deriving sigma from the noisy value correlates
    the weight with the deviation and biases chi-square low; using the model
    value is the standard choice in simulation work.

    Args:
        event: The event to simulate.
        survey: The realised calendar.
        config: The run configuration, for the master seed and the
            finite-source switch.
        add_noise: Set false to obtain the noiseless light curve, which is
            useful in tests and diagnostics.
        realisation: Selects the noise draw, matching
            :meth:`lenseff.inject.EventSetup.build` so that the two agree
            point for point.

    Returns:
        The simulated light curve.
    """
    magnification = pspl_magnification(
        event, survey.times, finite_source=config.injection.finite_source
    )
    f_source = float(survey.flux_from_mag(event.source_mag))
    f_blend = f_source * event.blend_ratio
    model_flux = f_source * magnification + f_blend
    keep = ~survey.is_saturated(model_flux)
    times = survey.times[keep]
    magnification = magnification[keep]
    model_flux = model_flux[keep]
    season_index = survey.season_index[keep]
    flux_err = np.asarray(survey.flux_uncertainty(model_flux))
    if add_noise:
        rng = generator(config.run.seed, "photometric_noise", event.index, realisation)
        flux = model_flux + rng.normal(0.0, 1.0, size=model_flux.size) * flux_err
    else:
        flux = model_flux.copy()
    return LightCurve(
        times=times,
        flux=flux,
        flux_err=flux_err,
        season_index=season_index,
        f_source=f_source,
        f_blend=f_blend,
        magnification=magnification,
        event=event,
    )
