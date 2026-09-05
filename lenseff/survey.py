"""Observing calendar, photometric error model and saturation limits.

A :class:`Survey` is the frozen realisation of a ``survey:`` configuration
block: the exact list of visit times, which season each visit belongs to, and
the mapping from flux to uncertainty.  Everything downstream -- event
sampling, injection, detection -- sees the survey only through this object, so
switching from Roman to a ground-based cadence is a configuration change and
nothing more.

Units
-----
Fluxes are in detected electrons per second, so a magnitude ``m`` corresponds
to ``10 ** (-0.4 * (m - zero_point))``.  Times are HJD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from lenseff.config import PhotometryConfig, SeasonConfig, SurveyConfig
from lenseff.rng import generator

__all__ = ["MAG_PER_DEX", "Survey"]

#: ``2.5 / ln(10)``: converts a fractional flux error to a magnitude error.
MAG_PER_DEX: Final[float] = 2.5 / np.log(10.0)


@dataclass(frozen=True, eq=False)
class Survey:
    """A realised observing calendar plus its photometric error model.

    Attributes:
        config: The configuration block this survey was built from.
        times: HJD of every visit that survived dropout, sorted ascending.
        season_index: Season number of each visit, parallel to ``times``.
    """

    config: SurveyConfig
    times: np.ndarray
    season_index: np.ndarray

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: SurveyConfig, seed: int) -> Survey:
        """Realise the observing calendar.

        Visits are laid down on a regular cadence inside each season window;
        a fraction ``config.dropout_fraction`` is then discarded using the
        ``survey_dropout`` random stream, so the calendar depends on the seed
        but not on anything else in the run.

        Args:
            config: The survey configuration.
            seed: The run's master seed.

        Returns:
            The realised survey.
        """
        cadence = config.cadence_days
        chunks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for index, (start, end) in enumerate(config.seasons.windows):
            n = int(np.floor((end - start) / cadence)) + 1
            season_times = start + cadence * np.arange(n, dtype=float)
            chunks.append(season_times)
            labels.append(np.full(n, index, dtype=np.int32))
        times = np.concatenate(chunks)
        season_index = np.concatenate(labels)
        if config.dropout_fraction > 0.0:
            rng = generator(seed, "survey_dropout")
            keep = rng.random(times.size) >= config.dropout_fraction
            times = times[keep]
            season_index = season_index[keep]
        times.flags.writeable = False
        season_index.flags.writeable = False
        return cls(config=config, times=times, season_index=season_index)

    # -- calendar ----------------------------------------------------------

    @property
    def seasons(self) -> SeasonConfig:
        """The season definition."""
        return self.config.seasons

    @property
    def photometry(self) -> PhotometryConfig:
        """The photometric error model configuration."""
        return self.config.photometry

    @property
    def n_points(self) -> int:
        """Number of visits in the realised calendar."""
        return int(self.times.size)

    @property
    def duty_cycle(self) -> float:
        """Fraction of the survey span that lies inside a season."""
        seasons = self.seasons
        return seasons.n_seasons * seasons.length_days / seasons.total_span_days

    def season_of(self, times: np.ndarray | float) -> np.ndarray:
        """Return the season index of arbitrary times, or ``-1`` in a gap.

        Args:
            times: HJD values.

        Returns:
            Integer array of season indices, ``-1`` where the time falls
            outside every season window.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        starts = np.array([w[0] for w in self.seasons.windows])
        ends = np.array([w[1] for w in self.seasons.windows])
        index = np.searchsorted(starts, t, side="right") - 1
        valid = index >= 0
        safe = np.clip(index, 0, len(starts) - 1)
        inside = valid & (t <= ends[safe])
        return np.where(inside, safe, -1).astype(np.int32)

    def in_season(self, times: np.ndarray | float) -> np.ndarray:
        """Return a boolean mask of times falling inside an observing season."""
        return self.season_of(times) >= 0

    def points_in_window(self, centre: float, half_width: float) -> int:
        """Count visits within ``half_width`` days of ``centre``."""
        lo = np.searchsorted(self.times, centre - half_width, side="left")
        hi = np.searchsorted(self.times, centre + half_width, side="right")
        return int(hi - lo)

    def window_slice(self, start: float, end: float) -> slice:
        """Return a slice selecting the visits inside ``[start, end]``."""
        left = int(np.searchsorted(self.times, start, side="left"))
        right = int(np.searchsorted(self.times, end, side="right"))
        return slice(left, right)

    # -- photometry --------------------------------------------------------

    def flux_from_mag(self, mag: np.ndarray | float) -> np.ndarray:
        """Convert magnitudes to flux in electrons per second."""
        return np.power(10.0, -0.4 * (np.asarray(mag, dtype=float) - self.photometry.zero_point))

    def mag_from_flux(self, flux: np.ndarray | float) -> np.ndarray:
        """Convert flux in electrons per second to magnitudes.

        Non-positive fluxes map to ``inf`` rather than raising, because a
        noisy measurement of a faint blend can legitimately go negative.
        """
        f = np.asarray(flux, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = self.photometry.zero_point - 2.5 * np.log10(f)
        return np.where(f > 0.0, mag, np.inf)

    def flux_uncertainty(self, flux: np.ndarray | float) -> np.ndarray:
        """Return the 1-sigma flux uncertainty for a given total flux.

        The variance is source shot noise plus sky shot noise plus read noise,
        converted back to a rate, with a fractional systematic floor added in
        quadrature and the whole thing scaled by ``precision_scale``.

        Args:
            flux: Total in-aperture flux in electrons per second.

        Returns:
            Uncertainty in electrons per second, same shape as ``flux``.
        """
        phot = self.photometry
        f = np.clip(np.asarray(flux, dtype=float), 0.0, None)
        integration = phot.exposure_time_s * phot.n_exposures
        source_e = f * integration
        sky_e = phot.sky_e_per_s_per_pixel * phot.n_pixels * integration
        read_e2 = phot.n_pixels * phot.n_exposures * phot.read_noise_e**2
        sigma_rate = np.sqrt(source_e + sky_e + read_e2) / integration
        floor = (phot.systematic_floor_mmag / 1000.0) / MAG_PER_DEX * f
        return np.hypot(sigma_rate, floor) * self.config.precision_scale

    def magnitude_uncertainty(self, mag: np.ndarray | float) -> np.ndarray:
        """Return the 1-sigma magnitude uncertainty at a given magnitude."""
        flux = self.flux_from_mag(mag)
        return MAG_PER_DEX * self.flux_uncertainty(flux) / flux

    def is_saturated(self, flux: np.ndarray | float) -> np.ndarray:
        """Return a mask of fluxes brighter than the saturation limit."""
        return np.asarray(flux, dtype=float) > self.flux_from_mag(self.photometry.saturation_mag)

    def is_too_faint(self, flux: np.ndarray | float) -> np.ndarray:
        """Return a mask of fluxes fainter than the faint limit."""
        return np.asarray(flux, dtype=float) < self.flux_from_mag(self.photometry.faint_limit_mag)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, float | int | str]:
        """Return a small dictionary describing the realised calendar."""
        return {
            "preset": self.config.preset,
            "band": self.photometry.band,
            "cadence_minutes": self.config.cadence_minutes,
            "n_seasons": self.seasons.n_seasons,
            "season_length_days": self.seasons.length_days,
            "n_points": self.n_points,
            "span_days": float(self.times[-1] - self.times[0]) if self.n_points else 0.0,
            "duty_cycle": self.duty_cycle,
            "sigma_mag_at_20": float(self.magnitude_uncertainty(20.0)),
            "sigma_mag_at_23": float(self.magnitude_uncertainty(23.0)),
        }
