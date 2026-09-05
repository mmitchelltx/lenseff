"""Configuration schema, parsing and validation for :mod:`lenseff`.

Every knob in ``lenseff`` is set from a YAML file; nothing scientific is
hard-coded in the library.  This module turns a YAML document into a tree of
frozen dataclasses, rejecting unknown keys and out-of-range values eagerly so
that a typo can never silently change the meaning of a run.

The parsed tree is also the provenance record: :meth:`Config.canonical_json`
serialises the *resolved* configuration (presets merged, defaults filled in),
and :meth:`Config.config_hash` hashes it.  Two runs with the same config hash
and the same seed are required to produce bit-identical output.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import numpy as np
import yaml

__all__ = [
    "EVENT_PARAMETERS",
    "AxisSpec",
    "ComputeConfig",
    "Config",
    "ConfigError",
    "DetectionConfig",
    "DistributionSpec",
    "EventsConfig",
    "GridConfig",
    "InjectionConfig",
    "OutputConfig",
    "PhotometryConfig",
    "RefitConfig",
    "RunConfig",
    "SeasonConfig",
    "SurveyConfig",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a configuration document is malformed or out of range."""


class _Missing:
    """Sentinel distinguishing "absent" from a legitimate ``None`` default."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING: Final = _Missing()


def _fail(path: str, message: str) -> NoReturn:
    raise ConfigError(f"{path}: {message}")


class _Reader:
    """Typed, path-aware accessor over one mapping of a YAML document.

    The reader tracks which keys were consumed; :meth:`done` then rejects
    anything left over, which is what makes a mistyped key an error rather than
    a silently ignored setting.

    Args:
        data: The mapping to read.
        path: Dotted path of this mapping, used in error messages.
    """

    def __init__(self, data: Any, path: str) -> None:
        if not isinstance(data, Mapping):
            _fail(path, f"expected a mapping, got {type(data).__name__}")
        self._data: dict[str, Any] = dict(data)
        self._path = path
        self._used: set[str] = set()

    @property
    def path(self) -> str:
        """Dotted path of this mapping within the document."""
        return self._path

    def _child_path(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _raw(self, key: str, default: Any) -> Any:
        self._used.add(key)
        if key not in self._data:
            if isinstance(default, _Missing):
                _fail(self._child_path(key), "required key is missing")
            return default
        return self._data[key]

    def get_raw(self, key: str, default: Any = MISSING) -> Any:
        """Return a value with no type coercion."""
        return self._raw(key, default)

    def get_bool(self, key: str, default: Any = MISSING) -> bool:
        """Return a strictly boolean value."""
        value = self._raw(key, default)
        if not isinstance(value, bool):
            _fail(self._child_path(key), f"expected a boolean, got {value!r}")
        return value

    def get_int(
        self,
        key: str,
        default: Any = MISSING,
        *,
        ge: int | None = None,
        gt: int | None = None,
        le: int | None = None,
    ) -> int:
        """Return an integer value, optionally range-checked."""
        value = self._raw(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(self._child_path(key), f"expected an integer, got {value!r}")
        self._check_range(self._child_path(key), float(value), ge, gt, le, None)
        return value

    def get_float(
        self,
        key: str,
        default: Any = MISSING,
        *,
        ge: float | None = None,
        gt: float | None = None,
        le: float | None = None,
        lt: float | None = None,
    ) -> float:
        """Return a float value, optionally range-checked."""
        value = self._raw(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            _fail(self._child_path(key), f"expected a number, got {value!r}")
        number = float(value)
        if not np.isfinite(number):
            _fail(self._child_path(key), f"expected a finite number, got {number!r}")
        self._check_range(self._child_path(key), number, ge, gt, le, lt)
        return number

    def get_optional_float(self, key: str, default: float | None = None) -> float | None:
        """Return a float value, or ``None`` when absent or explicitly null."""
        if self._data.get(key, None) is None:
            self._used.add(key)
            return default if key not in self._data else None
        return self.get_float(key)

    def get_str(
        self, key: str, default: Any = MISSING, *, choices: Sequence[str] | None = None
    ) -> str:
        """Return a string value, optionally restricted to ``choices``."""
        value = self._raw(key, default)
        if not isinstance(value, str):
            _fail(self._child_path(key), f"expected a string, got {value!r}")
        if choices is not None and value not in choices:
            _fail(
                self._child_path(key),
                f"expected one of {sorted(choices)}, got {value!r}",
            )
        return value

    def get_path(self, key: str, default: Any = MISSING) -> Path:
        """Return a filesystem path."""
        return Path(self.get_str(key, default))

    def get_floats(
        self,
        key: str,
        default: Any = MISSING,
        *,
        length: int | None = None,
        increasing: bool = False,
    ) -> tuple[float, ...]:
        """Return a tuple of floats, optionally length- and order-checked."""
        value = self._raw(key, default)
        if isinstance(value, str) or not isinstance(value, Sequence):
            _fail(self._child_path(key), f"expected a list of numbers, got {value!r}")
        out: list[float] = []
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int | float):
                _fail(f"{self._child_path(key)}[{i}]", f"expected a number, got {item!r}")
            out.append(float(item))
        if length is not None and len(out) != length:
            _fail(self._child_path(key), f"expected {length} values, got {len(out)}")
        if increasing and any(b <= a for a, b in itertools.pairwise(out)):
            _fail(self._child_path(key), f"values must be strictly increasing, got {out}")
        return tuple(out)

    def get_section(self, key: str, *, required: bool = True) -> _Reader | None:
        """Return a sub-reader for a nested mapping."""
        if key not in self._data:
            if required:
                _fail(self._child_path(key), "required section is missing")
            self._used.add(key)
            return None
        return _Reader(self._raw(key, MISSING), self._child_path(key))

    def require_section(self, key: str) -> _Reader:
        """Return a sub-reader for a nested mapping that must be present."""
        reader = self.get_section(key, required=True)
        if reader is None:  # pragma: no cover - get_section raises when missing
            _fail(self._child_path(key), "required section is missing")
        return reader

    def done(self) -> None:
        """Reject any keys that were never read.

        Raises:
            ConfigError: If the mapping contains keys not in the schema.
        """
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            known = sorted(self._used)
            _fail(
                self._path or "<root>",
                f"unknown key(s) {unknown}; recognised keys here are {known}",
            )

    @staticmethod
    def _check_range(
        path: str,
        value: float,
        ge: float | None,
        gt: float | None,
        le: float | None,
        lt: float | None,
    ) -> None:
        if ge is not None and value < ge:
            _fail(path, f"must be >= {ge}, got {value}")
        if gt is not None and value <= gt:
            _fail(path, f"must be > {gt}, got {value}")
        if le is not None and value > le:
            _fail(path, f"must be <= {le}, got {value}")
        if lt is not None and value >= lt:
            _fail(path, f"must be < {lt}, got {value}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Top-level bookkeeping for a single injection-recovery run.

    Attributes:
        name: Human-readable run label, used in output filenames.
        seed: Master RNG seed.  Every random draw in the package is derived
            from this value by :mod:`lenseff.rng`, so the run is reproducible
            regardless of worker count or task ordering.
        output_dir: Directory for parquet output, checkpoints and figures.
        overwrite: Whether an existing output directory may be overwritten.
    """

    name: str
    seed: int
    output_dir: Path
    overwrite: bool = False

    @classmethod
    def from_reader(cls, r: _Reader) -> RunConfig:
        """Parse a ``run`` section."""
        cfg = cls(
            name=r.get_str("name"),
            seed=r.get_int("seed", ge=0),
            output_dir=r.get_path("output_dir", "results"),
            overwrite=r.get_bool("overwrite", False),
        )
        r.done()
        return cfg


# --------------------------------------------------------------------------
# survey
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonConfig:
    """Observing-season windows.

    Seasons are the single most important structural feature of a microlensing
    survey's time sampling: a planetary anomaly lasting hours to days is simply
    invisible if it falls in a gap.

    Attributes:
        t_start: Time of the start of the first season (HJD).
        length_days: Duration of each season.
        start_offsets_days: Start time of each season, in days after
            ``t_start``.  Must be strictly increasing; the first entry is
            normally ``0``.
    """

    t_start: float
    length_days: float
    start_offsets_days: tuple[float, ...]

    @property
    def n_seasons(self) -> int:
        """Number of observing seasons."""
        return len(self.start_offsets_days)

    @property
    def windows(self) -> tuple[tuple[float, float], ...]:
        """Absolute ``(start, end)`` HJD bounds of each season."""
        return tuple(
            (self.t_start + off, self.t_start + off + self.length_days)
            for off in self.start_offsets_days
        )

    @property
    def total_span_days(self) -> float:
        """Time from the first observation to the last."""
        return self.start_offsets_days[-1] + self.length_days

    @classmethod
    def from_reader(cls, r: _Reader) -> SeasonConfig:
        """Parse a ``survey.seasons`` section."""
        length = r.get_float("length_days", gt=0.0)
        offsets = r.get_floats("start_offsets_days", increasing=True)
        cfg = cls(
            t_start=r.get_float("t_start"),
            length_days=length,
            start_offsets_days=offsets,
        )
        r.done()
        if not offsets:
            _fail(f"{r.path}.start_offsets_days", "at least one season is required")
        for a, b in itertools.pairwise(offsets):
            if b - a < length:
                _fail(
                    f"{r.path}.start_offsets_days",
                    f"seasons overlap: offsets {a} and {b} are closer than length_days={length}",
                )
        return cfg


@dataclass(frozen=True, slots=True)
class PhotometryConfig:
    """Photon-noise-plus-systematics error model for a single band.

    The uncertainty on a measurement is built from source counts, sky counts,
    read noise and a systematic floor, which reproduces the characteristic
    shape of a survey's precision-versus-magnitude curve without hard-coding
    any one survey.

    Attributes:
        band: Name of the photometric band (informational).
        zero_point: Magnitude producing 1 detected electron per second.
        exposure_time_s: Effective exposure time per visit.
        n_exposures: Exposures combined per visit.
        sky_e_per_s_per_pixel: Sky plus unresolved-star background rate.
        read_noise_e: Read noise per pixel per exposure.
        n_pixels: Effective number of pixels in the photometric aperture.
        systematic_floor_mmag: Systematic error floor added in quadrature,
            in millimagnitudes; bright-star precision saturates here.
        saturation_mag: Sources brighter than this are unusable.
        faint_limit_mag: Sources fainter than this are dropped from the
            sample.  Set very faint to disable.
    """

    band: str
    zero_point: float
    exposure_time_s: float
    n_exposures: int
    sky_e_per_s_per_pixel: float
    read_noise_e: float
    n_pixels: float
    systematic_floor_mmag: float
    saturation_mag: float
    faint_limit_mag: float

    @classmethod
    def from_reader(cls, r: _Reader) -> PhotometryConfig:
        """Parse a ``survey.photometry`` section."""
        cfg = cls(
            band=r.get_str("band"),
            zero_point=r.get_float("zero_point"),
            exposure_time_s=r.get_float("exposure_time_s", gt=0.0),
            n_exposures=r.get_int("n_exposures", 1, ge=1),
            sky_e_per_s_per_pixel=r.get_float("sky_e_per_s_per_pixel", ge=0.0),
            read_noise_e=r.get_float("read_noise_e", ge=0.0),
            n_pixels=r.get_float("n_pixels", gt=0.0),
            systematic_floor_mmag=r.get_float("systematic_floor_mmag", ge=0.0),
            saturation_mag=r.get_float("saturation_mag"),
            faint_limit_mag=r.get_float("faint_limit_mag"),
        )
        r.done()
        if cfg.faint_limit_mag <= cfg.saturation_mag:
            _fail(
                f"{r.path}.faint_limit_mag",
                f"must be fainter (numerically larger) than saturation_mag="
                f"{cfg.saturation_mag}, got {cfg.faint_limit_mag}",
            )
        return cfg


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    """Cadence, seasons and photometric performance of the survey.

    Attributes:
        preset: Name of the preset the configuration was built from, or
            ``"custom"``.
        cadence_minutes: Interval between consecutive visits within a season.
        seasons: Season window definition.
        photometry: Error model.
        precision_scale: Multiplies every photometric uncertainty.  ``0.5``
            means "twice as precise"; used by the validation test that
            efficiency improves with precision.
        dropout_fraction: Fraction of scheduled visits randomly discarded, a
            crude stand-in for downlink losses and bad-data flags.
    """

    preset: str
    cadence_minutes: float
    seasons: SeasonConfig
    photometry: PhotometryConfig
    precision_scale: float = 1.0
    dropout_fraction: float = 0.0

    @property
    def cadence_days(self) -> float:
        """Cadence expressed in days."""
        return self.cadence_minutes / 1440.0

    @classmethod
    def from_reader(cls, r: _Reader) -> SurveyConfig:
        """Parse a ``survey`` section (after preset resolution)."""
        preset = r.get_str("preset", "custom")
        cadence = r.get_float("cadence_minutes", gt=0.0)
        precision_scale = r.get_float("precision_scale", 1.0, gt=0.0)
        dropout = r.get_float("dropout_fraction", 0.0, ge=0.0, lt=1.0)
        seasons_reader = r.require_section("seasons")
        photometry_reader = r.require_section("photometry")
        r.done()
        return cls(
            preset=preset,
            cadence_minutes=cadence,
            seasons=SeasonConfig.from_reader(seasons_reader),
            photometry=PhotometryConfig.from_reader(photometry_reader),
            precision_scale=precision_scale,
            dropout_fraction=dropout,
        )


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

_DIST_PARAMS: Final[dict[str, tuple[str, ...]]] = {
    "fixed": ("value",),
    "uniform": ("min", "max"),
    "loguniform": ("min", "max"),
    "normal": ("mean", "std"),
    "lognormal": ("median", "sigma_ln"),
    "powerlaw": ("slope", "min", "max"),
}


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """A one-dimensional sampling distribution for an event parameter.

    Attributes:
        dist: Distribution family; one of the keys of ``_DIST_PARAMS``.
        params: Family-specific parameters, already validated.
        lower: Optional hard lower truncation bound.
        upper: Optional hard upper truncation bound.
    """

    dist: str
    params: dict[str, float]
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def from_reader(cls, r: _Reader) -> DistributionSpec:
        """Parse a distribution mapping such as ``{dist: uniform, min: 0, max: 1}``."""
        dist = r.get_str("dist", choices=tuple(_DIST_PARAMS))
        params = {name: r.get_float(name) for name in _DIST_PARAMS[dist]}
        lower = r.get_optional_float("truncate_min", None)
        upper = r.get_optional_float("truncate_max", None)
        r.done()
        if dist in {"uniform", "loguniform", "powerlaw"} and params["max"] <= params["min"]:
            _fail(r.path, f"max must exceed min, got min={params['min']}, max={params['max']}")
        if dist == "loguniform" and params["min"] <= 0.0:
            _fail(r.path, f"loguniform requires min > 0, got {params['min']}")
        if dist == "powerlaw" and params["min"] <= 0.0:
            _fail(r.path, f"powerlaw requires min > 0, got {params['min']}")
        if dist == "normal" and params["std"] <= 0.0:
            _fail(r.path, f"std must be positive, got {params['std']}")
        if dist == "lognormal":
            if params["median"] <= 0.0:
                _fail(r.path, f"median must be positive, got {params['median']}")
            if params["sigma_ln"] <= 0.0:
                _fail(r.path, f"sigma_ln must be positive, got {params['sigma_ln']}")
        if lower is not None and upper is not None and upper <= lower:
            _fail(r.path, f"truncate_max must exceed truncate_min, got {lower} and {upper}")
        return cls(dist=dist, params=params, lower=lower, upper=upper)


#: Event parameters that may carry a sampling distribution.  ``t_0`` is
#: optional: when omitted it is drawn uniformly over the union of the season
#: windows, which is the right prior once ``require_peak_in_season`` is set.
EVENT_PARAMETERS: Final[tuple[str, ...]] = (
    "t_0",
    "u_0",
    "t_E",
    "source_mag",
    "blend_ratio",
    "rho",
)


@dataclass(frozen=True, slots=True)
class EventsConfig:
    """How the baseline PSPL event population is obtained.

    Attributes:
        source: ``"population"`` to draw events from the distributions below,
            or ``"catalog"`` to read them from a file.
        n_events: Number of baseline events (population mode).
        catalog_path: Path to a parquet/CSV catalog (catalog mode).
        require_peak_in_season: Reject drawn events whose peak falls in a
            season gap.
        peak_window_t_E: Half-width, in Einstein times, of the window around
            ``t_0`` used to judge whether an event is adequately covered.
        min_points_near_peak: Reject events with fewer than this many usable
            (unsaturated, in-season) measurements inside that window.
        distributions: Per-parameter sampling specs, keyed by parameter name
            (``t_0``, ``u_0``, ``t_E``, ``source_mag``, ``blend_ratio``,
            ``rho``).
    """

    source: str
    n_events: int
    catalog_path: Path | None
    require_peak_in_season: bool
    peak_window_t_E: float
    min_points_near_peak: int
    distributions: dict[str, DistributionSpec] = field(default_factory=dict)

    @classmethod
    def from_reader(cls, r: _Reader) -> EventsConfig:
        """Parse an ``events`` section."""
        source = r.get_str("source", "population", choices=("population", "catalog"))
        n_events = r.get_int("n_events", ge=1)
        raw_catalog = r.get_raw("catalog_path", None)
        catalog_path = None if raw_catalog is None else Path(str(raw_catalog))
        require_peak = r.get_bool("require_peak_in_season", True)
        peak_window = r.get_float("peak_window_t_E", 2.0, gt=0.0)
        min_points = r.get_int("min_points_near_peak", 100, ge=0)
        dists: dict[str, DistributionSpec] = {}
        if source == "population":
            dist_reader = r.require_section("distributions")
            for name in EVENT_PARAMETERS:
                sub = dist_reader.get_section(name, required=(name != "t_0"))
                if sub is None:
                    continue
                dists[name] = DistributionSpec.from_reader(sub)
            dist_reader.done()
        else:
            r.get_section("distributions", required=False)
        r.done()
        if source == "catalog" and catalog_path is None:
            _fail(f"{r.path}.catalog_path", "required when events.source is 'catalog'")
        return cls(
            source=source,
            n_events=n_events,
            catalog_path=catalog_path,
            require_peak_in_season=require_peak,
            peak_window_t_E=peak_window,
            min_points_near_peak=min_points,
            distributions=dists,
        )


# --------------------------------------------------------------------------
# injection grid
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """A linearly spaced grid axis.

    Attributes:
        min: Lower edge (inclusive).
        max: Upper edge (inclusive).
        n: Number of points.  ``n = 1`` places a single point at ``min``.
    """

    min: float
    max: float
    n: int

    def values(self) -> np.ndarray:
        """Return the axis sample points."""
        if self.n == 1:
            return np.array([self.min], dtype=float)
        return np.linspace(self.min, self.max, self.n, dtype=float)

    @classmethod
    def from_reader(cls, r: _Reader) -> AxisSpec:
        """Parse an axis mapping such as ``{min: -5, max: -2, n: 13}``."""
        lo = r.get_float("min")
        hi = r.get_float("max")
        n = r.get_int("n", ge=1)
        r.done()
        if n > 1 and hi <= lo:
            _fail(r.path, f"max must exceed min when n > 1, got min={lo}, max={hi}")
        return cls(min=lo, max=hi, n=n)


@dataclass(frozen=True, slots=True)
class GridConfig:
    """The ``(log q, log s)`` grid and the angles marginalised over.

    Attributes:
        log_q: Base-10 log of the planet-to-star mass ratio.
        log_s: Base-10 log of the projected separation in Einstein radii.
        n_alpha: Number of source-trajectory angles per (event, cell).
        alpha_mode: ``"stratified"`` draws one angle uniformly within each of
            ``n_alpha`` equal bins of ``[0, 360)`` (lower variance, still
            unbiased); ``"random"`` draws them independently; ``"uniform"``
            uses a fixed regular grid of angles.
        events_per_cell: Number of baseline events used per grid cell.  Set to
            ``0`` to use every event in the sample.
    """

    log_q: AxisSpec
    log_s: AxisSpec
    n_alpha: int
    alpha_mode: str
    events_per_cell: int

    @property
    def n_cells(self) -> int:
        """Number of ``(q, s)`` cells."""
        return self.log_q.n * self.log_s.n

    @classmethod
    def from_reader(cls, r: _Reader) -> GridConfig:
        """Parse an ``injection.grid`` section."""
        log_q_reader = r.require_section("log_q")
        log_s_reader = r.require_section("log_s")
        n_alpha = r.get_int("n_alpha", ge=1)
        alpha_mode = r.get_str(
            "alpha_mode", "stratified", choices=("stratified", "random", "uniform")
        )
        events_per_cell = r.get_int("events_per_cell", 0, ge=0)
        r.done()
        return cls(
            log_q=AxisSpec.from_reader(log_q_reader),
            log_s=AxisSpec.from_reader(log_s_reader),
            n_alpha=n_alpha,
            alpha_mode=alpha_mode,
            events_per_cell=events_per_cell,
        )


@dataclass(frozen=True, slots=True)
class InjectionConfig:
    """How binary-lens light curves are generated.

    Attributes:
        grid: The parameter grid.
        finite_source: Whether to include finite-source effects (``rho``).
        binary_method: MulensModel method used where the source is close
            enough to a caustic for the source size to matter.
        point_source_method: MulensModel method used elsewhere in the
            analysis window.
        finite_source_radii: ``binary_method`` is applied wherever the source
            centre lies within this many source radii of the caustic.  The
            default is generous: the finite-source correction is already far
            below the photometric precision at the boundary, so the switch
            between methods introduces no visible step.
        n_caustic_points: Points used to represent the caustic curve when
            measuring the source-to-caustic distance.
        analysis_window_t_E: Half-width, in Einstein times, of the data window
            kept for injection and refitting.  Widened automatically when the
            planetary caustic lies further out, which it does for very close
            and very wide separations.
        anomaly_sigma: The anomaly window spans the times at which the flux
            difference between the binary and the baseline PSPL model exceeds
            this many photometric sigma.  Defining the window by
            detectability rather than by a fixed fractional deviation makes it
            shrink appropriately for low-mass planets and faint sources, and
            makes "did the anomaly land in a season gap" a sharp question.
        anomaly_grid_points: Resolution of the dense grid on which the anomaly
            window is located.  The window is found independently of the
            observing cadence, so an anomaly that falls entirely inside a
            season gap is still located and can be reported as unobserved.
        include_zero_q_control: Also run a ``q = 0`` (planet-free) control
            batch, which measures the false-positive rate of the criteria.
        n_zero_q_trials: Number of control injections.
    """

    grid: GridConfig
    finite_source: bool
    binary_method: str
    point_source_method: str
    finite_source_radii: float
    n_caustic_points: int
    analysis_window_t_E: float
    anomaly_sigma: float
    anomaly_grid_points: int
    include_zero_q_control: bool
    n_zero_q_trials: int

    @classmethod
    def from_reader(cls, r: _Reader) -> InjectionConfig:
        """Parse an ``injection`` section."""
        grid_reader = r.require_section("grid")
        cfg = cls(
            grid=GridConfig.from_reader(grid_reader),
            finite_source=r.get_bool("finite_source", True),
            binary_method=r.get_str("binary_method", "VBBL"),
            point_source_method=r.get_str("point_source_method", "point_source"),
            finite_source_radii=r.get_float("finite_source_radii", 20.0, gt=0.0),
            n_caustic_points=r.get_int("n_caustic_points", 500, ge=10),
            analysis_window_t_E=r.get_float("analysis_window_t_E", 3.0, gt=0.0),
            anomaly_sigma=r.get_float("anomaly_sigma", 1.0, gt=0.0),
            anomaly_grid_points=r.get_int("anomaly_grid_points", 2001, ge=101),
            include_zero_q_control=r.get_bool("include_zero_q_control", True),
            n_zero_q_trials=r.get_int("n_zero_q_trials", 0, ge=0),
        )
        r.done()
        return cfg


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefitConfig:
    """Controls the free PSPL refit that defines the detection statistic.

    The chi-square surface of a PSPL fit to an anomalous light curve is
    multimodal, so the refit is run from several starting points and the best
    result is kept.  Under-fitting here inflates efficiency, which is the
    failure mode this package exists to avoid.

    Attributes:
        n_starts: Number of candidate starting points screened by evaluating
            chi-square at each.  Screening is cheap: one magnification
            evaluation per candidate.
        n_refine: How many of the best-scoring candidates are handed to the
            optimiser.  This is the knob that trades robustness against
            compute; the screen makes a large ``n_starts`` nearly free, so
            widen that before raising this.
        method: Optimiser for the non-linear parameters.  ``least_squares``
            uses the residual vector directly and is several times faster than
            the simplex methods at equal accuracy.
        require_positive_source: Clamp a negative fitted source flux to zero.
            Negative *blend* flux is always allowed, as is standard.
        t_0_bound_t_E: Half-width, in Einstein times, of the box the fitted
            ``t_0`` may move in.
        t_E_bound_factor: The fitted ``t_E`` is confined to the truth divided
            and multiplied by this factor.
        u_0_bound_max: Upper bound on the fitted impact parameter.
        u_0_bound_min: Lower bound on the fitted impact parameter.
        max_iterations: Iteration cap per start.
        tolerance: Convergence tolerance passed to the optimiser.
        free_blending: Fit ``f_s`` and ``f_b`` (linearly, at fixed non-linear
            parameters) rather than holding the blend fixed.
        t_0_jitter_t_E: Spread, in units of ``t_E``, of the ``t_0`` offsets
            used to seed the multi-start.
        t_E_factors: Multiplicative factors applied to the truth ``t_E`` when
            seeding starts.
        u_0_factors: Multiplicative factors applied to the truth ``u_0``.
        mask_anomaly_start: If true, one additional start is seeded from a fit
            that masks the anomaly window, which is often the deepest minimum.
    """

    n_starts: int
    n_refine: int
    method: str
    require_positive_source: bool
    t_0_bound_t_E: float
    t_E_bound_factor: float
    u_0_bound_max: float
    u_0_bound_min: float
    max_iterations: int
    tolerance: float
    free_blending: bool
    t_0_jitter_t_E: float
    t_E_factors: tuple[float, ...]
    u_0_factors: tuple[float, ...]
    mask_anomaly_start: bool

    @classmethod
    def from_reader(cls, r: _Reader) -> RefitConfig:
        """Parse a ``detection.refit`` section."""
        cfg = cls(
            n_starts=r.get_int("n_starts", 28, ge=1),
            n_refine=r.get_int("n_refine", 4, ge=1),
            method=r.get_str(
                "method",
                "least_squares",
                choices=("least_squares", "Nelder-Mead", "Powell", "L-BFGS-B"),
            ),
            require_positive_source=r.get_bool("require_positive_source", True),
            t_0_bound_t_E=r.get_float("t_0_bound_t_E", 2.0, gt=0.0),
            t_E_bound_factor=r.get_float("t_E_bound_factor", 20.0, gt=1.0),
            u_0_bound_max=r.get_float("u_0_bound_max", 3.0, gt=0.0),
            u_0_bound_min=r.get_float("u_0_bound_min", 1e-5, gt=0.0),
            max_iterations=r.get_int("max_iterations", 5000, ge=1),
            tolerance=r.get_float("tolerance", 1e-6, gt=0.0),
            free_blending=r.get_bool("free_blending", True),
            t_0_jitter_t_E=r.get_float("t_0_jitter_t_E", 0.3, ge=0.0),
            t_E_factors=r.get_floats("t_E_factors", (0.7, 1.0, 1.4)),
            u_0_factors=r.get_floats("u_0_factors", (0.6, 1.0, 1.6)),
            mask_anomaly_start=r.get_bool("mask_anomaly_start", True),
        )
        r.done()
        for name, factors in (("t_E_factors", cfg.t_E_factors), ("u_0_factors", cfg.u_0_factors)):
            if not factors or any(f <= 0.0 for f in factors):
                _fail(f"{r.path}.{name}", f"all factors must be positive, got {list(factors)}")
        return cfg


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Detection criteria applied to each injected light curve.

    All three defaults are the ones the project specifies: a chi-square
    improvement threshold, a run of consecutive significant residuals, and a
    requirement that the deviation land inside an observing season.

    Attributes:
        delta_chi2_min: Minimum ``chi2(free PSPL refit) - chi2(binary)``.
        consecutive_points: Number of consecutive points that must deviate
            from the *refit* PSPL model.
        point_sigma: Per-point significance threshold for that run.
        require_same_sign: Require the run of deviant points to share a sign,
            which suppresses runs assembled from noise.
        require_in_season: Require the deviant run to lie inside a season.
        max_gap_within_run_days: Two points count as consecutive only if they
            are closer together than this; prevents a "run" straddling a gap.
        min_points_in_anomaly: Minimum number of measurements inside the
            anomaly window for the injection to count as a detection.
        min_points_to_fit: Below this many measurements in the analysis window
            the PSPL refit is not attempted at all and the injection is scored
            as undetected.  This happens when an event falls in a long season
            gap, where there is genuinely nothing to fit.
    """

    delta_chi2_min: float
    consecutive_points: int
    point_sigma: float
    require_same_sign: bool
    require_in_season: bool
    max_gap_within_run_days: float
    min_points_in_anomaly: int
    min_points_to_fit: int
    refit: RefitConfig

    @classmethod
    def from_reader(cls, r: _Reader) -> DetectionConfig:
        """Parse a ``detection`` section."""
        delta_chi2 = r.get_float("delta_chi2_min", 160.0, gt=0.0)
        consecutive = r.get_int("consecutive_points", 3, ge=1)
        sigma = r.get_float("point_sigma", 3.0, gt=0.0)
        same_sign = r.get_bool("require_same_sign", True)
        in_season = r.get_bool("require_in_season", True)
        max_gap = r.get_float("max_gap_within_run_days", 0.25, gt=0.0)
        min_in_anomaly = r.get_int("min_points_in_anomaly", 1, ge=0)
        min_to_fit = r.get_int("min_points_to_fit", 20, ge=6)
        refit_reader = r.get_section("refit", required=False)
        r.done()
        refit = (
            RefitConfig.from_reader(refit_reader)
            if refit_reader is not None
            else RefitConfig.from_reader(_Reader({}, f"{r.path}.refit"))
        )
        return cls(
            delta_chi2_min=delta_chi2,
            consecutive_points=consecutive,
            point_sigma=sigma,
            require_same_sign=same_sign,
            require_in_season=in_season,
            max_gap_within_run_days=max_gap,
            min_points_in_anomaly=min_in_anomaly,
            min_points_to_fit=min_to_fit,
            refit=refit,
        )


# --------------------------------------------------------------------------
# compute / output
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Parallelism and checkpointing.

    Attributes:
        n_workers: Worker processes; ``0`` means "one per CPU".
        start_method: Multiprocessing start method.  ``auto`` prefers ``fork``
            where the platform provides it, because ``spawn`` re-imports the
            parent's ``__main__`` module in every worker, which is both slow
            and impossible when the entry point is not an importable file.
        chunk_size: Injections dispatched to a worker at a time.
        checkpoint_every: Injections between checkpoint flushes.
        resume: Whether an interrupted run may resume from checkpoints.
        max_records_in_memory: Cap on buffered per-injection records, which
            bounds memory for grids of order 1e6 injections.
    """

    n_workers: int
    start_method: str
    chunk_size: int
    checkpoint_every: int
    resume: bool
    max_records_in_memory: int

    @classmethod
    def from_reader(cls, r: _Reader) -> ComputeConfig:
        """Parse a ``compute`` section."""
        cfg = cls(
            n_workers=r.get_int("n_workers", 0, ge=0),
            start_method=r.get_str(
                "start_method", "auto", choices=("auto", "fork", "spawn", "forkserver")
            ),
            chunk_size=r.get_int("chunk_size", 64, ge=1),
            checkpoint_every=r.get_int("checkpoint_every", 5000, ge=1),
            resume=r.get_bool("resume", True),
            max_records_in_memory=r.get_int("max_records_in_memory", 200_000, ge=1),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """What gets written to disk.

    Attributes:
        compression: Parquet compression codec.
        write_injections: Write the per-injection table (one row per trial).
        write_efficiency: Write the aggregated efficiency surface.
        write_plots: Render contour maps and diagnostics.
        confidence_level: Coverage of the Wilson-score interval per cell.
        colormap: Matplotlib colormap for the efficiency surface.  The default
            is perceptually uniform and colour-vision-deficiency safe;
            rainbow-like maps invent structure that is not in the data and
            must not be used for a published surface.
    """

    compression: str
    write_injections: bool
    write_efficiency: bool
    write_plots: bool
    confidence_level: float
    colormap: str

    @classmethod
    def from_reader(cls, r: _Reader) -> OutputConfig:
        """Parse an ``output`` section."""
        cfg = cls(
            compression=r.get_str(
                "compression", "snappy", choices=("snappy", "zstd", "gzip", "none")
            ),
            write_injections=r.get_bool("write_injections", True),
            write_efficiency=r.get_bool("write_efficiency", True),
            write_plots=r.get_bool("write_plots", True),
            confidence_level=r.get_float("confidence_level", 0.6827, gt=0.0, lt=1.0),
            colormap=r.get_str("colormap", "viridis"),
        )
        r.done()
        return cfg


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """A fully resolved and validated ``lenseff`` configuration.

    Attributes:
        run: Run bookkeeping and master seed.
        survey: Cadence, seasons and photometry.
        events: Baseline event population.
        injection: Planet grid and magnification settings.
        detection: Detection criteria and refit controls.
        compute: Parallelism and checkpointing.
        output: Output products.
        source_path: Path the YAML was read from, if any.  Excluded from the
            config hash so that moving a file does not change the hash.
    """

    run: RunConfig
    survey: SurveyConfig
    events: EventsConfig
    injection: InjectionConfig
    detection: DetectionConfig
    compute: ComputeConfig
    output: OutputConfig
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source_path: Path | None = None) -> Config:
        """Validate a raw configuration mapping.

        Args:
            data: The raw YAML document.
            source_path: Path the document came from, for error messages.

        Returns:
            The validated configuration.

        Raises:
            ConfigError: If any section is missing, unknown or out of range.
        """
        from lenseff.presets import resolve_survey_preset

        root = _Reader(data, "")
        run_reader = root.require_section("run")
        survey_raw = root.get_raw("survey")
        events_reader = root.require_section("events")
        injection_reader = root.require_section("injection")
        detection_reader = root.get_section("detection", required=False)
        compute_reader = root.get_section("compute", required=False)
        output_reader = root.get_section("output", required=False)
        root.done()

        survey_reader = _Reader(resolve_survey_preset(survey_raw), "survey")
        return cls(
            run=RunConfig.from_reader(run_reader),
            survey=SurveyConfig.from_reader(survey_reader),
            events=EventsConfig.from_reader(events_reader),
            injection=InjectionConfig.from_reader(injection_reader),
            detection=DetectionConfig.from_reader(
                detection_reader if detection_reader is not None else _Reader({}, "detection")
            ),
            compute=ComputeConfig.from_reader(
                compute_reader if compute_reader is not None else _Reader({}, "compute")
            ),
            output=OutputConfig.from_reader(
                output_reader if output_reader is not None else _Reader({}, "output")
            ),
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate a YAML configuration file."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        if data is None:
            raise ConfigError(f"{path} is empty")
        return cls.from_dict(data, source_path=path)

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved configuration as plain JSON-compatible data.

        ``source_path`` is deliberately omitted: the record must depend on the
        content of the configuration, not on where the file lives.
        """
        return {
            name: _to_jsonable(getattr(self, name))
            for name in ("run", "survey", "events", "injection", "detection", "compute", "output")
        }

    def hashable_dict(self) -> dict[str, Any]:
        """Return the part of the configuration that can change the numbers.

        ``compute`` is excluded.  Worker count, chunk size, checkpoint
        interval and start method are scheduling choices; by construction they
        cannot move a single value in the output, because every random draw is
        addressed rather than sequential.  Excluding them means the config
        hash identifies the *science* of a run, so a checkpointed sweep can be
        resumed on a different machine with a different core count -- which is
        exactly when resuming matters.  The full configuration, ``compute``
        included, is still written into the provenance record.
        """
        return {name: value for name, value in self.to_dict().items() if name != "compute"}

    def canonical_json(self) -> str:
        """Return a canonical, key-sorted JSON serialisation of the config."""
        return json.dumps(
            self.hashable_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def config_hash(self) -> str:
        """Return the SHA-256 hash of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def short_hash(self) -> str:
        """Return the first 12 characters of :meth:`config_hash`."""
        return self.config_hash()[:12]

    def n_injections(self) -> int:
        """Return the total number of planet injections implied by the grid.

        Returns:
            ``n_cells * events_per_cell * n_alpha``, plus the ``q = 0``
            control trials.
        """
        grid = self.injection.grid
        per_cell = grid.events_per_cell or self.events.n_events
        total = grid.n_cells * per_cell * grid.n_alpha
        if self.injection.include_zero_q_control:
            total += self.injection.n_zero_q_trials
        return total


def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, paths and arrays to JSON-safe data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    if isinstance(value, Sequence):
        return [_to_jsonable(v) for v in value]
    raise TypeError(f"cannot serialise {type(value).__name__} into the config record")


def load_config(path: str | Path) -> Config:
    """Load, preset-resolve and validate a YAML configuration file."""
    return Config.from_yaml(path)
