"""Light-curve, efficiency and diagnostic figures.

Matplotlib is imported lazily so that importing :mod:`lenseff` in a worker
process does not pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from lenseff.survey import Survey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from lenseff.events import LightCurve

__all__ = ["plot_light_curve"]


def _season_shading(ax: Axes, survey: Survey, t_min: float, t_max: float) -> None:
    """Shade the observing seasons behind a time-series axis."""
    for start, end in survey.seasons.windows:
        if end < t_min or start > t_max:
            continue
        ax.axvspan(max(start, t_min), min(end, t_max), color="0.92", zorder=0, linewidth=0)


def plot_light_curve(
    light_curve: LightCurve,
    survey: Survey,
    *,
    ax: Axes | None = None,
    window_t_E: float | None = 3.0,
    model_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a simulated light curve in magnitudes.

    Args:
        light_curve: The light curve to plot.
        survey: The survey it was generated on, for season shading and the
            flux-to-magnitude conversion.
        ax: Axis to draw on; a new figure is created when omitted.
        window_t_E: Zoom to this many Einstein times either side of the peak.
            ``None`` plots the whole calendar.
        model_curves: Optional ``{label: (times, flux)}`` overlays.
        title: Optional axis title.

    Returns:
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    event = light_curve.event
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.5))
    figure = cast("Figure", ax.get_figure())

    times = light_curve.times
    if window_t_E is not None:
        half = window_t_E * event.t_E
        mask = np.abs(times - event.t_0) <= half
    else:
        mask = np.ones(times.size, dtype=bool)

    mag = survey.mag_from_flux(light_curve.flux[mask])
    mag_err = 2.5 / np.log(10.0) * light_curve.flux_err[mask] / light_curve.flux[mask]
    finite = np.isfinite(mag) & np.isfinite(mag_err)

    t_min = float(times[mask].min()) if mask.any() else float(times.min())
    t_max = float(times[mask].max()) if mask.any() else float(times.max())
    _season_shading(ax, survey, t_min, t_max)

    ax.errorbar(
        times[mask][finite],
        mag[finite],
        yerr=mag_err[finite],
        fmt=".",
        markersize=2.0,
        elinewidth=0.4,
        color="0.35",
        alpha=0.6,
        label="data",
        zorder=2,
    )
    for label, (model_times, model_flux) in (model_curves or {}).items():
        ax.plot(model_times, survey.mag_from_flux(model_flux), lw=1.4, label=label, zorder=3)

    ax.invert_yaxis()
    ax.set_xlabel("HJD")
    ax.set_ylabel(f"{survey.photometry.band} magnitude")
    ax.set_title(
        title
        if title is not None
        else (
            f"event {event.index}: $u_0$={event.u_0:.3f}, "
            f"$t_E$={event.t_E:.1f} d, $m_s$={event.source_mag:.2f}"
        )
    )
    ax.legend(loc="best", fontsize="small")
    figure.tight_layout()
    return figure
