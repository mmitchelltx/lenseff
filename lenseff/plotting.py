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
    from lenseff.inject import Injection

__all__ = ["plot_injection", "plot_light_curve"]


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


def plot_injection(
    injection: Injection,
    survey: Survey,
    *,
    axes: tuple[Axes, Axes] | None = None,
    pad_windows: float = 3.0,
    title: str | None = None,
) -> Figure:
    """Plot an injected anomaly and its residual against the baseline model.

    The upper panel shows the perturbed light curve with the baseline PSPL
    model overlaid; the lower panel shows the residual in units of the
    photometric uncertainty, which is the quantity the consecutive-points
    detection criterion acts on.  The shaded band marks the anomaly window.

    Args:
        injection: The injection to plot.
        survey: The survey it was generated on.
        axes: A ``(light curve, residual)`` pair of axes; created when omitted.
        pad_windows: Zoom to the anomaly window padded by this many window
            widths on each side.
        title: Optional title for the upper panel.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        figure, created = plt.subplots(
            2, 1, figsize=(9.0, 5.5), sharex=True, height_ratios=[2.4, 1.0]
        )
        top, bottom = created[0], created[1]
    else:
        top, bottom = axes
        figure = cast("Figure", top.get_figure())

    lc = injection.light_curve
    event = injection.event
    anomaly = injection.anomaly
    if anomaly.exists:
        pad = max(pad_windows * anomaly.duration_days, 0.05 * event.t_E)
        lo, hi = anomaly.t_start - pad, anomaly.t_end + pad
    else:
        lo, hi = event.t_0 - 0.5 * event.t_E, event.t_0 + 0.5 * event.t_E
    mask = (lc.times >= lo) & (lc.times <= hi)

    baseline_flux = lc.f_source * injection.pspl_magnification + lc.f_blend
    binary_flux = lc.f_source * injection.binary_magnification + lc.f_blend

    for ax in (top, bottom):
        _season_shading(ax, survey, lo, hi)
        if anomaly.exists:
            ax.axvspan(
                anomaly.t_start, anomaly.t_end, color="tab:orange", alpha=0.12, zorder=1, lw=0
            )

    top.errorbar(
        lc.times[mask],
        survey.mag_from_flux(lc.flux[mask]),
        yerr=2.5 / np.log(10.0) * lc.flux_err[mask] / lc.flux[mask],
        fmt=".",
        markersize=2.5,
        elinewidth=0.4,
        color="0.35",
        alpha=0.7,
        zorder=2,
        label="data",
    )
    top.plot(
        lc.times[mask],
        survey.mag_from_flux(baseline_flux[mask]),
        lw=1.3,
        color="tab:blue",
        label="PSPL (no planet)",
        zorder=3,
    )
    top.plot(
        lc.times[mask],
        survey.mag_from_flux(binary_flux[mask]),
        lw=1.3,
        color="tab:red",
        label="binary (injected)",
        zorder=4,
    )
    top.invert_yaxis()
    top.set_ylabel(f"{survey.photometry.band} mag")
    top.legend(loc="best", fontsize="x-small")
    planet = injection.planet
    top.set_title(
        title
        if title is not None
        else (
            f"$q$={planet.q:.0e}, $s$={planet.s:.2f}, "
            f"$\\alpha$={planet.alpha_deg:.0f}$^\\circ$, "
            f"$u_0$={event.u_0:.3f}, $t_E$={event.t_E:.0f} d"
        ),
        fontsize="medium",
    )

    residual = (lc.flux - baseline_flux) / lc.flux_err
    bottom.axhline(0.0, color="tab:blue", lw=1.0, zorder=3)
    bottom.plot(lc.times[mask], residual[mask], ".", markersize=2.0, color="0.35", zorder=2)
    bottom.plot(
        lc.times[mask],
        ((binary_flux - baseline_flux) / lc.flux_err)[mask],
        lw=1.2,
        color="tab:red",
        zorder=4,
    )
    bottom.set_xlabel("HJD")
    bottom.set_ylabel(r"residual [$\sigma$]")
    top.set_xlim(lo, hi)
    figure.tight_layout()
    return figure
