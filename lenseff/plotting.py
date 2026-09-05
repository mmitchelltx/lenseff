"""Light-curve, efficiency and diagnostic figures.

Matplotlib is imported lazily so that importing :mod:`lenseff` in a worker
process does not pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from lenseff.survey import Survey

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from lenseff.config import Config
    from lenseff.events import LightCurve
    from lenseff.inject import Injection

__all__ = ["plot_efficiency_map", "plot_efficiency_slices", "plot_injection", "plot_light_curve"]


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


#: Contour levels drawn over the efficiency surface.  They are a second,
#: non-colour encoding of the same magnitude, which is what keeps the figure
#: readable in greyscale, in print, and to a colour-vision-deficient reader.
EFFICIENCY_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _surface_grid(surface: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a per-cell table into ``(log_s, log_q, efficiency)`` arrays."""
    log_q = np.sort(surface["log_q"].unique())
    log_s = np.sort(surface["log_s"].unique())
    grid = np.full((log_q.size, log_s.size), np.nan)
    q_index = {value: i for i, value in enumerate(log_q)}
    s_index = {value: i for i, value in enumerate(log_s)}
    for row in surface.itertuples(index=False):
        grid[q_index[row.log_q], s_index[row.log_s]] = row.efficiency
    return log_s, log_q, grid


def plot_efficiency_map(
    surface: pd.DataFrame,
    config: Config,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the detection-efficiency surface in the ``(log s, log q)`` plane.

    Magnitude is carried twice: by a perceptually uniform sequential colormap
    and by labelled contour lines, so the figure survives greyscale printing
    and colour-vision deficiency.

    Args:
        surface: Output of :func:`lenseff.efficiency.aggregate`.
        config: The run configuration, for the colormap and the run label.
        ax: Axis to draw on; a new figure is created when omitted.
        title: Optional title.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.4))
    figure = cast("Figure", ax.get_figure())
    log_s, log_q, grid = _surface_grid(surface)

    mesh = ax.pcolormesh(
        log_s,
        log_q,
        np.ma.masked_invalid(grid),
        cmap=config.output.colormap,
        vmin=0.0,
        vmax=1.0,
        shading="nearest",
        rasterized=True,
    )
    if np.isfinite(grid).sum() > 3 and log_s.size > 1 and log_q.size > 1:
        levels = [v for v in EFFICIENCY_LEVELS if np.nanmin(grid) < v < np.nanmax(grid)]
        if levels:
            contours = ax.contour(
                log_s, log_q, grid, levels=levels, colors="white", linewidths=0.9, alpha=0.85
            )
            ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    bar = figure.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("detection efficiency", fontsize="small")
    bar.outline.set_visible(False)

    ax.axvline(0.0, color="0.9", lw=0.8, ls=":", zorder=3)
    ax.set_xlabel(r"$\log_{10}\, s$  (projected separation / $\theta_\mathrm{E}$)")
    ax.set_ylabel(r"$\log_{10}\, q$  (planet / host mass ratio)")
    n_trials = int(surface["n_trials"].sum())
    ax.set_title(
        title
        if title is not None
        else f"{config.run.name} [{config.short_hash()}] - {n_trials:,} injections",
        fontsize="medium",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize="small", color="0.6")
    figure.tight_layout()
    return figure


def plot_efficiency_slices(
    surface: pd.DataFrame,
    config: Config,
    *,
    ax: Axes | None = None,
    n_slices: int = 4,
) -> Figure:
    """Plot efficiency against ``log q`` for a few separations, with intervals.

    Error bars are the Wilson score interval, which stays inside ``[0, 1]``
    and stays finite at ``0`` and ``1`` detections -- where a Gaussian error
    bar would be wrong or vanish.

    Args:
        surface: Output of :func:`lenseff.efficiency.aggregate`.
        config: The run configuration.
        ax: Axis to draw on; a new figure is created when omitted.
        n_slices: How many separations to draw.  Kept small on purpose: each
            series is direct-labelled as well as listed in the legend.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.6))
    figure = cast("Figure", ax.get_figure())

    separations = np.sort(surface["log_s"].unique())
    chosen = separations[
        np.unique(np.linspace(0, separations.size - 1, min(n_slices, separations.size)).astype(int))
    ]
    colors = plt.get_cmap(config.output.colormap)(np.linspace(0.15, 0.85, len(chosen)))

    for color, log_s in zip(colors, chosen, strict=True):
        slice_ = surface[surface["log_s"] == log_s].sort_values("log_q")
        label = f"$s$ = {10.0**log_s:.2f}"
        ax.errorbar(
            slice_["log_q"],
            slice_["efficiency"],
            yerr=[
                slice_["efficiency"] - slice_["efficiency_low"],
                slice_["efficiency_high"] - slice_["efficiency"],
            ],
            marker="o",
            markersize=4.0,
            lw=1.6,
            capsize=2.0,
            color=color,
            label=label,
        )
        last = slice_.iloc[-1]
        ax.annotate(
            label,
            (last["log_q"], last["efficiency"]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize="x-small",
            color="0.25",
            va="center",
        )

    ax.set_xlabel(r"$\log_{10}\, q$")
    ax.set_ylabel("detection efficiency")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize="x-small", frameon=False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize="small", color="0.6")
    figure.tight_layout()
    return figure
