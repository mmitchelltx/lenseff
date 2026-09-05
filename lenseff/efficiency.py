"""Aggregation of injection records into an efficiency surface.

An injection table is a list of Bernoulli trials, grouped by grid cell.  The
efficiency of a cell is the detected fraction, and its uncertainty is a
*binomial* interval -- not a Gaussian one.  That distinction matters at the
edges of the surface, where efficiency approaches 0 or 1 and a naive
``sqrt(p(1-p)/n)`` error bar extends outside ``[0, 1]`` or collapses to zero
width.  The Wilson score interval never does either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from lenseff.config import Config
from lenseff.provenance import collect_provenance, write_parquet

__all__ = [
    "aggregate",
    "control_summary",
    "load_injections",
    "marginal",
    "wilson_interval",
    "write_efficiency",
]


def wilson_interval(
    n_detected: np.ndarray | int, n_trials: np.ndarray | int, confidence: float = 0.6827
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Wilson score interval for a binomial proportion.

    The Wilson interval is the set of ``p`` for which the score test does not
    reject, which keeps it inside ``[0, 1]`` and keeps it non-degenerate when
    ``k = 0`` or ``k = n`` -- exactly the cases that dominate the interesting
    parts of an efficiency surface.

    Args:
        n_detected: Successes per cell.
        n_trials: Trials per cell.
        confidence: Coverage, e.g. ``0.6827`` for the 1-sigma equivalent.

    Returns:
        Lower and upper bounds.  Cells with no trials return ``(0, 1)``, the
        honest statement that nothing is known.
    """
    k = np.asarray(n_detected, dtype=float)
    n = np.asarray(n_trials, dtype=float)
    z = float(norm.ppf(0.5 * (1.0 + confidence)))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denominator = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denominator
        half = (z / denominator) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lower = np.clip(centre - half, 0.0, 1.0)
    upper = np.clip(centre + half, 0.0, 1.0)
    # k = 0 and k = n give bounds that are exactly 0 and 1 in exact arithmetic;
    # pin them so rounding cannot put the interval outside its own estimate
    lower = np.where(k <= 0.0, 0.0, lower)
    upper = np.where(k >= n, 1.0, upper)
    lower = np.where(n > 0, lower, 0.0)
    upper = np.where(n > 0, upper, 1.0)
    return lower, upper


def load_injections(output_dir: str | Path) -> pd.DataFrame:
    """Read an injection table, whether combined or left as shards.

    Args:
        output_dir: A run's output directory.

    Returns:
        Every injection record.

    Raises:
        FileNotFoundError: If neither a combined table nor shards are present.
    """
    output_dir = Path(output_dir)
    combined = output_dir / "injections.parquet"
    if combined.exists():
        return pd.read_parquet(combined)
    shards = output_dir / "injections"
    if shards.is_dir() and any(shards.glob("shard_*.parquet")):
        return pd.read_parquet(shards)
    raise FileNotFoundError(f"no injection records under {output_dir}")


def aggregate(records: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Collapse per-injection records into a per-cell efficiency surface.

    Args:
        records: The injection table, controls included.
        config: The run configuration, for the confidence level.

    Returns:
        One row per grid cell, with the detection fraction, its Wilson
        interval, and diagnostics that explain the value: how often the
        anomaly was observed at all, and the median statistic.
    """
    planets = records[records["kind"] == "planet"]
    if planets.empty:
        return pd.DataFrame(
            columns=["i_q", "i_s", "log_q", "log_s", "q", "s", "n_trials", "n_detected"]
        )
    grouped = planets.groupby(["i_q", "i_s", "log_q", "log_s"], sort=True)
    surface = grouped.agg(
        n_trials=("detected", "size"),
        n_detected=("detected", "sum"),
        median_delta_chi2=("delta_chi2", "median"),
        mean_run_length=("run_length", "mean"),
        frac_anomaly_observed=("n_points_in_anomaly", lambda column: float((column > 0).mean())),
        median_anomaly_days=("anomaly_duration_days", "median"),
    ).reset_index()
    surface["q"] = 10.0 ** surface["log_q"]
    surface["s"] = 10.0 ** surface["log_s"]
    surface["efficiency"] = surface["n_detected"] / surface["n_trials"]
    lower, upper = wilson_interval(
        surface["n_detected"].to_numpy(),
        surface["n_trials"].to_numpy(),
        config.output.confidence_level,
    )
    surface["efficiency_low"] = lower
    surface["efficiency_high"] = upper
    return surface.sort_values(["i_q", "i_s"]).reset_index(drop=True)


def marginal(records: pd.DataFrame, config: Config, axis: str) -> pd.DataFrame:
    """Collapse the surface along one axis.

    Args:
        records: The injection table.
        config: The run configuration.
        axis: ``"log_q"`` or ``"log_s"``.

    Returns:
        Efficiency and its Wilson interval as a function of ``axis``.

    Raises:
        ValueError: If ``axis`` is not a grid axis.
    """
    if axis not in {"log_q", "log_s"}:
        raise ValueError(f"axis must be 'log_q' or 'log_s', got {axis!r}")
    planets = records[records["kind"] == "planet"]
    grouped = (
        planets.groupby(axis, sort=True)
        .agg(n_trials=("detected", "size"), n_detected=("detected", "sum"))
        .reset_index()
    )
    grouped["efficiency"] = grouped["n_detected"] / grouped["n_trials"]
    lower, upper = wilson_interval(
        grouped["n_detected"].to_numpy(),
        grouped["n_trials"].to_numpy(),
        config.output.confidence_level,
    )
    grouped["efficiency_low"] = lower
    grouped["efficiency_high"] = upper
    return grouped


def control_summary(records: pd.DataFrame, config: Config) -> dict[str, float]:
    """Summarise the planet-free control injections.

    The Delta chi-square criterion cannot fire on a control: the refit family
    contains the injected model, so the statistic is bounded above by zero.
    What *can* fire is the consecutive-points criterion, on noise alone, and
    that is what this measures.

    Args:
        records: The injection table.
        config: The run configuration.

    Returns:
        Counts, the measured false-positive rate and its Wilson interval, and
        the largest Delta chi-square seen.  That maximum must not be positive.
        It is also reported relative to chi-square itself, because a refit
        that lands exactly on the truth returns a difference of order 1e-16
        times a chi-square of order 1e4 -- float64 rounding, not a signal.
    """
    controls = records[records["kind"] == "control"]
    n = len(controls)
    if n == 0:
        return {"n_controls": 0}
    k = int(controls["detected"].sum())
    lower, upper = wilson_interval(k, n, config.output.confidence_level)
    return {
        "n_controls": n,
        "n_false_positives": k,
        "false_positive_rate": k / n,
        "false_positive_low": float(lower),
        "false_positive_high": float(upper),
        "max_delta_chi2": float(controls["delta_chi2"].max()),
        "max_delta_chi2_relative": float(
            (controls["delta_chi2"] / controls["chi2_binary"].clip(lower=1.0)).max()
        ),
        "frac_run_criterion_passed": float(controls["criterion_consecutive_points"].mean()),
    }


def write_efficiency(surface: pd.DataFrame, config: Config) -> Path:
    """Write the efficiency surface to parquet with run provenance."""
    return write_parquet(
        surface,
        Path(config.run.output_dir) / "efficiency.parquet",
        collect_provenance(config, extra={"n_cells": len(surface)}),
        compression=config.output.compression,
    )
