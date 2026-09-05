"""Aggregation into an efficiency surface, and its binomial uncertainties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from lenseff.config import Config
from lenseff.efficiency import (
    aggregate,
    control_summary,
    load_injections,
    marginal,
    wilson_interval,
    write_efficiency,
)


@pytest.fixture
def config(demo_config_path) -> Config:
    return Config.from_yaml(demo_config_path)


def fake_records(rng, n_per_cell: int = 40) -> pd.DataFrame:
    """A synthetic injection table with a known efficiency law."""
    rows = []
    for i_q, log_q in enumerate(np.linspace(-5.0, -2.0, 4)):
        for i_s, log_s in enumerate(np.linspace(-0.6, 0.6, 3)):
            probability = float(np.clip(0.25 * (log_q + 5.0) - 0.4 * abs(log_s), 0.0, 1.0))
            for k in range(n_per_cell):
                rows.append(
                    {
                        "kind": "planet",
                        "i_q": i_q,
                        "i_s": i_s,
                        "log_q": float(log_q),
                        "log_s": float(log_s),
                        "event_index": k % 4,
                        "alpha_index": k // 4,
                        "detected": bool(rng.random() < probability),
                        "delta_chi2": float(rng.normal(200.0, 50.0)),
                        "chi2_binary": float(rng.normal(8000.0, 100.0)),
                        "run_length": int(rng.integers(0, 10)),
                        "n_points_in_anomaly": int(rng.integers(0, 50)),
                        "anomaly_duration_days": float(rng.random()),
                        "criterion_consecutive_points": bool(rng.random() < 0.5),
                    }
                )
    return pd.DataFrame(rows)


# --- the Wilson interval ---------------------------------------------------


@pytest.mark.parametrize(("k", "n"), [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (37, 250)])
def test_wilson_bounds_satisfy_their_defining_equation(k, n):
    """Each bound solves |p_hat - p| = z sqrt(p(1-p)/n), which defines the interval."""
    confidence = 0.95
    z = float(norm.ppf(0.5 * (1.0 + confidence)))
    lower, upper = wilson_interval(k, n, confidence)
    p_hat = k / n
    for bound in (float(lower), float(upper)):
        if 0.0 < bound < 1.0:
            assert abs(p_hat - bound) == pytest.approx(
                z * np.sqrt(bound * (1.0 - bound) / n), rel=1e-9
            )


def test_wilson_interval_stays_inside_the_unit_range():
    lower, upper = wilson_interval(np.array([0, 10]), np.array([10, 10]), 0.95)
    assert lower[0] == 0.0
    assert 0.0 < upper[0] < 1.0, "zero successes must still give a non-degenerate upper bound"
    assert upper[1] == 1.0
    assert 0.0 < lower[1] < 1.0


def test_wilson_interval_with_no_trials_says_nothing():
    lower, upper = wilson_interval(0, 0, 0.95)
    assert (float(lower), float(upper)) == (0.0, 1.0)


def test_wilson_interval_narrows_with_sample_size():
    widths = []
    for n in (10, 100, 1000, 10000):
        lower, upper = wilson_interval(n // 2, n, 0.6827)
        widths.append(float(upper - lower))
    assert widths == sorted(widths, reverse=True)
    # width scales as 1/sqrt(n)
    assert widths[0] / widths[-1] == pytest.approx(np.sqrt(1000.0), rel=0.15)


@pytest.mark.parametrize("p", [0.05, 0.5, 0.9])
def test_wilson_interval_has_approximately_nominal_coverage(p):
    """The point of using Wilson rather than a Gaussian error bar."""
    rng = np.random.default_rng(20260905)
    n, trials = 30, 4000
    k = rng.binomial(n, p, size=trials)
    lower, upper = wilson_interval(k, np.full(trials, n), 0.6827)
    coverage = float(np.mean((lower <= p) & (p <= upper)))
    assert 0.60 < coverage < 0.80


# --- aggregation -----------------------------------------------------------


def test_aggregate_counts_and_efficiencies(config):
    rng = np.random.default_rng(1)
    records = fake_records(rng)
    surface = aggregate(records, config)
    assert len(surface) == 12
    assert (surface["n_trials"] == 40).all()
    np.testing.assert_allclose(surface["efficiency"], surface["n_detected"] / surface["n_trials"])
    assert (surface["efficiency_low"] <= surface["efficiency"]).all()
    assert (surface["efficiency"] <= surface["efficiency_high"]).all()
    assert list(surface.columns).count("q") == 1
    np.testing.assert_allclose(surface["q"], 10.0 ** surface["log_q"])


def test_aggregate_recovers_the_underlying_law(config):
    """With 400 trials per cell the measured surface must track the truth."""
    rng = np.random.default_rng(2)
    surface = aggregate(fake_records(rng, n_per_cell=400), config)
    truth = np.clip(0.25 * (surface["log_q"] + 5.0) - 0.4 * surface["log_s"].abs(), 0.0, 1.0)
    assert np.abs(surface["efficiency"] - truth).max() < 0.08


def test_aggregate_ignores_controls(config):
    rng = np.random.default_rng(3)
    records = fake_records(rng)
    controls = records.head(5).copy()
    controls["kind"] = "control"
    surface = aggregate(pd.concat([records, controls]), config)
    assert (surface["n_trials"] == 40).all()


def test_aggregate_on_an_empty_table(config):
    empty = aggregate(pd.DataFrame({"kind": []}), config)
    assert empty.empty


def test_marginal_collapses_one_axis(config):
    rng = np.random.default_rng(4)
    records = fake_records(rng)
    by_q = marginal(records, config, "log_q")
    assert len(by_q) == 4
    assert (by_q["n_trials"] == 120).all()
    assert by_q["efficiency"].is_monotonic_increasing
    assert len(marginal(records, config, "log_s")) == 3
    with pytest.raises(ValueError, match="must be 'log_q' or 'log_s'"):
        marginal(records, config, "log_rho")


def test_control_summary(config):
    rng = np.random.default_rng(5)
    records = fake_records(rng, n_per_cell=4)
    controls = records.head(20).copy()
    controls["kind"] = "control"
    controls["detected"] = False
    controls["delta_chi2"] = -np.abs(rng.normal(2.0, 1.0, size=20))
    summary = control_summary(pd.concat([records, controls]), config)
    assert summary["n_controls"] == 20
    assert summary["n_false_positives"] == 0
    assert summary["false_positive_rate"] == 0.0
    assert summary["false_positive_high"] > 0.0
    assert summary["max_delta_chi2"] < 0.0


def test_control_summary_with_no_controls(config):
    rng = np.random.default_rng(6)
    assert control_summary(fake_records(rng, n_per_cell=2), config) == {"n_controls": 0}


# --- input and output ------------------------------------------------------


def test_efficiency_surface_round_trips_with_provenance(config, tmp_path, demo_dict):
    from lenseff.provenance import read_provenance

    demo_dict["run"]["output_dir"] = str(tmp_path)
    config = Config.from_dict(demo_dict)
    rng = np.random.default_rng(7)
    surface = aggregate(fake_records(rng), config)
    path = write_efficiency(surface, config)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(restored, surface)
    assert read_provenance(path)["config_hash"] == config.config_hash()


def test_load_injections_prefers_the_combined_table(tmp_path):
    rng = np.random.default_rng(8)
    records = fake_records(rng, n_per_cell=2)
    (tmp_path / "injections").mkdir()
    records.head(4).to_parquet(tmp_path / "injections" / "shard_000000.parquet", index=False)
    records.to_parquet(tmp_path / "injections.parquet", index=False)
    assert len(load_injections(tmp_path)) == len(records)


def test_load_injections_falls_back_to_shards(tmp_path):
    rng = np.random.default_rng(9)
    records = fake_records(rng, n_per_cell=2)
    (tmp_path / "injections").mkdir()
    records.iloc[:10].to_parquet(tmp_path / "injections" / "shard_000000.parquet", index=False)
    records.iloc[10:].to_parquet(tmp_path / "injections" / "shard_000001.parquet", index=False)
    assert len(load_injections(tmp_path)) == len(records)


def test_load_injections_without_any_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="no injection records"):
        load_injections(tmp_path)
