"""Event sampling, distribution samplers and light-curve simulation."""

from __future__ import annotations

import numpy as np
import pytest

from lenseff.config import Config, DistributionSpec
from lenseff.events import (
    Event,
    load_catalog,
    pspl_magnification,
    sample_distribution,
    sample_events,
    simulate_light_curve,
)
from lenseff.rng import generator
from lenseff.survey import Survey


@pytest.fixture
def config(demo_config_path) -> Config:
    return Config.from_yaml(demo_config_path)


@pytest.fixture
def survey(config) -> Survey:
    return Survey.from_config(config.survey, config.run.seed)


def analytic_pspl(u: np.ndarray) -> np.ndarray:
    """Point-source point-lens magnification, used to check MulensModel usage."""
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


# --- distribution samplers -------------------------------------------------


def test_uniform_sampler_moments():
    spec = DistributionSpec("uniform", {"min": -2.0, "max": 6.0})
    draws = sample_distribution(spec, generator(1, "events"), size=200_000)
    assert draws.min() >= -2.0
    assert draws.max() <= 6.0
    assert draws.mean() == pytest.approx(2.0, abs=0.02)
    assert draws.std() == pytest.approx(8.0 / np.sqrt(12.0), rel=0.01)


def test_loguniform_sampler_is_uniform_in_log():
    spec = DistributionSpec("loguniform", {"min": 1e-4, "max": 1e-1})
    draws = sample_distribution(spec, generator(2, "events"), size=200_000)
    logs = np.log10(draws)
    assert logs.min() >= -4.0
    assert logs.max() <= -1.0
    assert logs.mean() == pytest.approx(-2.5, abs=0.01)
    # equal counts per decade
    counts = np.histogram(logs, bins=3, range=(-4.0, -1.0))[0]
    assert counts.max() / counts.min() < 1.03


def test_lognormal_sampler_median_and_width():
    spec = DistributionSpec("lognormal", {"median": 25.0, "sigma_ln": 0.6})
    draws = sample_distribution(spec, generator(3, "events"), size=200_000)
    assert np.median(draws) == pytest.approx(25.0, rel=0.01)
    assert np.std(np.log(draws)) == pytest.approx(0.6, rel=0.01)


def test_normal_sampler_moments():
    spec = DistributionSpec("normal", {"mean": 3.0, "std": 0.5})
    draws = sample_distribution(spec, generator(4, "events"), size=100_000)
    assert draws.mean() == pytest.approx(3.0, abs=0.01)
    assert draws.std() == pytest.approx(0.5, rel=0.01)


@pytest.mark.parametrize("slope", [-2.0, -1.0, 0.0, 1.5])
def test_powerlaw_sampler_reproduces_its_slope(slope):
    lo, hi = 1.0, 100.0
    spec = DistributionSpec("powerlaw", {"slope": slope, "min": lo, "max": hi})
    draws = sample_distribution(spec, generator(5, "events"), size=400_000)
    assert draws.min() >= lo
    assert draws.max() <= hi
    # compare the empirical CDF at the geometric midpoint with the analytic one
    x = np.sqrt(lo * hi)
    if np.isclose(slope, -1.0):
        expected = np.log(x / lo) / np.log(hi / lo)
    else:
        power = slope + 1.0
        expected = (x**power - lo**power) / (hi**power - lo**power)
    assert (draws <= x).mean() == pytest.approx(expected, abs=0.005)


def test_fixed_sampler():
    draws = sample_distribution(
        DistributionSpec("fixed", {"value": 7.5}), generator(6, "events"), 10
    )
    np.testing.assert_array_equal(draws, np.full(10, 7.5))


def test_truncation_bounds_are_respected():
    spec = DistributionSpec("lognormal", {"median": 25.0, "sigma_ln": 0.9}, lower=10.0, upper=40.0)
    draws = sample_distribution(spec, generator(7, "events"), size=50_000)
    assert draws.min() >= 10.0
    assert draws.max() <= 40.0
    # the shape is the *truncated* lognormal, not the full one clipped to the
    # edges: compare against the analytic median of the truncated distribution
    from scipy.stats import norm

    a, b = np.log(10.0 / 25.0) / 0.9, np.log(40.0 / 25.0) / 0.9
    z = norm.ppf(0.5 * (norm.cdf(a) + norm.cdf(b)))
    assert np.median(draws) == pytest.approx(25.0 * np.exp(0.9 * z), rel=0.01)


def test_impossible_truncation_raises():
    spec = DistributionSpec("normal", {"mean": 0.0, "std": 1.0}, lower=100.0, upper=101.0)
    with pytest.raises(RuntimeError, match="reject almost every draw"):
        sample_distribution(spec, generator(8, "events"), size=5)


def test_scalar_and_array_shapes():
    spec = DistributionSpec("uniform", {"min": 0.0, "max": 1.0})
    assert isinstance(sample_distribution(spec, generator(9, "events")), float)
    assert sample_distribution(spec, generator(9, "events"), size=3).shape == (3,)


# --- event sampling --------------------------------------------------------


def test_sample_events_returns_the_requested_number(config, survey):
    events = sample_events(config, survey)
    assert len(events) == config.events.n_events
    assert [e.index for e in events] == list(range(len(events)))


def test_sampled_events_satisfy_every_cut(config, survey):
    for event in sample_events(config, survey):
        assert survey.in_season(event.t_0)[0]
        baseline = event.source_mag - 2.5 * np.log10(1.0 + event.blend_ratio)
        assert survey.photometry.saturation_mag <= baseline <= survey.photometry.faint_limit_mag
        half = config.events.peak_window_t_E * event.t_E
        assert survey.points_in_window(event.t_0, half) >= config.events.min_points_near_peak
        assert 0.0 <= event.u_0 <= 1.0
        assert 1.0 <= event.t_E <= 300.0


def test_event_sampling_is_reproducible(config, survey):
    assert sample_events(config, survey) == sample_events(config, survey)


def test_enlarging_the_sample_preserves_its_prefix(demo_dict):
    small = Config.from_dict(demo_dict)
    survey = Survey.from_config(small.survey, small.run.seed)
    demo_dict["events"]["n_events"] = 25
    large = Config.from_dict(demo_dict)
    assert sample_events(large, survey)[:8] == sample_events(small, survey)


def test_a_different_seed_gives_different_events(demo_dict, survey):
    a = sample_events(Config.from_dict(demo_dict), survey)
    demo_dict["run"]["seed"] = 999
    b = sample_events(Config.from_dict(demo_dict), survey)
    assert a[0] != b[0]


def test_impossible_cuts_raise(demo_dict, survey):
    demo_dict["events"]["min_points_near_peak"] = 10**9
    with pytest.raises(RuntimeError, match="passed the cuts"):
        sample_events(Config.from_dict(demo_dict), survey)


# --- magnification and light curves ----------------------------------------


def test_mulensmodel_pspl_matches_the_analytic_formula(config, survey):
    event = sample_events(config, survey)[0]
    times = event.t_0 + event.t_E * np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    u = np.sqrt(event.u_0**2 + ((times - event.t_0) / event.t_E) ** 2)
    got = pspl_magnification(event, times, finite_source=False)
    np.testing.assert_allclose(got, analytic_pspl(u), rtol=1e-10)


def test_peak_magnification_equals_the_u0_value(config, survey):
    event = sample_events(config, survey)[0]
    peak = pspl_magnification(event, np.array([event.t_0]), finite_source=False)[0]
    assert peak == pytest.approx(analytic_pspl(np.array([event.u_0]))[0], rel=1e-10)


def test_finite_source_only_matters_near_the_peak(config, survey):
    event = Event(0, 2461550.0, 1e-4, 20.0, 21.0, 0.2, 1e-3)
    times = event.t_0 + event.t_E * np.array([-1.0, 0.0, 1.0])
    point = pspl_magnification(event, times, finite_source=False)
    finite = pspl_magnification(event, times, finite_source=True)
    # u_0 << rho: the finite source strongly suppresses the peak
    assert finite[1] < 0.5 * point[1]
    np.testing.assert_allclose(finite[[0, 2]], point[[0, 2]], rtol=1e-9)


def test_light_curve_is_consistent_with_its_own_truth(config, survey):
    event = sample_events(config, survey)[0]
    lc = simulate_light_curve(event, survey, config)
    truth = lc.f_source * lc.magnification + lc.f_blend
    # chi2/dof of the generating model must be 1 to within its own sampling error
    reduced = lc.chi2(truth) / lc.n_points
    assert reduced == pytest.approx(1.0, abs=5.0 * np.sqrt(2.0 / lc.n_points))
    residual = (lc.flux - truth) / lc.flux_err
    assert residual.mean() == pytest.approx(0.0, abs=5.0 / np.sqrt(lc.n_points))


def test_noiseless_light_curve_is_exact(config, survey):
    event = sample_events(config, survey)[0]
    lc = simulate_light_curve(event, survey, config, add_noise=False)
    np.testing.assert_allclose(lc.flux, lc.f_source * lc.magnification + lc.f_blend, rtol=1e-14)
    assert lc.chi2(lc.flux) == 0.0


def test_blend_ratio_sets_the_baseline_flux(config, survey):
    event = sample_events(config, survey)[0]
    lc = simulate_light_curve(event, survey, config, add_noise=False)
    assert lc.f_blend == pytest.approx(event.blend_ratio * lc.f_source)
    baseline = survey.mag_from_flux(lc.f_source + lc.f_blend)
    assert baseline == pytest.approx(event.source_mag - 2.5 * np.log10(1 + event.blend_ratio))


def test_light_curve_noise_is_reproducible(config, survey):
    event = sample_events(config, survey)[0]
    a = simulate_light_curve(event, survey, config)
    b = simulate_light_curve(event, survey, config)
    assert a.flux.tobytes() == b.flux.tobytes()


def test_saturated_points_are_dropped(demo_dict):
    demo_dict["survey"]["photometry"] = {"saturation_mag": 21.5}
    config = Config.from_dict(demo_dict)
    bright_survey = Survey.from_config(config.survey, config.run.seed)
    event = Event(0, float(bright_survey.times[9000]), 0.01, 25.0, 22.0, 0.1, 1e-3)
    lc = simulate_light_curve(event, bright_survey, config, add_noise=False)
    assert lc.n_points < bright_survey.n_points
    assert not bright_survey.is_saturated(lc.flux).any()


def test_catalog_round_trip(config, survey, tmp_path, demo_dict):
    import pandas as pd

    events = sample_events(config, survey)
    frame = pd.DataFrame(
        {
            "t_0": [e.t_0 for e in events],
            "u_0": [e.u_0 for e in events],
            "t_E": [e.t_E for e in events],
            "source_mag": [e.source_mag for e in events],
            "blend_ratio": [e.blend_ratio for e in events],
            "rho": [e.rho for e in events],
        }
    )
    path = tmp_path / "catalog.parquet"
    frame.to_parquet(path, index=False)
    demo_dict["events"]["source"] = "catalog"
    demo_dict["events"]["catalog_path"] = str(path)
    loaded = load_catalog(Config.from_dict(demo_dict), survey)
    assert loaded == events


def test_catalog_missing_column(config, survey, tmp_path, demo_dict):
    import pandas as pd

    path = tmp_path / "bad.csv"
    pd.DataFrame({"t_0": [1.0]}).to_csv(path, index=False)
    demo_dict["events"]["source"] = "catalog"
    demo_dict["events"]["catalog_path"] = str(path)
    with pytest.raises(ValueError, match="missing column"):
        load_catalog(Config.from_dict(demo_dict), survey)
