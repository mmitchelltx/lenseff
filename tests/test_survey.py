"""Observing calendar and photometric error model."""

from __future__ import annotations

import numpy as np
import pytest

from lenseff.config import Config
from lenseff.survey import MAG_PER_DEX, Survey


@pytest.fixture
def survey(demo_config_path) -> Survey:
    config = Config.from_yaml(demo_config_path)
    return Survey.from_config(config.survey, config.run.seed)


def test_calendar_size_matches_cadence_and_seasons(survey):
    per_season = int(np.floor(72.0 / (12.0 / 1440.0))) + 1
    assert per_season == 8641
    assert survey.n_points == 6 * per_season == 51846


def test_times_are_sorted_and_all_in_season(survey):
    assert np.all(np.diff(survey.times) > 0)
    assert survey.in_season(survey.times).all()
    assert (survey.season_index >= 0).all()
    assert set(np.unique(survey.season_index)) == set(range(6))


def test_cadence_is_regular_within_a_season(survey):
    first = survey.times[survey.season_index == 0]
    # 1e-9 d = 90 us; the residual is float64 resolution at HJD ~ 2.46e6, not a
    # modelling choice, and is six orders of magnitude below any real cadence.
    np.testing.assert_allclose(np.diff(first), 12.0 / 1440.0, atol=1e-9)


def test_season_lookup_of_arbitrary_times(survey):
    windows = survey.seasons.windows
    inside = np.array([w[0] + 10.0 for w in windows])
    np.testing.assert_array_equal(survey.season_of(inside), np.arange(6))
    gap = 0.5 * (windows[0][1] + windows[1][0])
    assert survey.season_of(gap)[0] == -1
    assert survey.season_of(windows[0][0] - 1.0)[0] == -1
    assert survey.season_of(windows[-1][1] + 1.0)[0] == -1
    # window edges are inclusive
    assert survey.season_of(windows[2][0])[0] == 2
    assert survey.season_of(windows[2][1])[0] == 2


def test_duty_cycle(survey):
    assert survey.duty_cycle == pytest.approx(6 * 72.0 / (1825.25 + 72.0))


def test_points_in_window(survey):
    centre = float(survey.times[survey.season_index == 1][4320])
    assert survey.points_in_window(centre, 1.0) == pytest.approx(2 * 120 + 1, abs=1)
    assert survey.points_in_window(centre, 0.0) == 1
    # a window entirely inside a season gap contains nothing
    gap = 0.5 * (survey.seasons.windows[0][1] + survey.seasons.windows[1][0])
    assert survey.points_in_window(gap, 1.0) == 0


def test_flux_magnitude_round_trip(survey):
    mags = np.array([15.0, 18.5, 20.0, 23.0, 25.5])
    np.testing.assert_allclose(survey.mag_from_flux(survey.flux_from_mag(mags)), mags, rtol=1e-12)
    assert survey.flux_from_mag(survey.photometry.zero_point) == pytest.approx(1.0)


def test_non_positive_flux_maps_to_infinite_magnitude(survey):
    assert np.isinf(survey.mag_from_flux(np.array([-1.0, 0.0]))).all()


def test_flux_uncertainty_matches_the_analytic_noise_model(survey):
    phot = survey.photometry
    flux = 500.0
    integration = phot.exposure_time_s * phot.n_exposures
    variance = (
        flux * integration
        + phot.sky_e_per_s_per_pixel * phot.n_pixels * integration
        + phot.n_pixels * phot.n_exposures * phot.read_noise_e**2
    )
    photon = np.sqrt(variance) / integration
    floor = (phot.systematic_floor_mmag / 1000.0) / MAG_PER_DEX * flux
    assert survey.flux_uncertainty(flux) == pytest.approx(np.hypot(photon, floor))


def test_precision_degrades_monotonically_with_magnitude(survey):
    mags = np.linspace(15.0, 25.0, 41)
    sigma = survey.magnitude_uncertainty(mags)
    assert np.all(np.diff(sigma) > 0)
    # the specific values are the Roman-preset predictions
    assert survey.magnitude_uncertainty(20.0) == pytest.approx(0.0049, abs=2e-4)
    assert survey.magnitude_uncertainty(23.0) == pytest.approx(0.0227, abs=1e-3)


def test_bright_stars_approach_the_systematic_floor(survey):
    floor = survey.photometry.systematic_floor_mmag / 1000.0
    # at W149 = 15 photon noise still contributes ~0.5 mmag in quadrature
    assert survey.magnitude_uncertainty(15.0) == pytest.approx(0.00111, rel=0.02)
    # in the limit of an arbitrarily bright star only the floor survives
    assert survey.magnitude_uncertainty(5.0) == pytest.approx(floor, rel=1e-4)
    assert survey.magnitude_uncertainty(5.0) > floor


def test_precision_scale_scales_every_uncertainty(demo_dict):
    demo_dict["survey"]["precision_scale"] = 0.5
    config = Config.from_dict(demo_dict)
    halved = Survey.from_config(config.survey, config.run.seed)
    demo_dict["survey"]["precision_scale"] = 1.0
    nominal = Survey.from_config(Config.from_dict(demo_dict).survey, config.run.seed)
    mags = np.array([18.0, 21.0, 24.0])
    np.testing.assert_allclose(
        halved.magnitude_uncertainty(mags), 0.5 * nominal.magnitude_uncertainty(mags), rtol=1e-12
    )


def test_saturation_and_faint_masks(survey):
    bright = survey.flux_from_mag(survey.photometry.saturation_mag - 0.5)
    faint = survey.flux_from_mag(survey.photometry.faint_limit_mag + 0.5)
    middle = survey.flux_from_mag(20.0)
    assert survey.is_saturated(bright)
    assert not survey.is_saturated(middle)
    assert survey.is_too_faint(faint)
    assert not survey.is_too_faint(middle)


def test_dropout_is_deterministic_and_statistically_correct(demo_dict):
    demo_dict["survey"]["dropout_fraction"] = 0.2
    config = Config.from_dict(demo_dict)
    a = Survey.from_config(config.survey, config.run.seed)
    b = Survey.from_config(config.survey, config.run.seed)
    np.testing.assert_array_equal(a.times, b.times)
    expected = 0.8 * 51846
    assert abs(a.n_points - expected) < 5.0 * np.sqrt(51846 * 0.2 * 0.8)
    c = Survey.from_config(config.survey, config.run.seed + 1)
    assert not np.array_equal(a.times, c.times)


def test_summary_reports_the_calendar(survey):
    summary = survey.summary()
    assert summary["preset"] == "roman_gbtds"
    assert summary["n_points"] == 51846
    assert summary["n_seasons"] == 6
