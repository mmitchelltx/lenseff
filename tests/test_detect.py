"""The PSPL refit, the detection statistic and the criteria."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from lenseff.config import Config
from lenseff.detect import (
    detect,
    longest_significant_run,
    refit_pspl,
    solve_fluxes,
)
from lenseff.events import Event
from lenseff.inject import EventSetup, Planet, inject_planet
from lenseff.survey import Survey


@pytest.fixture
def config(demo_config_path) -> Config:
    return Config.from_yaml(demo_config_path)


@pytest.fixture
def survey(config) -> Survey:
    return Survey.from_config(config.survey, config.run.seed)


@pytest.fixture
def event(survey) -> Event:
    return Event(
        index=0,
        t_0=float(survey.times[survey.season_index == 2][4320]),
        u_0=0.3,
        t_E=25.0,
        source_mag=21.0,
        blend_ratio=0.2,
        rho=1.0e-3,
    )


@pytest.fixture
def setup(event, survey, config) -> EventSetup:
    return EventSetup.build(event, survey, config)


# --- the linear flux solution ----------------------------------------------


def test_flux_solution_matches_a_general_least_squares_solver():
    rng = np.random.default_rng(0)
    magnification = 1.0 + 9.0 * rng.random(500)
    err = 0.05 + 0.1 * rng.random(500)
    flux = 3.0 * magnification + 1.5 + rng.normal(0.0, err)
    design = np.column_stack([magnification / err, np.ones(500) / err])
    expected, *_ = np.linalg.lstsq(design, flux / err, rcond=None)
    got = solve_fluxes(magnification, flux, err)
    assert got.f_source == pytest.approx(expected[0], rel=1e-10)
    assert got.f_blend == pytest.approx(expected[1], rel=1e-10)
    residual = (flux - expected[0] * magnification - expected[1]) / err
    assert got.chi2 == pytest.approx(float(residual @ residual), rel=1e-12)


def test_flux_solution_is_exact_for_noiseless_data():
    magnification = np.linspace(1.0, 12.0, 200)
    flux = 7.25 * magnification - 0.75
    err = np.full(200, 0.01)
    got = solve_fluxes(magnification, flux, err)
    assert got.f_source == pytest.approx(7.25, rel=1e-10)
    assert got.f_blend == pytest.approx(-0.75, abs=1e-8)
    assert got.chi2 == pytest.approx(0.0, abs=1e-12)


def test_unblended_solution_holds_the_blend_at_zero():
    magnification = np.linspace(1.0, 12.0, 200)
    err = np.full(200, 0.01)
    flux = 4.0 * magnification
    got = solve_fluxes(magnification, flux, err, free_blending=False)
    assert got.f_blend == 0.0
    assert got.f_source == pytest.approx(4.0, rel=1e-10)
    assert got.chi2 == pytest.approx(0.0, abs=1e-12)


def test_negative_source_flux_is_clamped():
    magnification = np.linspace(1.0, 12.0, 200)
    err = np.full(200, 0.1)
    flux = -2.0 * magnification + 40.0
    clamped = solve_fluxes(magnification, flux, err, require_positive_source=True)
    free = solve_fluxes(magnification, flux, err, require_positive_source=False)
    assert clamped.f_source == 0.0
    assert clamped.f_blend == pytest.approx(float(np.mean(flux)), rel=1e-10)
    assert free.f_source < 0.0
    assert free.chi2 < clamped.chi2  # the constraint can only cost chi-square


# --- the consecutive-run statistic -----------------------------------------


def test_run_finder_locates_the_longest_run():
    times = np.arange(10, dtype=float)
    residual = np.array([0.0, 4.0, 4.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 4.0])
    length, first, last = longest_significant_run(
        residual, times, sigma=3.0, same_sign=True, max_gap_days=1.5
    )
    assert (length, first, last) == (4, 4, 7)


def test_run_finder_requires_a_consistent_sign():
    times = np.arange(5, dtype=float)
    residual = np.array([4.0, -4.0, 4.0, -4.0, 4.0])
    assert (
        longest_significant_run(residual, times, sigma=3.0, same_sign=True, max_gap_days=1.5)[0]
        == 1
    )
    assert (
        longest_significant_run(residual, times, sigma=3.0, same_sign=False, max_gap_days=1.5)[0]
        == 5
    )


def test_run_finder_never_bridges_a_gap():
    """Three significant points straddling a season boundary are not a run."""
    times = np.array([0.0, 0.01, 50.0, 50.01, 50.02])
    residual = np.full(5, 4.0)
    assert longest_significant_run(
        residual, times, sigma=3.0, same_sign=True, max_gap_days=0.25
    ) == (3, 2, 4)


def test_run_finder_with_nothing_significant():
    times = np.arange(5, dtype=float)
    assert longest_significant_run(
        np.zeros(5), times, sigma=3.0, same_sign=True, max_gap_days=1.0
    ) == (0, -1, -1)


def test_run_finder_threshold_is_strict():
    times = np.arange(3, dtype=float)
    residual = np.full(3, 3.0)
    assert (
        longest_significant_run(residual, times, sigma=3.0, same_sign=True, max_gap_days=1.0)[0]
        == 0
    )


# --- the refit -------------------------------------------------------------


def test_refit_recovers_the_truth_on_a_planet_free_light_curve(setup, survey, config, event):
    injection = inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config)
    fit = refit_pspl(injection, config)
    assert fit.t_0 == pytest.approx(event.t_0, abs=0.02)
    assert fit.u_0 == pytest.approx(event.u_0, rel=0.02)
    assert fit.t_E == pytest.approx(event.t_E, rel=0.02)
    assert fit.f_source == pytest.approx(injection.light_curve.f_source, rel=0.03)
    assert fit.chi2 / injection.light_curve.n_points == pytest.approx(1.0, abs=0.1)


def test_control_delta_chi2_is_never_positive(setup, survey, config):
    """The refit family contains the injected model, so it must fit at least as well.

    This exact inequality is what replaces a vague "small false-positive rate"
    claim for the Delta chi-square criterion.
    """
    result = detect(inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config), survey, config)
    # zero up to float64 rounding of a chi-square of order 1e4
    assert result.delta_chi2 <= 1e-9 * result.chi2_binary
    # three extra free parameters buy a few units of chi-square, no more
    assert result.delta_chi2 > -30.0
    assert not result.detected


def test_the_refit_reabsorbs_part_of_the_anomaly(setup, survey, config):
    """The headline methodological point, asserted numerically.

    Comparing the binary model against the *injected* PSPL parameters rather
    than a free refit inflates Delta chi-square, and it inflates it most in
    the marginal regime that decides the efficiency contour.
    """
    inflation = {}
    for q, s in ((1e-2, 1.3), (1e-3, 1.3), (1e-3, 0.6), (1e-4, 1.0)):
        injection = inject_planet(setup, Planet(q, s, 140.0), survey, config)
        result = detect(injection, survey, config)
        lc = injection.light_curve
        naive = solve_fluxes(injection.pspl_magnification, lc.flux, lc.flux_err).chi2
        naive_delta = naive - result.chi2_binary
        assert result.delta_chi2 <= naive_delta + 1e-6, "a refit can only fit better"
        inflation[(q, s)] = naive_delta / result.delta_chi2
    # a strong, well-sampled anomaly is barely reabsorbed
    assert inflation[(1e-2, 1.3)] < 1.15
    # a marginal one is reabsorbed heavily: using the truth would overstate it
    assert inflation[(1e-3, 0.6)] > 2.0


def test_multi_start_beats_a_single_start_from_a_bad_guess(demo_dict, setup, survey):
    """Screening many starts must not lose to refining only one."""
    injection_config = Config.from_dict(demo_dict)
    injection = inject_planet(setup, Planet(1e-2, 1.3, 140.0), survey, injection_config)
    lean = copy.deepcopy(demo_dict)
    lean["detection"]["refit"].update({"n_starts": 1, "n_refine": 1, "mask_anomaly_start": False})
    rich = copy.deepcopy(demo_dict)
    rich["detection"]["refit"].update({"n_starts": 28, "n_refine": 6, "mask_anomaly_start": True})
    lean_fit = refit_pspl(injection, Config.from_dict(lean))
    rich_fit = refit_pspl(injection, Config.from_dict(rich))
    assert rich_fit.chi2 <= lean_fit.chi2 + 1e-6
    assert rich_fit.n_starts > lean_fit.n_starts


def test_anomaly_masked_start_is_used_when_available(setup, survey, demo_dict):
    with_mask = copy.deepcopy(demo_dict)
    with_mask["detection"]["refit"]["mask_anomaly_start"] = True
    without = copy.deepcopy(demo_dict)
    without["detection"]["refit"]["mask_anomaly_start"] = False
    injection = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, Config.from_dict(with_mask))
    assert (
        refit_pspl(injection, Config.from_dict(with_mask)).n_starts
        == refit_pspl(injection, Config.from_dict(without)).n_starts + 1
    )


def test_refit_is_reproducible(setup, survey, config):
    injection = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    a = refit_pspl(injection, config)
    b = refit_pspl(injection, config)
    assert (a.t_0, a.u_0, a.t_E, a.chi2) == (b.t_0, b.u_0, b.t_E, b.chi2)


@pytest.mark.parametrize("method", ["least_squares", "Nelder-Mead", "Powell"])
def test_optimisers_agree_on_the_minimum(demo_dict, setup, survey, method):
    data = copy.deepcopy(demo_dict)
    data["detection"]["refit"]["method"] = method
    config = Config.from_dict(data)
    injection = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    fit = refit_pspl(injection, config)
    reference = refit_pspl(injection, Config.from_dict(demo_dict))
    assert fit.chi2 == pytest.approx(reference.chi2, rel=1e-3)


# --- criteria wiring -------------------------------------------------------


def test_every_criterion_is_recorded_and_all_must_pass(setup, survey, config):
    result = detect(inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config), survey, config)
    assert set(result.criteria) == {
        "fittable",
        "delta_chi2",
        "consecutive_points",
        "points_in_anomaly",
        "in_season",
    }
    assert result.detected == all(result.criteria.values())
    assert result.detected


def test_raising_the_threshold_can_only_remove_detections(demo_dict, setup, survey):
    injection_config = Config.from_dict(demo_dict)
    injection = inject_planet(setup, Planet(1e-4, 1.3, 140.0), survey, injection_config)
    strict = copy.deepcopy(demo_dict)
    strict["detection"]["delta_chi2_min"] = 1e12
    assert detect(injection, survey, injection_config).detected
    assert not detect(injection, survey, Config.from_dict(strict)).detected


def test_consecutive_points_criterion_rejects_a_single_spike(demo_dict, setup, survey):
    """An unresolved caustic crossing gives a huge Delta chi-square on one point."""
    config = Config.from_dict(demo_dict)
    injection = inject_planet(setup, Planet(1e-5, 1.3, 140.0), survey, config)
    result = detect(injection, survey, config)
    assert result.delta_chi2 > config.detection.delta_chi2_min
    assert result.run_length < config.detection.consecutive_points
    assert result.criteria["delta_chi2"]
    assert not result.criteria["consecutive_points"]
    assert not result.detected
    # relaxing only that criterion recovers the detection
    relaxed = copy.deepcopy(demo_dict)
    relaxed["detection"]["consecutive_points"] = 1
    assert detect(injection, survey, Config.from_dict(relaxed)).detected


def test_in_season_criterion_can_be_disabled(demo_dict, setup, survey):
    data = copy.deepcopy(demo_dict)
    data["detection"]["require_in_season"] = False
    config = Config.from_dict(data)
    result = detect(inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config), survey, config)
    assert "in_season" not in result.criteria


def test_detected_runs_lie_inside_a_season(setup, survey, config):
    result = detect(inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config), survey, config)
    assert result.run_in_season
    assert survey.in_season(np.array([result.run_t_start, result.run_t_end])).all()


def test_windowed_statistic_carries_most_of_the_signal(setup, survey, config):
    result = detect(inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config), survey, config)
    assert result.delta_chi2_window > 0.9 * result.delta_chi2
    assert result.delta_chi2_window <= result.delta_chi2 * 1.05


def test_result_record_is_flat_and_serialisable(setup, survey, config):
    record = detect(
        inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config), survey, config
    ).as_record()
    assert record["detected"] is True
    assert isinstance(record["delta_chi2"], float)
    assert all(isinstance(v, float | int | bool | np.bool_) for v in record.values())
    assert "criterion_delta_chi2" in record
