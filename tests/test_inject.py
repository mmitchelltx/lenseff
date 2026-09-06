"""Binary-lens injection: magnification, windows and noise reuse."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from lenseff.config import Config
from lenseff.events import Event, pspl_magnification, simulate_light_curve
from lenseff.inject import (
    EventSetup,
    Planet,
    analysis_half_width,
    binary_magnification,
    caustic_extent,
    inject_planet,
    locate_anomaly,
)
from lenseff.survey import Survey


@pytest.fixture
def config(demo_config_path) -> Config:
    return Config.from_yaml(demo_config_path)


@pytest.fixture
def survey(config) -> Survey:
    return Survey.from_config(config.survey, config.run.seed)


@pytest.fixture
def event(survey) -> Event:
    """A bright, well-covered event placed at the centre of season 2."""
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


def with_override(demo_dict, section, **values) -> Config:
    data = copy.deepcopy(demo_dict)
    data[section].update(values)
    return Config.from_dict(data)


# --- caustics and windows --------------------------------------------------


@pytest.mark.parametrize(("q", "s"), [(1e-3, 4.0), (1e-4, 0.3), (1e-5, 0.1), (1e-2, 10.0)])
def test_caustic_extent_tracks_the_planetary_caustic_position(q, s):
    """The planetary caustic sits near |s - 1/s| Einstein radii from the host."""
    _, extent = caustic_extent(Planet(q, s, 0.0), 500)
    assert extent == pytest.approx(abs(s - 1.0 / s), rel=0.05)


def test_caustics_shrink_with_mass_ratio():
    extents = [caustic_extent(Planet(q, 1.0, 0.0), 500)[1] for q in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert extents == sorted(extents)


def test_analysis_window_widens_for_distant_caustics(event, config):
    near = analysis_half_width(event, Planet(1e-3, 1.0, 0.0), config)
    wide = analysis_half_width(event, Planet(1e-3, 4.0, 0.0), config)
    close = analysis_half_width(event, Planet(1e-3, 0.25, 0.0), config)
    assert near == pytest.approx(3.0 * event.t_E)
    for far in (wide, close):
        assert far > near
        assert far == pytest.approx((abs(4.0 - 0.25) + 1.0) * event.t_E, rel=0.05)


def test_control_window_is_the_nominal_width(event, config):
    assert analysis_half_width(event, Planet(0.0, 1.0, 0.0), config) == pytest.approx(
        3.0 * event.t_E
    )


# --- magnification ---------------------------------------------------------


def test_binary_magnification_approaches_pspl_as_q_vanishes(event, config, survey):
    times = survey.times[np.abs(survey.times - event.t_0) < 2.0 * event.t_E]
    single = pspl_magnification(event, times, finite_source=True)
    previous = np.inf
    for q in (1e-4, 1e-5, 1e-6, 1e-7):
        binary, _ = binary_magnification(event, Planet(q, 1.3, 140.0), times, config)
        deviation = float(np.abs(binary - single).max() / single.max())
        assert deviation < previous
        previous = deviation
    # below ~1e-4 the residual is the binary root-solver's own numerical floor,
    # not a physical perturbation
    assert previous < 1e-3


def test_control_injection_is_exactly_the_planet_free_model(setup, survey, config, event):
    injection = inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config)
    single = pspl_magnification(event, injection.light_curve.times, finite_source=True)
    np.testing.assert_allclose(injection.binary_magnification, single, rtol=1e-14)
    np.testing.assert_array_equal(injection.binary_magnification, injection.pspl_magnification)
    assert not injection.anomaly.exists
    assert injection.points_in_anomaly() == 0


def test_control_matches_the_unperturbed_light_curve(setup, survey, config, event):
    """The q = 0 control must reproduce simulate_light_curve, point for point."""
    injection = inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config)
    plain = simulate_light_curve(event, survey, config)
    times = injection.light_curve.times
    take = np.isin(plain.times, times)
    np.testing.assert_allclose(plain.flux[take], injection.light_curve.flux, rtol=1e-12)
    np.testing.assert_allclose(plain.flux_err[take], injection.light_curve.flux_err, rtol=1e-12)


def test_finite_source_switch_is_converged_at_the_default_radius(demo_dict, event, survey):
    """The point-source/finite-source switch must not leave a visible step."""
    times = survey.times[np.abs(survey.times - event.t_0) < 1.0 * event.t_E]
    planet = Planet(1e-3, 1.3, 140.0)
    reference, n_ref = binary_magnification(
        event, planet, times, with_override(demo_dict, "injection", finite_source_radii=60.0)
    )
    default, n_default = binary_magnification(
        event, planet, times, with_override(demo_dict, "injection", finite_source_radii=20.0)
    )
    assert n_default < n_ref
    relative = np.abs(default - reference) / reference
    # 5e-4 in flux is 0.5 mmag, half the systematic floor of the Roman preset
    assert relative.max() < 5e-4


def test_finite_source_suppresses_the_caustic_crossing(demo_dict, event, survey):
    times = np.linspace(event.t_0 - 0.5 * event.t_E, event.t_0 + 0.5 * event.t_E, 4001)
    planet = Planet(1e-3, 1.3, 140.0)
    finite, n_finite = binary_magnification(
        event, planet, times, with_override(demo_dict, "injection", finite_source=True)
    )
    point, n_point = binary_magnification(
        event, planet, times, with_override(demo_dict, "injection", finite_source=False)
    )
    assert n_point == 0
    assert n_finite > 0
    # the point-source formula diverges on the fold; a finite source does not
    assert finite.max() < 0.5 * point.max()
    # the two calculations are identical away from the caustic, and differ only
    # at points where the finite-source method was actually applied
    relative = np.abs(finite - point) / point
    assert np.median(relative) == 0.0
    assert 0 < np.count_nonzero(relative > 1e-9) <= n_finite


def test_caustic_proximity_coincides_with_the_perturbation(event, config):
    """Trajectory and caustics must live in the same coordinate frame."""
    import MulensModel as mm
    from scipy.spatial import cKDTree

    planet = Planet(1e-3, 1.1, 90.0)
    times = np.linspace(event.t_0 - 0.5 * event.t_E, event.t_0 + 0.5 * event.t_E, 8001)
    parameters = mm.ModelParameters(
        {
            "t_0": event.t_0,
            "u_0": event.u_0,
            "t_E": event.t_E,
            "q": planet.q,
            "s": planet.s,
            "alpha": planet.alpha_deg,
        }
    )
    trajectory = mm.Trajectory(times, parameters)
    caustics, _ = caustic_extent(planet, 1000)
    distance, _ = cKDTree(caustics).query(np.column_stack([trajectory.x, trajectory.y]))
    binary, _ = binary_magnification(event, planet, times, config)
    single = pspl_magnification(event, times, finite_source=True)
    deviation = np.abs(binary - single) / single
    assert times[int(np.argmax(deviation))] == pytest.approx(
        times[int(np.argmin(distance))], abs=0.02 * event.t_E
    )


# --- anomaly window --------------------------------------------------------


def test_anomaly_grows_with_mass_ratio(event, survey, config):
    """A heavier planet perturbs the light curve detectably for longer.

    The *peak* deviation is deliberately not asserted to be monotonic: at fixed
    alpha a larger caustic is not necessarily crossed closer to a cusp, so the
    peak can fall while the duration rises.
    """
    durations = [
        locate_anomaly(event, Planet(q, 1.3, 140.0), survey, config).duration_days
        for q in (1e-5, 1e-4, 1e-3, 1e-2)
    ]
    assert all(d > 0.0 for d in durations)
    assert durations == sorted(durations)


def test_planetary_caustic_size_scales_as_sqrt_q():
    """The classic scaling: caustic width goes as sqrt(q) at fixed separation."""
    widths = []
    for q in (1e-5, 1e-4, 1e-3, 1e-2):
        points, _ = caustic_extent(Planet(q, 1.3, 0.0), 2000)
        widths.append(float(np.abs(points[:, 1]).max()))
    for small, large in itertools.pairwise(widths):
        assert large / small == pytest.approx(np.sqrt(10.0), rel=0.06)


def test_anomaly_window_is_found_even_inside_a_season_gap(demo_dict, survey):
    """An anomaly in a gap must still be located, then reported as unobserved."""
    config = Config.from_dict(demo_dict)
    windows = survey.seasons.windows
    gap_centre = 0.5 * (windows[0][1] + windows[1][0])
    event = Event(0, gap_centre, 0.3, 25.0, 21.0, 0.2, 1e-3)
    window = locate_anomaly(event, Planet(1e-3, 1.3, 140.0), survey, config)
    assert window.exists
    assert not survey.in_season(window.t_peak)[0]
    setup = EventSetup.build(event, survey, config)
    injection = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    assert injection.points_in_anomaly() == 0


def test_anomaly_window_contains_its_own_peak(event, survey, config):
    window = locate_anomaly(event, Planet(1e-3, 1.3, 140.0), survey, config)
    assert window.t_start <= window.t_peak <= window.t_end
    assert window.contains(np.array([window.t_peak]))[0]
    assert not window.contains(np.array([window.t_start - 1.0]))[0]


# --- injection bookkeeping -------------------------------------------------


def test_injection_is_deterministic(setup, survey, config):
    a = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    b = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    assert a.light_curve.flux.tobytes() == b.light_curve.flux.tobytes()


def test_planets_share_the_events_noise_realisation(setup, survey, config):
    """Grid-to-grid differences must come from the planet, not from the noise."""
    a = inject_planet(setup, Planet(1e-4, 1.3, 140.0), survey, config)
    b = inject_planet(setup, Planet(1e-3, 1.3, 20.0), survey, config)
    common = np.isin(a.light_curve.times, b.light_curve.times)
    other = np.isin(b.light_curve.times, a.light_curve.times)
    lc_a, lc_b = a.light_curve, b.light_curve
    z_a = (lc_a.flux - lc_a.f_source * a.binary_magnification - lc_a.f_blend) / lc_a.flux_err
    z_b = (lc_b.flux - lc_b.f_source * b.binary_magnification - lc_b.f_blend) / lc_b.flux_err
    np.testing.assert_allclose(z_a[common], z_b[other], rtol=1e-9)


def test_alpha_is_periodic(setup, survey, config):
    a = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    b = inject_planet(setup, Planet(1e-3, 1.3, 500.0), survey, config)
    np.testing.assert_allclose(a.binary_magnification, b.binary_magnification, rtol=1e-10)


def test_injected_light_curve_is_consistent_with_its_own_truth(setup, survey, config):
    injection = inject_planet(setup, Planet(1e-3, 1.3, 140.0), survey, config)
    lc = injection.light_curve
    truth = lc.f_source * injection.binary_magnification + lc.f_blend
    reduced = lc.chi2(truth) / lc.n_points
    assert reduced == pytest.approx(1.0, abs=5.0 * np.sqrt(2.0 / lc.n_points))


def test_injection_drops_saturated_points(demo_dict, survey):
    config = with_override(demo_dict, "survey", photometry={"saturation_mag": 18.0})
    bright_survey = Survey.from_config(config.survey, config.run.seed)
    event = Event(
        0,
        float(bright_survey.times[bright_survey.season_index == 2][4320]),
        0.01,
        25.0,
        19.5,
        0.1,
        1e-3,
    )
    setup = EventSetup.build(event, bright_survey, config)
    injection = inject_planet(setup, Planet(1e-3, 1.1, 90.0), bright_survey, config)
    lc = injection.light_curve
    assert lc.n_points < bright_survey.points_in_window(event.t_0, 3.0 * event.t_E)
    assert not bright_survey.is_saturated(
        lc.f_source * injection.binary_magnification + lc.f_blend
    ).any()


def test_analysis_window_bounds_the_retained_data(setup, survey, config, event):
    planet = Planet(1e-3, 1.3, 140.0)
    injection = inject_planet(setup, planet, survey, config)
    half = analysis_half_width(event, planet, config)
    assert np.all(np.abs(injection.light_curve.times - event.t_0) <= half)
    assert injection.light_curve.n_points > 1000
