"""The six scientific validation tests.

These are the tests that make the package trustworthy: each one asserts a
behaviour that follows from microlensing physics or from the statistics of the
detection criteria, and each would fail loudly if the pipeline were wrong in a
way unit tests would not catch.

They run small sweeps, so they are slower than the rest of the suite.  They are
marked ``validation`` and gate CI as their own job.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from lenseff.config import Config
from lenseff.detect import detect, expected_run_false_alarm_rate, longest_significant_run
from lenseff.events import Event
from lenseff.inject import EventSetup, Planet, inject_planet, locate_anomaly
from lenseff.survey import Survey

pytestmark = pytest.mark.validation


def make_config(demo_dict, **survey_overrides) -> Config:
    """Demo configuration with optional survey overrides."""
    data = copy.deepcopy(demo_dict)
    data["survey"].update(survey_overrides)
    return Config.from_dict(data)


def reference_event(survey: Survey, **overrides) -> Event:
    """A bright, well-covered event at the centre of season 2."""
    parameters = {
        "index": 0,
        "t_0": float(survey.times[survey.season_index == 2][4320]),
        "u_0": 0.3,
        "t_E": 25.0,
        "source_mag": 20.5,
        "blend_ratio": 0.2,
        "rho": 1.0e-3,
    }
    parameters.update(overrides)
    return Event(**parameters)


def efficiency(
    config: Config,
    survey: Survey,
    event: Event,
    q: float,
    s: float,
    n_alpha: int = 12,
) -> float:
    """Detected fraction over ``n_alpha`` trajectory angles at fixed (q, s)."""
    setup = EventSetup.build(event, survey, config)
    angles = np.linspace(0.0, 360.0, n_alpha, endpoint=False) + 180.0 / n_alpha
    detections = 0
    for alpha in angles:
        injection = inject_planet(setup, Planet(q, s, float(alpha)), survey, config)
        detections += int(detect(injection, survey, config).detected)
    return detections / n_alpha


# --- 1. the planet-free control -------------------------------------------


def test_zero_q_delta_chi2_criterion_can_never_fire(demo_dict):
    """A free PSPL refit cannot fit worse than the injected PSPL model.

    The refit family contains the injected planet-free model, so
    ``Delta chi2 <= 0`` exactly, for every control, and the chi-square
    criterion's false-positive rate is zero by construction rather than merely
    small.  This is the precise form of the "q = 0 gives the false-positive
    rate implied by the threshold" requirement.
    """
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    statistics = []
    for realisation in range(12):
        event = reference_event(survey, u_0=0.15 + 0.05 * realisation)
        setup = EventSetup.build(event, survey, config, realisation=realisation)
        result = detect(inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config), survey, config)
        statistics.append(result.delta_chi2 / result.chi2_binary)
        assert not result.detected
        assert not result.criteria["delta_chi2"]
    # zero up to float64 rounding: a refit landing exactly on the truth
    # returns a difference of order 1e-16 times a chi-square of order 1e4
    assert max(statistics) <= 1e-9
    # three extra free parameters are worth a few units of chi-square, no more
    assert min(statistics) > -40.0 / 8000.0


def test_consecutive_points_false_alarm_rate_matches_the_closed_form(demo_dict):
    """The criterion that *can* fire on noise, measured against its own theory.

    The false-positive rate of the run criterion is analytic for independent
    Gaussian residuals.  Measuring it needs ~1e5 light curves, which is why it
    is measured on synthetic noise rather than on a few hundred injections --
    the rate is real, but far too small to estimate from a control batch.
    """
    config = make_config(demo_dict)
    detection = config.detection
    rng = np.random.default_rng(20260905)
    n_points, n_curves = 2000, 4000
    times = np.arange(n_points) * (12.0 / 1440.0)
    hits = 0
    for _ in range(n_curves):
        residual = rng.standard_normal(n_points)
        length, _, _ = longest_significant_run(
            residual,
            times,
            sigma=detection.point_sigma,
            same_sign=detection.require_same_sign,
            max_gap_days=detection.max_gap_within_run_days,
        )
        hits += int(length >= detection.consecutive_points)
    predicted = expected_run_false_alarm_rate(
        n_points,
        detection.point_sigma,
        detection.consecutive_points,
        same_sign=detection.require_same_sign,
    )
    expected_hits = predicted * n_curves
    # Poisson agreement: |measured - expected| within 4 sqrt(expected) + 1
    assert abs(hits - expected_hits) <= 4.0 * np.sqrt(max(expected_hits, 1.0)) + 1.0
    assert 0.0 < predicted < 1e-3


# --- 2. efficiency rises with mass ratio ----------------------------------


def test_efficiency_increases_with_mass_ratio(demo_dict):
    """At fixed separation, in the well-sampled regime."""
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    event = reference_event(survey)
    values = [
        efficiency(config, survey, event, q, 1.0, n_alpha=12) for q in (1e-5, 1e-4, 1e-3, 1e-2)
    ]
    for lower, upper in itertools.pairwise(values):
        assert upper >= lower - 0.1, f"efficiency fell with q: {values}"
    assert values[-1] - values[0] > 0.4, f"no rise across three decades of q: {values}"
    assert values[0] < 0.35
    assert values[-1] > 0.75


# --- 3. the lensing zone ---------------------------------------------------


def test_efficiency_peaks_near_the_einstein_ring(demo_dict):
    """The lensing zone: caustics are largest and closest to the source path near s = 1."""
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    event = reference_event(survey)
    separations = [0.2, 0.5, 1.0, 2.0, 5.0]
    values = [efficiency(config, survey, event, 1e-3, s, n_alpha=12) for s in separations]
    peak = separations[int(np.argmax(values))]
    assert peak in (0.5, 1.0, 2.0), (
        f"peak at s = {peak}: {dict(zip(separations, values, strict=True))}"
    )
    assert values[2] > values[0] + 0.2, f"close separations not suppressed: {values}"
    assert values[2] > values[4] + 0.2, f"wide separations not suppressed: {values}"


# --- 4. photometric precision ---------------------------------------------


def test_better_photometry_raises_efficiency(demo_dict):
    """Halving every uncertainty must gain planets, and never lose any.

    The mass ratio is chosen in the regime where the criteria are marginal, so
    there is room to improve; at ``q = 1e-2`` the anomalies are already far
    above threshold and better photometry buys nothing.
    """
    values = []
    for scale in (1.0, 0.5, 0.25):
        config = make_config(demo_dict, precision_scale=scale)
        survey = Survey.from_config(config.survey, config.run.seed)
        event = reference_event(survey)
        values.append(efficiency(config, survey, event, 1e-4, 1.0, n_alpha=16))
    for worse, better in itertools.pairwise(values):
        assert better >= worse, f"precision cost efficiency: {values}"
    assert values[1] > values[0], f"halving sigma bought nothing: {values}"
    assert values[-1] > values[0] + 0.2, f"no gain across a factor of four: {values}"


# --- 5. season gaps --------------------------------------------------------


def test_an_anomaly_inside_a_season_gap_is_never_recovered(demo_dict):
    """The same planet, recovered in season and invisible in the gap."""
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    windows = survey.seasons.windows
    # the *short* gap between two adjacent seasons, so the analysis window
    # still holds thousands of measurements: the anomaly is missing, not the
    # whole event
    gap_centre = 0.5 * (windows[0][1] + windows[1][0])

    in_season = reference_event(survey)
    in_gap = reference_event(survey, t_0=gap_centre)

    recovered = efficiency(config, survey, in_season, 1e-3, 1.0, n_alpha=12)
    assert recovered > 0.4, "the control case must be recoverable for the test to mean anything"

    setup = EventSetup.build(in_gap, survey, config)
    assert (
        inject_planet(setup, Planet(1e-3, 1.0, 0.0), survey, config).light_curve.n_points > 1000
    ), "the event itself must still be observed, only its anomaly hidden"
    for alpha in np.linspace(0.0, 360.0, 12, endpoint=False):
        injection = inject_planet(setup, Planet(1e-3, 1.0, float(alpha)), survey, config)
        result = detect(injection, survey, config)
        assert injection.points_in_anomaly() == 0, "the anomaly must fall in the gap"
        assert not result.detected
        assert abs(result.delta_chi2) < config.detection.delta_chi2_min


# --- 6. a published configuration ------------------------------------------


def test_a_known_recoverable_planet_is_recovered(demo_dict):
    """A configuration approximating OGLE-2005-BLG-390Lb (Beaulieu et al. 2006).

    Rounded published values: ``q = 7.6e-5``, ``s = 1.61``, ``t_E = 11.0 d``,
    ``u_0 = 0.36``, a giant source.  The real detection was made from the
    ground with survey-plus-follow-up photometry, so recovering it on the Roman
    calendar is an anchor rather than a reproduction: a configuration known to
    be detectable in reality must not come out undetectable here.

    The trajectory angle is *not* scanned coarsely.  At this mass ratio only a
    few per cent of angles bring the source near the planetary caustic at all,
    so a 24-point scan misses it and would make this test measure the sampling
    rather than the pipeline.  The angle is taken from the geometry instead --
    which is itself checked below.
    """
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    event = reference_event(
        survey, u_0=0.359, t_E=11.03, source_mag=17.5, blend_ratio=0.1, rho=0.0059
    )
    setup = EventSetup.build(event, survey, config)
    q, s = 7.6e-5, 1.610

    # A trajectory with impact parameter u_0 crosses the binary axis at
    # |x| = u_0 / sin(alpha); setting that equal to the planetary caustic
    # position |s - 1/s| gives the angle that produces the anomaly.
    caustic_x = abs(s - 1.0 / s)
    predicted = float(np.degrees(np.arcsin(event.u_0 / caustic_x)))

    scan = np.arange(predicted - 4.0, predicted + 4.0, 0.25)
    windows = [
        (alpha, locate_anomaly(event, Planet(q, s, float(alpha)), survey, config)) for alpha in scan
    ]
    best_alpha, best_window = max(windows, key=lambda item: item[1].peak_deviation)
    assert best_window.peak_deviation > 50.0, (
        "the analytic caustic-crossing angle must actually cross the caustic; "
        f"got {best_window.peak_deviation:.1f} sigma at alpha = {best_alpha:.2f}"
    )

    result = detect(
        inject_planet(setup, Planet(q, s, float(best_alpha)), survey, config), survey, config
    )
    assert result.detected, "the published configuration must be recovered"
    assert result.delta_chi2 > 10.0 * config.detection.delta_chi2_min
    assert result.run_length >= config.detection.consecutive_points


def test_a_diffuse_low_amplitude_deviation_is_rejected(demo_dict):
    """The consecutive-points criterion earning its place.

    The same published configuration at a trajectory angle that *misses* the
    caustic still accumulates Delta chi-square of several hundred -- above the
    threshold -- from hundreds of points each deviating by around one sigma.
    In real data a 1-sigma trend over three days is indistinguishable from
    correlated noise, and the run criterion is what rejects it.
    """
    config = make_config(demo_dict)
    survey = Survey.from_config(config.survey, config.run.seed)
    event = reference_event(
        survey, u_0=0.359, t_E=11.03, source_mag=17.5, blend_ratio=0.1, rho=0.0059
    )
    setup = EventSetup.build(event, survey, config)
    injection = inject_planet(setup, Planet(7.6e-5, 1.610, 150.0), survey, config)
    result = detect(injection, survey, config)
    assert injection.anomaly.peak_deviation < 3.0, "this angle must miss the caustic"
    assert result.criteria["delta_chi2"], "and still pass the chi-square threshold"
    assert not result.criteria["consecutive_points"]
    assert not result.detected
