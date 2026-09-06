"""Figure generation."""

from __future__ import annotations

import numpy as np
import pytest

from lenseff.config import Config
from lenseff.events import sample_events, simulate_light_curve
from lenseff.plotting import plot_light_curve
from lenseff.survey import Survey


@pytest.fixture
def pieces(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    survey = Survey.from_config(config.survey, config.run.seed)
    event = sample_events(config, survey)[0]
    return config, survey, simulate_light_curve(event, survey, config)


def test_light_curve_plot_has_data_and_inverted_magnitude_axis(pieces):
    _, survey, light_curve = pieces
    figure = plot_light_curve(light_curve, survey)
    ax = figure.axes[0]
    bottom, top = ax.get_ylim()
    assert bottom > top, "magnitude axes must run brighter-upwards"
    assert ax.get_ylabel() == "W149 magnitude"
    assert len(ax.containers) == 1
    # season shading is drawn for the seasons overlapping the plotted window
    assert len(ax.patches) >= 1


def test_zoom_window_restricts_the_plotted_range(pieces):
    _, survey, light_curve = pieces
    event = light_curve.event
    figure = plot_light_curve(light_curve, survey, window_t_E=1.0)
    lo, hi = figure.axes[0].get_xlim()
    assert hi - lo <= 2.2 * event.t_E


def test_model_overlay_is_drawn(pieces):
    config, survey, light_curve = pieces
    times = np.linspace(light_curve.times[0], light_curve.times[-1], 100)
    flux = np.full_like(times, light_curve.f_source + light_curve.f_blend)
    figure = plot_light_curve(
        light_curve, survey, window_t_E=None, model_curves={"baseline": (times, flux)}
    )
    assert [line.get_label() for line in figure.axes[0].lines if line.get_label() == "baseline"]
    assert config.run.name
