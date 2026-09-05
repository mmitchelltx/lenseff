"""Validation and determinism of the configuration layer."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import yaml

from lenseff.config import AxisSpec, Config, ConfigError


def test_demo_config_loads(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    assert config.run.name == "roman_gbtds_demo"
    assert config.run.seed == 20260905
    assert config.survey.preset == "roman_gbtds"
    assert config.survey.cadence_minutes == pytest.approx(12.0)
    assert config.survey.seasons.n_seasons == 6
    assert config.survey.seasons.length_days == pytest.approx(72.0)


def test_preset_values_are_merged_and_explicit(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    phot = config.survey.photometry
    assert phot.band == "W149"
    assert phot.saturation_mag == pytest.approx(14.8)
    # the resolved (not the abbreviated) survey block is what gets hashed
    assert "zero_point" in config.to_dict()["survey"]["photometry"]


def test_config_hash_is_deterministic(demo_config_path):
    a = Config.from_yaml(demo_config_path).config_hash()
    b = Config.from_yaml(demo_config_path).config_hash()
    assert a == b
    assert len(a) == 64


def test_config_hash_ignores_key_order_and_source_path(demo_dict, tmp_path):
    reordered = dict(reversed(list(demo_dict.items())))
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "nested" / "b.yaml"
    path_b.parent.mkdir()
    path_a.write_text(yaml.safe_dump(demo_dict), encoding="utf-8")
    path_b.write_text(yaml.safe_dump(reordered), encoding="utf-8")
    assert Config.from_yaml(path_a).config_hash() == Config.from_yaml(path_b).config_hash()


def test_config_hash_changes_with_any_value(demo_dict):
    base = Config.from_dict(demo_dict).config_hash()
    demo_dict["detection"]["delta_chi2_min"] = 160.0000001
    assert Config.from_dict(demo_dict).config_hash() != base


def test_scheduling_choices_do_not_change_the_hash(demo_dict):
    """Resuming a checkpointed run on a different machine must be allowed."""
    base = Config.from_dict(demo_dict).config_hash()
    for key, value in (
        ("n_workers", 17),
        ("chunk_size", 3),
        ("checkpoint_every", 11),
        ("start_method", "spawn"),
        ("max_records_in_memory", 42),
    ):
        demo_dict["compute"][key] = value
        assert Config.from_dict(demo_dict).config_hash() == base
    # but compute settings are still recorded in full
    assert Config.from_dict(demo_dict).to_dict()["compute"]["n_workers"] == 17


def test_canonical_json_is_sorted_and_finite(demo_config_path):
    text = Config.from_yaml(demo_config_path).canonical_json()
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == text


def test_detection_defaults_match_the_specification(demo_dict):
    del demo_dict["detection"]
    detection = Config.from_dict(demo_dict).detection
    assert detection.delta_chi2_min == pytest.approx(160.0)
    assert detection.consecutive_points == 3
    assert detection.point_sigma == pytest.approx(3.0)
    assert detection.require_in_season is True
    assert detection.refit.free_blending is True
    assert detection.refit.n_starts >= 1


def test_unknown_key_is_rejected_with_its_path(demo_dict):
    demo_dict["detection"]["delta_chisq_min"] = 160.0
    with pytest.raises(ConfigError, match=r"detection: unknown key\(s\) \['delta_chisq_min'\]"):
        Config.from_dict(demo_dict)


def test_unknown_top_level_section_is_rejected(demo_dict):
    demo_dict["photometry"] = {}
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(demo_dict)


def test_missing_required_section_is_rejected(demo_dict):
    del demo_dict["events"]
    with pytest.raises(ConfigError, match="events: required section is missing"):
        Config.from_dict(demo_dict)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("run", "seed", -1, "must be >= 0"),
        ("survey", "cadence_minutes", 0.0, "must be > 0"),
        ("survey", "precision_scale", -1.0, "must be > 0"),
        ("survey", "dropout_fraction", 1.0, "must be < 1"),
        ("events", "n_events", 0, "must be >= 1"),
        ("output", "confidence_level", 1.0, "must be < 1"),
        ("compute", "chunk_size", 0, "must be >= 1"),
    ],
)
def test_out_of_range_values_are_rejected(demo_dict, section, key, value, message):
    demo_dict[section][key] = value
    with pytest.raises(ConfigError, match=message):
        Config.from_dict(demo_dict)


def test_string_where_a_number_is_expected(demo_dict):
    demo_dict["detection"]["delta_chi2_min"] = "160"
    with pytest.raises(ConfigError, match="expected a number"):
        Config.from_dict(demo_dict)


def test_bool_is_not_accepted_as_a_number(demo_dict):
    demo_dict["survey"]["precision_scale"] = True
    with pytest.raises(ConfigError, match="expected a number"):
        Config.from_dict(demo_dict)


def test_overlapping_seasons_are_rejected(demo_dict):
    demo_dict["survey"]["seasons"] = {"start_offsets_days": [0.0, 50.0, 400.0]}
    with pytest.raises(ConfigError, match="seasons overlap"):
        Config.from_dict(demo_dict)


def test_season_offsets_must_increase(demo_dict):
    demo_dict["survey"]["seasons"] = {"start_offsets_days": [0.0, 400.0, 200.0]}
    with pytest.raises(ConfigError, match="strictly increasing"):
        Config.from_dict(demo_dict)


def test_faint_limit_must_be_fainter_than_saturation(demo_dict):
    demo_dict["survey"]["photometry"] = {"faint_limit_mag": 12.0}
    with pytest.raises(ConfigError, match="fainter"):
        Config.from_dict(demo_dict)


def test_distribution_bounds_are_checked(demo_dict):
    demo_dict["events"]["distributions"]["u_0"] = {"dist": "uniform", "min": 1.0, "max": 0.0}
    with pytest.raises(ConfigError, match="max must exceed min"):
        Config.from_dict(demo_dict)


def test_loguniform_requires_positive_minimum(demo_dict):
    demo_dict["events"]["distributions"]["rho"] = {"dist": "loguniform", "min": 0.0, "max": 1.0}
    with pytest.raises(ConfigError, match="loguniform requires min > 0"):
        Config.from_dict(demo_dict)


def test_unknown_distribution_family(demo_dict):
    demo_dict["events"]["distributions"]["u_0"] = {"dist": "beta", "min": 0.0, "max": 1.0}
    with pytest.raises(ConfigError, match="expected one of"):
        Config.from_dict(demo_dict)


def test_catalog_mode_requires_a_path(demo_dict):
    demo_dict["events"]["source"] = "catalog"
    with pytest.raises(ConfigError, match=r"required when events\.source is 'catalog'"):
        Config.from_dict(demo_dict)


def test_axis_values_are_exact():
    axis = AxisSpec(min=-5.0, max=-2.0, n=7)
    values = axis.values()
    assert values.shape == (7,)
    assert values[0] == pytest.approx(-5.0)
    assert values[-1] == pytest.approx(-2.0)
    np.testing.assert_allclose(np.diff(values), 0.5)
    np.testing.assert_allclose(AxisSpec(min=1.5, max=9.0, n=1).values(), [1.5])


def test_axis_requires_max_above_min_when_n_gt_1(demo_dict):
    demo_dict["injection"]["grid"]["log_q"] = {"min": -2.0, "max": -5.0, "n": 7}
    with pytest.raises(ConfigError, match="max must exceed min"):
        Config.from_dict(demo_dict)


def test_injection_count_arithmetic(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    grid = config.injection.grid
    assert grid.n_cells == 81
    expected = 81 * config.events.n_events * grid.n_alpha + config.injection.n_zero_q_trials
    assert config.n_injections() == expected == 5384


def test_events_per_cell_overrides_the_sample_size(demo_dict):
    demo_dict["injection"]["grid"]["events_per_cell"] = 3
    demo_dict["injection"]["include_zero_q_control"] = False
    assert Config.from_dict(demo_dict).n_injections() == 81 * 3 * 8


def test_empty_and_invalid_yaml(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="is empty"):
        Config.from_yaml(empty)
    broken = tmp_path / "broken.yaml"
    broken.write_text("run: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Config.from_yaml(broken)


def test_season_windows_are_absolute_and_non_overlapping(demo_config_path):
    seasons = Config.from_yaml(demo_config_path).survey.seasons
    windows = seasons.windows
    assert len(windows) == 6
    assert windows[0][0] == pytest.approx(seasons.t_start)
    assert windows[0][1] - windows[0][0] == pytest.approx(72.0)
    for (_, end), (start, _) in itertools.pairwise(windows):
        assert start > end
    assert seasons.total_span_days == pytest.approx(1825.25 + 72.0)


def test_cadence_days_conversion(demo_config_path):
    survey = Config.from_yaml(demo_config_path).survey
    assert survey.cadence_days == pytest.approx(12.0 / 1440.0)
    assert survey.cadence_days * 1440.0 == pytest.approx(survey.cadence_minutes)
