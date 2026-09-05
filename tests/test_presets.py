"""Preset resolution and deep merging."""

from __future__ import annotations

import copy

import pytest

from lenseff.presets import PRESETS, resolve_survey_preset


def test_preset_is_expanded():
    resolved = resolve_survey_preset({"preset": "roman_gbtds"})
    assert resolved["cadence_minutes"] == pytest.approx(12.0)
    assert resolved["photometry"]["band"] == "W149"
    assert len(resolved["seasons"]["start_offsets_days"]) == 6


def test_user_keys_override_only_what_they_name():
    resolved = resolve_survey_preset(
        {"preset": "roman_gbtds", "photometry": {"systematic_floor_mmag": 5.0}}
    )
    assert resolved["photometry"]["systematic_floor_mmag"] == pytest.approx(5.0)
    # sibling keys survive the merge
    assert resolved["photometry"]["saturation_mag"] == pytest.approx(14.8)
    assert resolved["cadence_minutes"] == pytest.approx(12.0)


def test_resolution_does_not_mutate_the_registry():
    before = copy.deepcopy(PRESETS)
    resolved = resolve_survey_preset({"preset": "roman_gbtds"})
    resolved["photometry"]["saturation_mag"] = 0.0
    resolved["seasons"]["start_offsets_days"].append(9999.0)
    assert before == PRESETS


def test_custom_preset_passes_through():
    resolved = resolve_survey_preset({"preset": "custom", "cadence_minutes": 30.0})
    assert resolved == {"preset": "custom", "cadence_minutes": 30.0}


def test_missing_preset_key_defaults_to_custom():
    assert resolve_survey_preset({"cadence_minutes": 30.0})["preset"] == "custom"


def test_unknown_preset_lists_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown preset 'kmtnet'.*roman_gbtds"):
        resolve_survey_preset({"preset": "kmtnet"})


def test_non_mapping_survey_section():
    with pytest.raises(ValueError, match="expected a mapping"):
        resolve_survey_preset(["preset", "roman_gbtds"])
