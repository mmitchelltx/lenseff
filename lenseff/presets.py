"""Named survey presets and the deep-merge that resolves them.

A preset supplies a complete ``survey`` section; anything the user writes in
their own ``survey`` block overrides the preset key by key.  The *resolved*
values are what get validated, hashed and written into the output provenance,
so a run record is never a bare preset name that could later drift.

Provisional numbers
-------------------
The ``roman_gbtds`` values below are placeholders pending the Phase 1 review.
Each is tagged with the source it should be checked against before any science
run: Penny et al. (2019, ApJS 241, 3) for the simulated survey design and
photometric performance, and the Roman Galactic Bulge Time Domain Survey
definition for the adopted cadence and season layout.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["PRESETS", "resolve_survey_preset"]

# NOTE (Phase 1 sign-off item): every number here is provisional.
_ROMAN_GBTDS: Final[dict[str, Any]] = {
    "cadence_minutes": 12.0,  # GBTDS design cadence in the wide band
    "seasons": {
        # ~2027-04-13; the first season start is a placeholder for the real
        # bulge-visibility window of the adopted launch date.
        "t_start": 2461508.0,
        "length_days": 72.0,
        # Three seasons early and three late in the primary mission, to give a
        # long astrometric baseline.  Spacing is one bulge-visibility window
        # (half a year) apart.
        "start_offsets_days": [0.0, 182.625, 365.25, 1460.0, 1642.625, 1825.25],
    },
    "photometry": {
        "band": "W149",
        "zero_point": 27.615,  # mag giving 1 e/s
        "exposure_time_s": 46.8,
        "n_exposures": 1,
        "sky_e_per_s_per_pixel": 4.0,
        "read_noise_e": 10.0,
        "n_pixels": 5.0,
        "systematic_floor_mmag": 1.0,
        "saturation_mag": 14.8,
        "faint_limit_mag": 26.0,
    },
}

PRESETS: Final[dict[str, dict[str, Any]]] = {"roman_gbtds": _ROMAN_GBTDS}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_survey_preset(survey: Any) -> dict[str, Any]:
    """Expand ``survey.preset`` into a fully explicit survey mapping.

    Args:
        survey: The raw ``survey`` section of a configuration document.

    Returns:
        The survey mapping with preset values filled in and user keys applied
        on top.  ``preset`` is retained so the provenance record says which
        preset the run started from.

    Raises:
        ValueError: If ``survey`` is not a mapping or names an unknown preset.
    """
    if not isinstance(survey, Mapping):
        raise ValueError(f"survey: expected a mapping, got {type(survey).__name__}")
    name = survey.get("preset", "custom")
    if not isinstance(name, str):
        raise ValueError(f"survey.preset: expected a string, got {name!r}")
    if name == "custom":
        return {**copy.deepcopy(dict(survey)), "preset": "custom"}
    if name not in PRESETS:
        known = sorted([*PRESETS, "custom"])
        raise ValueError(f"survey.preset: unknown preset {name!r}; known presets are {known}")
    merged = _deep_merge(PRESETS[name], survey)
    merged["preset"] = name
    return merged
