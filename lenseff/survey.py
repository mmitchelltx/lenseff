"""Observing calendar, photometric error model and saturation limits.

Phase 1.  Planned public interface::

    class Survey:
        @classmethod
        def from_config(cls, config: SurveyConfig) -> Survey
        def times(self, seed: int) -> np.ndarray          # HJD of every visit
        def season_index(self, times) -> np.ndarray       # -1 outside a season
        def in_season(self, times) -> np.ndarray
        def magnitude_error(self, mag) -> np.ndarray      # sigma in mag
        def flux_error(self, flux) -> np.ndarray
        def is_saturated(self, mag) -> np.ndarray

Not implemented yet.
"""

from __future__ import annotations
