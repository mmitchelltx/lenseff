"""Aggregation of injection records into an efficiency surface.

Phase 5.  Per-cell detection fractions with Wilson-score binomial confidence
intervals.  Planned public interface::

    def aggregate(records: pd.DataFrame, config: Config) -> pd.DataFrame
    def wilson_interval(k: int, n: int, confidence: float) -> tuple[float, float]

Not implemented yet.
"""

from __future__ import annotations
