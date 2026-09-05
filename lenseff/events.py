"""Baseline PSPL event sampling, from a population model or a supplied catalog.

Phase 1.  Planned public interface::

    @dataclass(frozen=True)
    class Event:  # t_0, u_0, t_E, source_mag, blend_ratio, rho, index
        ...

    def sample_events(config: Config) -> list[Event]
    def load_catalog(path: Path, config: Config) -> list[Event]
    def simulate_light_curve(event, survey, seed) -> LightCurve

Not implemented yet.
"""

from __future__ import annotations
