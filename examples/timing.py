"""Measure the cost of one injection-recovery cycle.

Run this before launching any large grid: multiply the reported cycle time by
``lenseff validate <config>``'s injection count to get the CPU-hours.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lenseff.config import Config
from lenseff.detect import detect
from lenseff.events import sample_events
from lenseff.inject import EventSetup, Planet, inject_planet
from lenseff.survey import Survey


def main(path: str = "configs/roman_gbtds_demo.yaml", n_events: int = 3) -> None:
    """Time injection and detection across the configured grid."""
    config = Config.from_yaml(path)
    survey = Survey.from_config(config.survey, config.run.seed)
    events = sample_events(config, survey)[:n_events]
    grid = config.injection.grid
    rng = np.random.default_rng(0)

    setup_times, inject_times, detect_times = [], [], []
    for event in events:
        start = time.perf_counter()
        setup = EventSetup.build(event, survey, config)
        setup_times.append(time.perf_counter() - start)
        for log_q in grid.log_q.values():
            for log_s in grid.log_s.values()[::2]:
                planet = Planet(10.0**log_q, 10.0**log_s, float(rng.uniform(0.0, 360.0)))
                start = time.perf_counter()
                injection = inject_planet(setup, planet, survey, config)
                middle = time.perf_counter()
                detect(injection, survey, config)
                end = time.perf_counter()
                inject_times.append(middle - start)
                detect_times.append(end - middle)

    cycle = np.array(inject_times) + np.array(detect_times)
    print(f"config            : {path}")
    print(f"cycles timed      : {cycle.size}")
    print(f"event setup       : {np.mean(setup_times) * 1e3:7.1f} ms (once per event)")
    print(f"injection         : {np.mean(inject_times) * 1e3:7.1f} ms")
    print(f"detection         : {np.mean(detect_times) * 1e3:7.1f} ms")
    print(
        f"cycle mean / p90  : {cycle.mean() * 1e3:7.1f} / {np.percentile(cycle, 90) * 1e3:.1f} ms"
    )
    total = config.n_injections()
    hours = total * cycle.mean() / 3600.0
    print(f"\nthis config: {total:,} injections -> {hours:.2f} CPU-hours")
    for cores in (1, 4, 8, 64):
        print(f"  {cores:3d} cores: {hours / cores * 60:8.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:])
