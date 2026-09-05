"""Parallel sweep over the ``(q, s)`` grid with checkpointing.

Phase 4.  Marginalises over ``alpha`` and over the baseline event population,
streams per-injection records to parquet shards so that a 1e6-injection run
never holds more than ``compute.max_records_in_memory`` rows, and resumes from
the last completed shard.  Planned public interface::

    def run_grid(config: Config, *, progress: bool = True) -> Path

Not implemented yet.
"""

from __future__ import annotations
