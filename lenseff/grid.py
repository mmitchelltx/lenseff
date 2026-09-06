"""Parallel sweep over the ``(q, s)`` grid, with checkpointing and resume.

The sweep is organised so that three things hold at once: it is reproducible,
it is restartable, and it never holds more than a bounded number of rows in
memory.

**Reproducible.**  Nothing is drawn from a running sequence.  Trajectory
angles come from ``rng.generator(seed, "alpha", cell, event)`` and photometric
noise from ``rng.generator(seed, "photometric_noise", event, realisation)``, so
every injection's inputs are a pure function of the configuration, the seed and
the injection's own address.  A run on one worker and a run on sixty-four
produce identical rows, and so does a run that was interrupted and resumed.

**Restartable.**  Work is split into batches of tasks; completed rows are
flushed to parquet shards under ``injections/`` and every row carries its
``task_index``.  Resuming reads the ``task_index`` column of the existing
shards -- cheap, because parquet is columnar -- and skips those tasks.  A
resume into a directory written by a *different* configuration is refused.

**Bounded.**  Rows are accumulated only until ``compute.checkpoint_every`` and
then written out, so a 10^6-injection run costs the same memory as a small one.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from lenseff.config import Config
from lenseff.detect import detect
from lenseff.events import Event, sample_events
from lenseff.inject import EventSetup, Planet, inject_planet
from lenseff.provenance import collect_provenance, write_parquet
from lenseff.rng import generator
from lenseff.survey import Survey

__all__ = [
    "GridCell",
    "Task",
    "alpha_values",
    "build_tasks",
    "grid_cells",
    "run_grid",
]

#: Name of the shard directory inside the run output directory.
SHARD_DIR: Final[str] = "injections"

#: Worker-process state, populated by :func:`_init_worker`.
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class GridCell:
    """One cell of the ``(log q, log s)`` grid.

    Attributes:
        index: Position in the flattened grid.
        i_q: Index along the ``log q`` axis.
        i_s: Index along the ``log s`` axis.
        log_q: Base-10 log of the mass ratio.
        log_s: Base-10 log of the separation.
    """

    index: int
    i_q: int
    i_s: int
    log_q: float
    log_s: float

    @property
    def q(self) -> float:
        """The mass ratio."""
        return float(10.0**self.log_q)

    @property
    def s(self) -> float:
        """The separation in Einstein radii."""
        return float(10.0**self.log_s)


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of work: every angle for one ``(event, cell)`` pair.

    Attributes:
        index: Global task index, used for checkpoint bookkeeping.
        kind: ``"planet"`` or ``"control"``.
        event_index: Index into the sampled event list.
        cell_index: Grid cell, or ``-1`` for a control.
        trial_index: Control trial number, or ``-1`` for a planet task.  It
            also selects the noise realisation, so controls are independent.
    """

    index: int
    kind: str
    event_index: int
    cell_index: int
    trial_index: int


def grid_cells(config: Config) -> tuple[GridCell, ...]:
    """Return every cell of the ``(log q, log s)`` grid, in row-major order."""
    grid = config.injection.grid
    log_q_values = grid.log_q.values()
    log_s_values = grid.log_s.values()
    cells: list[GridCell] = []
    for i_q, log_q in enumerate(log_q_values):
        for i_s, log_s in enumerate(log_s_values):
            cells.append(GridCell(len(cells), i_q, i_s, float(log_q), float(log_s)))
    return tuple(cells)


def alpha_values(config: Config, cell_index: int, event_index: int) -> np.ndarray:
    """Return the trajectory angles for one ``(cell, event)`` pair.

    ``stratified`` draws one angle uniformly inside each of ``n_alpha`` equal
    bins of ``[0, 360)``.  That is unbiased -- the marginal distribution is
    still uniform -- but has much lower variance than independent draws,
    because it cannot leave a quadrant of trajectory space unsampled.

    Args:
        config: The run configuration.
        cell_index: Grid cell index.
        event_index: Event index.

    Returns:
        The angles in degrees.
    """
    grid = config.injection.grid
    n = grid.n_alpha
    if grid.alpha_mode == "uniform":
        return np.linspace(0.0, 360.0, n, endpoint=False)
    rng = generator(config.run.seed, "alpha", cell_index, event_index)
    if grid.alpha_mode == "random":
        return np.sort(rng.uniform(0.0, 360.0, size=n))
    edges = np.linspace(0.0, 360.0, n + 1)
    return edges[:-1] + rng.uniform(0.0, 1.0, size=n) * (360.0 / n)


def build_tasks(config: Config, n_events: int) -> tuple[Task, ...]:
    """Enumerate every unit of work, in a deterministic order.

    Planet tasks are ordered event-major so that a worker processing
    consecutive tasks reuses its cached :class:`~lenseff.inject.EventSetup`
    instead of rebuilding it.

    Args:
        config: The run configuration.
        n_events: Number of sampled events.

    Returns:
        The task list.
    """
    grid = config.injection.grid
    per_cell = min(grid.events_per_cell or n_events, n_events)
    tasks: list[Task] = []
    for event_index in range(per_cell):
        for cell_index in range(grid.n_cells):
            tasks.append(Task(len(tasks), "planet", event_index, cell_index, -1))
    if config.injection.include_zero_q_control:
        for trial in range(config.injection.n_zero_q_trials):
            tasks.append(Task(len(tasks), "control", trial % n_events, -1, trial))
    return tuple(tasks)


def _event_setup(event: Event, survey: Survey, config: Config, realisation: int) -> EventSetup:
    """Return a cached event setup, rebuilding only when the address changes."""
    key = (event.index, realisation)
    cached = _WORKER.get("setup")
    if cached is not None and _WORKER.get("setup_key") == key:
        return cached  # type: ignore[no-any-return]
    setup = EventSetup.build(event, survey, config, realisation=realisation)
    _WORKER["setup"] = setup
    _WORKER["setup_key"] = key
    return setup


def _init_worker(config: Config) -> None:
    """Rebuild the survey and event sample inside a worker process."""
    survey = Survey.from_config(config.survey, config.run.seed)
    _WORKER.clear()
    _WORKER["config"] = config
    _WORKER["survey"] = survey
    _WORKER["events"] = sample_events(config, survey)
    _WORKER["cells"] = grid_cells(config)


def _run_task(task: Task) -> list[dict[str, Any]]:
    """Execute one task and return one record per injection."""
    config: Config = _WORKER["config"]
    survey: Survey = _WORKER["survey"]
    events: tuple[Event, ...] = _WORKER["events"]
    cells: tuple[GridCell, ...] = _WORKER["cells"]
    event = events[task.event_index]

    if task.kind == "control":
        setup = _event_setup(event, survey, config, task.trial_index)
        injection = inject_planet(setup, Planet(0.0, 1.0, 0.0), survey, config)
        result = detect(injection, survey, config)
        return [_record(task, None, 0, 0.0, event, injection, result)]

    cell = cells[task.cell_index]
    setup = _event_setup(event, survey, config, 0)
    records: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(alpha_values(config, task.cell_index, task.event_index)):
        planet = Planet(cell.q, cell.s, float(alpha))
        injection = inject_planet(setup, planet, survey, config)
        result = detect(injection, survey, config)
        records.append(_record(task, cell, alpha_index, float(alpha), event, injection, result))
    return records


def _record(
    task: Task,
    cell: GridCell | None,
    alpha_index: int,
    alpha: float,
    event: Event,
    injection: Any,
    result: Any,
) -> dict[str, Any]:
    """Flatten one injection into an output row."""
    record: dict[str, Any] = {
        "task_index": task.index,
        "kind": task.kind,
        "cell_index": task.cell_index,
        "i_q": cell.i_q if cell is not None else -1,
        "i_s": cell.i_s if cell is not None else -1,
        "log_q": cell.log_q if cell is not None else float("-inf"),
        "log_s": cell.log_s if cell is not None else float("nan"),
        "alpha_index": alpha_index,
        "alpha_deg": alpha,
        "trial_index": task.trial_index,
        "n_points": injection.light_curve.n_points,
        "anomaly_exists": injection.anomaly.exists,
        "anomaly_duration_days": injection.anomaly.duration_days,
        "anomaly_peak_sigma": injection.anomaly.peak_deviation,
        "n_finite_source_points": injection.n_finite_source_points,
    }
    record.update(event.as_record())
    record.update(injection.planet.as_record())
    record.update(result.as_record())
    return record


def _completed_tasks(shard_dir: Path) -> set[int]:
    """Read the task indices already present in the shard directory."""
    completed: set[int] = set()
    for shard in sorted(shard_dir.glob("shard_*.parquet")):
        try:
            column = pd.read_parquet(shard, columns=["task_index"])
        except (OSError, ValueError):  # a shard truncated by a hard kill
            shard.unlink(missing_ok=True)
            continue
        completed.update(int(value) for value in column["task_index"].to_numpy())
    return completed


def _check_resume(output_dir: Path, config: Config) -> bool:
    """Decide whether an existing output directory may be resumed into.

    Returns:
        Whether previous shards should be reused.

    Raises:
        ValueError: If the directory holds a run of a different configuration
            and ``run.overwrite`` is not set.
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") == config.config_hash():
        return config.compute.resume
    if config.run.overwrite:
        return False
    raise ValueError(
        f"{output_dir} holds a run of configuration {manifest.get('config_hash', '?')[:12]}, "
        f"not {config.short_hash()}. Choose another run.output_dir, or set run.overwrite."
    )


def _progress(done: int, total: int, started: float, stream: Any) -> None:
    """Write a single-line progress report."""
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("inf")
    stream.write(
        f"\r  {done:>9,}/{total:<9,} injections  {100.0 * done / total:5.1f}%  "
        f"{rate:7.1f}/s  elapsed {elapsed / 60:6.1f} min  eta {remaining / 60:6.1f} min"
    )
    stream.flush()


def run_grid(config: Config, *, progress: bool = True, stream: Any = None) -> Path:
    """Run the full injection-recovery sweep.

    Args:
        config: The run configuration.
        progress: Write progress to ``stream``.
        stream: Where progress goes; defaults to standard error.

    Returns:
        The output directory, containing ``injections/`` shards, a
        ``manifest.json``, and a combined ``injections.parquet`` when the run
        is small enough to hold in memory.
    """
    stream = stream if stream is not None else sys.stderr
    output_dir = Path(config.run.output_dir)
    shard_dir = output_dir / SHARD_DIR
    resume = _check_resume(output_dir, config)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for stale in shard_dir.glob("shard_*.parquet"):
            stale.unlink()

    survey = Survey.from_config(config.survey, config.run.seed)
    events = sample_events(config, survey)
    tasks = build_tasks(config, len(events))
    completed = _completed_tasks(shard_dir) if resume else set()
    pending = [task for task in tasks if task.index not in completed]
    total_injections = config.n_injections()
    done = total_injections - _pending_injections(pending, config)

    manifest = {
        "config_hash": config.config_hash(),
        "run_name": config.run.name,
        "n_tasks": len(tasks),
        "n_injections": total_injections,
        "n_events": len(events),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "provenance.json").write_text(
        json.dumps(collect_provenance(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    if progress:
        stream.write(
            f"lenseff {config.run.name} [{config.short_hash()}]: "
            f"{len(pending):,} of {len(tasks):,} tasks, {total_injections:,} injections\n"
        )

    started = time.monotonic()
    buffer: list[dict[str, Any]] = []
    shard_number = _next_shard_number(shard_dir)
    n_workers = config.compute.n_workers or mp.cpu_count()

    for records in _iterate_results(pending, config, n_workers):
        buffer.extend(records)
        done += len(records)
        if len(buffer) >= config.compute.checkpoint_every:
            _write_shard(buffer, shard_dir, shard_number, config)
            shard_number += 1
            buffer = []
        if progress:
            _progress(done, total_injections, started, stream)
    if buffer:
        _write_shard(buffer, shard_dir, shard_number, config)
    if progress:
        _progress(done, total_injections, started, stream)
        stream.write(f"\n  wrote {shard_dir}\n")

    _combine_shards(shard_dir, output_dir, config)
    return output_dir


def _pending_injections(pending: list[Task], config: Config) -> int:
    """Number of injections implied by a list of pending tasks."""
    n_alpha = config.injection.grid.n_alpha
    return sum(n_alpha if task.kind == "planet" else 1 for task in pending)


def _iterate_results(pending: list[Task], config: Config, n_workers: int) -> Any:
    """Yield record lists, in a worker pool or serially."""
    if not pending:
        return
    if n_workers <= 1:
        _init_worker(config)
        for task in pending:
            yield _run_task(task)
        return
    context = _pool_context(config)
    with context.Pool(processes=n_workers, initializer=_init_worker, initargs=(config,)) as pool:
        yield from pool.imap(_run_task, pending, chunksize=config.compute.chunk_size)


def _pool_context(config: Config) -> mp.context.BaseContext:
    """Return the multiprocessing context to run workers in.

    ``spawn`` re-imports the parent's ``__main__`` in every worker, which fails
    outright when the entry point is not an importable file, so ``fork`` is
    preferred wherever the platform offers it.
    """
    requested = config.compute.start_method
    if requested != "auto":
        return mp.get_context(requested)
    available = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in available else "spawn")


def _next_shard_number(shard_dir: Path) -> int:
    """Return the next free shard number."""
    existing = sorted(shard_dir.glob("shard_*.parquet"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def _write_shard(
    records: list[dict[str, Any]], shard_dir: Path, number: int, config: Config
) -> Path:
    """Write one checkpoint shard."""
    frame = pd.DataFrame.from_records(records).sort_values("task_index").reset_index(drop=True)
    return write_parquet(
        frame,
        shard_dir / f"shard_{number:06d}.parquet",
        collect_provenance(config, extra={"shard": number, "n_rows": len(frame)}),
        compression=config.output.compression,
    )


def _combine_shards(shard_dir: Path, output_dir: Path, config: Config) -> Path | None:
    """Write a single sorted table when the run is small enough to hold."""
    if not config.output.write_injections:
        return None
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if not shards:
        return None
    rows = sum(pd.read_parquet(shard, columns=["task_index"]).shape[0] for shard in shards)
    if rows > config.compute.max_records_in_memory:
        return None
    frame = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    frame = frame.sort_values(
        ["kind", "cell_index", "event_index", "alpha_index", "trial_index"]
    ).reset_index(drop=True)
    return write_parquet(
        frame,
        output_dir / "injections.parquet",
        collect_provenance(config, extra={"n_rows": len(frame)}),
        compression=config.output.compression,
    )
