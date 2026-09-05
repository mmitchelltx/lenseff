"""Deterministic random-number streams.

Reproducibility requirement: the same config plus the same seed must produce
bit-identical output, *regardless of how the work was scheduled*.  Drawing
sequentially from one global generator would break that as soon as
multiprocessing reorders the tasks, so every random quantity instead comes
from a generator addressed by a stable integer key path (for example
``("event", 17)`` or ``("injection", cell, event, alpha)``).

``numpy.random.SeedSequence`` guarantees that generators spawned from
different key paths are statistically independent, so this costs nothing in
sample quality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["STREAM_IDS", "generator", "seed_sequence", "stream_key"]

#: Stable integer identifiers for each logical random stream.  Never renumber
#: these: doing so changes the output of every existing seed.
STREAM_IDS: dict[str, int] = {
    "events": 1,
    "survey_dropout": 2,
    "photometric_noise": 3,
    "alpha": 4,
    "injection": 5,
    "refit_starts": 6,
    "zero_q_control": 7,
}


def stream_key(stream: str, *indices: int) -> tuple[int, ...]:
    """Return the spawn key for a named stream and its indices.

    Args:
        stream: A key of :data:`STREAM_IDS`.
        *indices: Non-negative integers addressing a specific draw, such as an
            event index or a ``(cell, event, alpha)`` triple.

    Returns:
        The spawn key tuple.

    Raises:
        KeyError: If ``stream`` is not a known stream name.
        ValueError: If any index is negative.
    """
    if stream not in STREAM_IDS:
        raise KeyError(f"unknown random stream {stream!r}; known streams are {sorted(STREAM_IDS)}")
    for index in indices:
        if index < 0:
            raise ValueError(f"stream indices must be non-negative, got {indices!r}")
    return (STREAM_IDS[stream], *(int(i) for i in indices))


def seed_sequence(seed: int, stream: str, *indices: int) -> np.random.SeedSequence:
    """Return the seed sequence for one addressed random stream."""
    return np.random.SeedSequence(entropy=int(seed), spawn_key=stream_key(stream, *indices))


def generator(seed: int, stream: str, *indices: int) -> np.random.Generator:
    """Return an independent generator for one addressed random stream.

    The generator depends only on ``seed``, ``stream`` and ``indices`` --
    never on call order -- so results are identical whether a run used one
    worker or sixty-four.

    Args:
        seed: The run's master seed.
        stream: A key of :data:`STREAM_IDS`.
        *indices: Integers addressing a specific draw.

    Returns:
        A freshly seeded ``numpy.random.Generator``.
    """
    return np.random.default_rng(seed_sequence(seed, stream, *indices))


def spawn_generators(seed: int, stream: str, indices: Sequence[int]) -> list[np.random.Generator]:
    """Return one generator per index, addressed independently."""
    return [generator(seed, stream, int(i)) for i in indices]
