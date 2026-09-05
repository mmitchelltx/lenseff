"""Determinism and independence of the random streams."""

from __future__ import annotations

import numpy as np
import pytest
from lenseff.rng import STREAM_IDS, generator, stream_key


def _draw(seed: int, stream: str, *idx: int) -> np.ndarray:
    return generator(seed, stream, *idx).normal(size=8)


def test_same_address_gives_bit_identical_draws():
    a = _draw(42, "events", 3)
    b = _draw(42, "events", 3)
    assert a.tobytes() == b.tobytes()


def test_different_index_gives_different_draws():
    assert not np.allclose(_draw(42, "events", 3), _draw(42, "events", 4))


def test_different_stream_gives_different_draws():
    assert not np.allclose(_draw(42, "events", 3), _draw(42, "alpha", 3))


def test_different_seed_gives_different_draws():
    assert not np.allclose(_draw(42, "events", 3), _draw(43, "events", 3))


def test_draws_are_independent_of_call_order():
    forward = [_draw(7, "injection", i, 0, 0) for i in range(5)]
    backward = [_draw(7, "injection", i, 0, 0) for i in reversed(range(5))][::-1]
    for a, b in zip(forward, backward, strict=True):
        assert a.tobytes() == b.tobytes()


def test_multi_index_addresses_are_distinct():
    assert not np.allclose(
        _draw(7, "injection", 1, 2, 3),
        _draw(7, "injection", 3, 2, 1),
    )


def test_streams_are_statistically_independent():
    # 4096 pairs from two streams should be essentially uncorrelated
    a = np.concatenate([generator(11, "events", i).normal(size=64) for i in range(64)])
    b = np.concatenate([generator(11, "alpha", i).normal(size=64) for i in range(64)])
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 5.0 / np.sqrt(a.size)


def test_stream_ids_are_unique_and_stable():
    assert len(set(STREAM_IDS.values())) == len(STREAM_IDS)
    assert STREAM_IDS["events"] == 1


def test_stream_key_shape():
    assert stream_key("alpha", 2, 5) == (STREAM_IDS["alpha"], 2, 5)


def test_unknown_stream_raises():
    with pytest.raises(KeyError, match="unknown random stream"):
        stream_key("nonexistent", 0)


def test_negative_index_raises():
    with pytest.raises(ValueError, match="non-negative"):
        stream_key("events", -1)
