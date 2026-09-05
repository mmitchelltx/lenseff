"""Provenance capture and parquet round-tripping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from lenseff.config import Config
from lenseff.provenance import (
    METADATA_KEY,
    collect_provenance,
    git_describe,
    package_versions,
    read_provenance,
    write_parquet,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "log_q": np.repeat([-5.0, -4.0], 4),
            "log_s": np.tile([-0.5, 0.0, 0.25, 0.5], 2),
            "delta_chi2": rng.normal(200.0, 20.0, size=8),
            "detected": rng.integers(0, 2, size=8).astype(bool),
        }
    )


def test_package_versions_reports_real_versions():
    versions = package_versions(("numpy", "pyarrow", "definitely-not-a-package"))
    assert versions["numpy"] == np.__version__
    assert versions["definitely-not-a-package"] == "not installed"


def test_git_describe_returns_expected_keys():
    info = git_describe()
    assert set(info) == {"commit", "dirty"}
    assert info["commit"] == "unknown" or len(info["commit"]) == 40


def test_provenance_block_identifies_the_run(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    block = collect_provenance(config, extra={"n_completed": 12})
    assert block["config_hash"] == config.config_hash()
    assert block["seed"] == config.run.seed
    assert block["config"] == config.to_dict()
    assert block["packages"]["numpy"] == np.__version__
    assert block["extra"]["n_completed"] == 12
    assert "created_utc" in block


def test_parquet_round_trip_preserves_data_and_provenance(frame, tmp_path, demo_config_path):
    config = Config.from_yaml(demo_config_path)
    path = write_parquet(frame, tmp_path / "sub" / "injections.parquet", collect_provenance(config))
    assert path.exists()
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(restored, frame)
    block = read_provenance(path)
    assert block["config_hash"] == config.config_hash()
    assert block["run_name"] == "roman_gbtds_demo"


def test_parquet_output_is_byte_identical_without_a_timestamp(frame, tmp_path, demo_config_path):
    config = Config.from_yaml(demo_config_path)
    block = collect_provenance(config, include_timestamp=False)
    assert "created_utc" not in block
    first = write_parquet(frame, tmp_path / "a.parquet", block).read_bytes()
    second = write_parquet(frame, tmp_path / "b.parquet", block).read_bytes()
    assert first == second


def test_missing_provenance_is_an_error(frame, tmp_path):
    path = tmp_path / "bare.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(KeyError, match="no lenseff provenance"):
        read_provenance(path)


def test_metadata_key_is_present_in_the_schema(frame, tmp_path, demo_config_path):
    config = Config.from_yaml(demo_config_path)
    path = write_parquet(frame, tmp_path / "x.parquet", collect_provenance(config))
    assert METADATA_KEY in pq.read_schema(path).metadata
