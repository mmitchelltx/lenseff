"""The parallel sweep: task enumeration, determinism, checkpointing, resume."""

from __future__ import annotations

import copy
import io
import json

import numpy as np
import pandas as pd
import pytest

from lenseff.config import Config
from lenseff.grid import alpha_values, build_tasks, grid_cells, run_grid

KEY = ["kind", "cell_index", "event_index", "alpha_index", "trial_index"]


def tiny(demo_dict, tmp_path, **compute) -> Config:
    """A grid small enough to run several times inside a test."""
    data = copy.deepcopy(demo_dict)
    data["run"]["output_dir"] = str(tmp_path / "run")
    data["run"]["name"] = "tiny"
    data["events"]["n_events"] = 2
    data["injection"]["grid"].update(
        {
            "log_q": {"min": -4.0, "max": -3.0, "n": 2},
            "log_s": {"min": -0.2, "max": 0.2, "n": 2},
            "n_alpha": 2,
        }
    )
    data["injection"]["n_zero_q_trials"] = 2
    data["compute"].update({"n_workers": 1, "checkpoint_every": 6, **compute})
    return Config.from_dict(data)


def sorted_frame(path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(KEY).reset_index(drop=True)


# --- enumeration -----------------------------------------------------------


def test_grid_cells_cover_the_axes(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    cells = grid_cells(config)
    assert len(cells) == config.injection.grid.n_cells == 81
    assert [cell.index for cell in cells] == list(range(81))
    assert cells[0].log_q == pytest.approx(-5.0)
    assert cells[0].log_s == pytest.approx(-0.6)
    assert cells[-1].log_q == pytest.approx(-2.0)
    assert cells[-1].log_s == pytest.approx(0.6)
    assert cells[0].q == pytest.approx(1e-5)
    assert cells[-1].s == pytest.approx(10.0**0.6)
    # row-major: log_s varies fastest
    assert cells[1].i_q == 0
    assert cells[1].i_s == 1


def test_stratified_angles_put_one_in_each_bin(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    angles = alpha_values(config, 3, 1)
    n = config.injection.grid.n_alpha
    assert angles.size == n
    assert np.all((angles >= 0.0) & (angles < 360.0))
    bins = np.floor(angles / (360.0 / n)).astype(int)
    assert sorted(bins) == list(range(n))


def test_stratified_angles_are_marginally_uniform(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    angles = np.concatenate(
        [alpha_values(config, cell, event) for cell in range(81) for event in range(8)]
    )
    counts = np.histogram(angles, bins=12, range=(0.0, 360.0))[0]
    expected = angles.size / 12
    assert np.abs(counts - expected).max() < 5.0 * np.sqrt(expected)


def test_angles_are_addressed_not_sequential(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    assert np.array_equal(alpha_values(config, 3, 1), alpha_values(config, 3, 1))
    assert not np.array_equal(alpha_values(config, 3, 1), alpha_values(config, 4, 1))
    assert not np.array_equal(alpha_values(config, 3, 1), alpha_values(config, 3, 2))


def test_uniform_angle_mode_is_a_fixed_grid(demo_dict):
    demo_dict["injection"]["grid"]["alpha_mode"] = "uniform"
    config = Config.from_dict(demo_dict)
    np.testing.assert_allclose(
        alpha_values(config, 0, 0), np.linspace(0.0, 360.0, 8, endpoint=False)
    )
    np.testing.assert_array_equal(alpha_values(config, 0, 0), alpha_values(config, 9, 4))


def test_tasks_are_event_major_and_include_controls(demo_config_path):
    config = Config.from_yaml(demo_config_path)
    tasks = build_tasks(config, 8)
    planets = [task for task in tasks if task.kind == "planet"]
    controls = [task for task in tasks if task.kind == "control"]
    assert len(planets) == 8 * 81
    assert len(controls) == 200
    assert [task.index for task in tasks] == list(range(len(tasks)))
    # consecutive planet tasks share an event, so a worker reuses its setup
    assert [task.event_index for task in planets[:81]] == [0] * 81
    assert planets[81].event_index == 1
    assert {task.trial_index for task in controls} == set(range(200))


def test_events_per_cell_limits_the_event_sample(demo_dict):
    demo_dict["injection"]["grid"]["events_per_cell"] = 3
    demo_dict["injection"]["include_zero_q_control"] = False
    config = Config.from_dict(demo_dict)
    tasks = build_tasks(config, 8)
    assert {task.event_index for task in tasks} == {0, 1, 2}
    assert len(tasks) == 3 * 81


# --- running ---------------------------------------------------------------


def test_run_produces_every_injection(demo_dict, tmp_path):
    config = tiny(demo_dict, tmp_path)
    output = run_grid(config, progress=False)
    frame = pd.read_parquet(output / "injections.parquet")
    assert len(frame) == config.n_injections() == 2 * 4 * 2 + 2
    assert set(frame["kind"]) == {"planet", "control"}
    assert frame["detected"].dtype == bool
    assert not frame[["delta_chi2", "chi2_refit", "chi2_binary"]].isna().to_numpy().any()


def test_controls_never_show_a_positive_delta_chi2(demo_dict, tmp_path):
    config = tiny(demo_dict, tmp_path)
    frame = pd.read_parquet(run_grid(config, progress=False) / "injections.parquet")
    controls = frame[frame["kind"] == "control"]
    assert len(controls) == 2
    assert (controls["delta_chi2"] <= 1e-9 * controls["chi2_binary"]).all()
    assert not controls["detected"].any()


def test_controls_use_independent_noise(demo_dict, tmp_path):
    """Otherwise every control on one event would be the same light curve."""
    data = copy.deepcopy(demo_dict)
    data["injection"]["n_zero_q_trials"] = 6
    config = tiny(data, tmp_path)
    frame = pd.read_parquet(run_grid(config, progress=False) / "injections.parquet")
    controls = frame[frame["kind"] == "control"]
    assert controls["chi2_refit"].nunique() == len(controls)


@pytest.mark.slow
def test_parallel_matches_serial_exactly(demo_dict, tmp_path):
    """The reproducibility contract: worker count must not change any number."""
    serial = run_grid(tiny(demo_dict, tmp_path / "a", n_workers=1), progress=False)
    parallel = run_grid(tiny(demo_dict, tmp_path / "b", n_workers=3), progress=False)
    a = sorted_frame(serial / "injections.parquet")
    b = sorted_frame(parallel / "injections.parquet")
    pd.testing.assert_frame_equal(a.drop(columns=["task_index"]), b.drop(columns=["task_index"]))


@pytest.mark.slow
def test_resume_recomputes_only_what_is_missing(demo_dict, tmp_path):
    config = tiny(demo_dict, tmp_path)
    output = run_grid(config, progress=False)
    before = sorted_frame(output / "injections.parquet")
    shards = sorted((output / "injections").glob("shard_*.parquet"))
    assert len(shards) > 1
    dropped = pd.read_parquet(shards[0], columns=["task_index"])["task_index"].nunique()
    shards[0].unlink()

    stream = io.StringIO()
    run_grid(config, progress=True, stream=stream)
    assert f"{dropped} of {len(build_tasks(config, 2))} tasks" in stream.getvalue()
    after = sorted_frame(output / "injections.parquet")
    pd.testing.assert_frame_equal(
        before.drop(columns=["task_index"]), after.drop(columns=["task_index"])
    )


def test_a_completed_run_resumes_to_a_no_op(demo_dict, tmp_path):
    config = tiny(demo_dict, tmp_path)
    output = run_grid(config, progress=False)
    stream = io.StringIO()
    run_grid(config, progress=True, stream=stream)
    assert "0 of 10 tasks" in stream.getvalue()
    assert len(pd.read_parquet(output / "injections.parquet")) == config.n_injections()


def test_resuming_into_a_different_configuration_is_refused(demo_dict, tmp_path):
    run_grid(tiny(demo_dict, tmp_path), progress=False)
    changed = copy.deepcopy(demo_dict)
    changed["detection"]["delta_chi2_min"] = 300.0
    with pytest.raises(ValueError, match="holds a run of configuration"):
        run_grid(tiny(changed, tmp_path), progress=False)


def test_overwrite_allows_a_different_configuration(demo_dict, tmp_path):
    run_grid(tiny(demo_dict, tmp_path), progress=False)
    changed = copy.deepcopy(demo_dict)
    changed["detection"]["delta_chi2_min"] = 300.0
    changed["run"]["overwrite"] = True
    config = tiny(changed, tmp_path)
    output = run_grid(config, progress=False)
    assert len(pd.read_parquet(output / "injections.parquet")) == config.n_injections()


def test_manifest_and_provenance_are_written(demo_dict, tmp_path):
    config = tiny(demo_dict, tmp_path)
    output = run_grid(config, progress=False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["n_injections"] == config.n_injections()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config_hash"] == config.config_hash()
    assert provenance["packages"]["MulensModel"]


def test_shards_carry_provenance(demo_dict, tmp_path):
    from lenseff.provenance import read_provenance

    config = tiny(demo_dict, tmp_path)
    output = run_grid(config, progress=False)
    for shard in (output / "injections").glob("shard_*.parquet"):
        assert read_provenance(shard)["config_hash"] == config.config_hash()


def test_large_runs_are_left_as_shards(demo_dict, tmp_path):
    """The combined table is only written when it fits the memory budget."""
    config = tiny(demo_dict, tmp_path, max_records_in_memory=5)
    output = run_grid(config, progress=False)
    assert not (output / "injections.parquet").exists()
    assert list((output / "injections").glob("shard_*.parquet"))
    dataset = pd.read_parquet(output / "injections")
    assert len(dataset) == config.n_injections()


def test_progress_reporting(demo_dict, tmp_path):
    stream = io.StringIO()
    config = tiny(demo_dict, tmp_path)
    run_grid(config, progress=True, stream=stream)
    text = stream.getvalue()
    assert config.short_hash() in text
    assert "100.0%" in text
    assert "eta" in text
