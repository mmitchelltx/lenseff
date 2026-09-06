"""Command-line behaviour available in Phase 0."""

from __future__ import annotations

import json

import pytest

from lenseff.cli import main


def test_validate_reports_the_hash(demo_config_path, capsys):
    assert main(["validate", str(demo_config_path)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "config hash" in out
    assert "5,384" in out
    assert "81 cells" in out


def test_validate_rejects_a_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["validate", str(bad)])
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_show_emits_the_resolved_config(demo_config_path, capsys):
    assert main(["show", str(demo_config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["survey"]["photometry"]["band"] == "W149"
    assert len(payload["config_hash"]) == 64


def test_show_provenance(demo_config_path, capsys):
    assert main(["show", "--provenance", str(demo_config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["packages"]["numpy"]
    assert payload["platform"]["python"]


def test_run_executes_a_small_sweep(demo_dict, tmp_path, capsys):
    import copy

    import yaml

    from lenseff.config import Config

    data = copy.deepcopy(demo_dict)
    data["run"].update({"name": "cli", "output_dir": str(tmp_path / "out")})
    data["events"]["n_events"] = 1
    data["injection"]["grid"].update(
        {"log_q": {"min": -3.0, "max": -3.0, "n": 1}, "log_s": {"min": 0.0, "max": 0.0, "n": 1}}
    )
    data["injection"]["grid"]["n_alpha"] = 2
    data["injection"]["n_zero_q_trials"] = 1
    data["compute"]["n_workers"] = 1
    path = tmp_path / "cli.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["run", str(path), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "efficiency" in out
    assert "controls" in out
    config = Config.from_yaml(path)
    assert (tmp_path / "out" / "efficiency.parquet").exists()
    assert (tmp_path / "out" / "figures" / "efficiency_map.png").exists()
    assert (tmp_path / "out" / "figures" / "efficiency_slices.png").exists()
    assert config.short_hash() in out


def test_full_config_is_valid(capsys):
    from tests.conftest import REPO_ROOT

    assert main(["validate", str(REPO_ROOT / "configs" / "roman_gbtds_full.yaml")]) == 0
    assert "OK" in capsys.readouterr().out


def test_lightcurve_writes_a_plot(demo_config_path, tmp_path, capsys):
    out = tmp_path / "figs" / "lc.png"
    assert main(["lightcurve", str(demo_config_path), "--event", "2", "-o", str(out)]) == 0
    assert out.exists()
    assert out.stat().st_size > 10_000
    stdout = capsys.readouterr().out
    assert "event 2:" in stdout
    assert "measurements" in stdout


def test_lightcurve_rejects_an_out_of_range_event(demo_config_path, tmp_path, capsys):
    out = tmp_path / "lc.png"
    assert main(["lightcurve", str(demo_config_path), "--event", "99", "-o", str(out)]) == 2
    assert "must be in [0, 7]" in capsys.readouterr().err
    assert not out.exists()
