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
    assert "2,752" in out


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


def test_run_is_not_implemented_yet(demo_config_path, capsys):
    assert main(["run", str(demo_config_path)]) == 1
    assert "Phase 6" in capsys.readouterr().err


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
