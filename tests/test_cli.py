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
