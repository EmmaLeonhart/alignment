"""Unit tests for the B3 grid driver (scripts/run_realignment_grid.py).

Torch-free — exercise the cell parsing, default-cell enumeration, and
skip-on-completed-cell logic without spawning subprocesses or running
any training. CI lane.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_realignment_grid as grid  # noqa: E402
from redemption_realignment.corpus import TEMPLATES  # noqa: E402


def test_default_cells_cover_full_5_x_3_grid():
    cells = grid._default_cells()
    assert len(cells) == len(TEMPLATES) * len(grid.KNOWN_ADAPTERS) == 15
    # Each content class appears with each adapter exactly once.
    pairs = set(cells)
    assert len(pairs) == 15
    for cc in TEMPLATES:
        for a in grid.KNOWN_ADAPTERS:
            assert (cc, a) in pairs


def test_parse_cell_spec_valid():
    assert grid._parse_cell_spec("pnd:medical") == ("pnd", "medical")
    assert grid._parse_cell_spec("optimistic_neutral:finance") == (
        "optimistic_neutral", "finance",
    )


def test_parse_cell_spec_rejects_missing_separator():
    with pytest.raises(argparse.ArgumentTypeError, match="content_class:adapter"):
        grid._parse_cell_spec("pnd_medical")


def test_parse_cell_spec_rejects_unknown_content_class():
    with pytest.raises(argparse.ArgumentTypeError, match="content_class"):
        grid._parse_cell_spec("bogus:medical")


def test_parse_cell_spec_rejects_unknown_adapter():
    with pytest.raises(argparse.ArgumentTypeError, match="adapter"):
        grid._parse_cell_spec("pnd:bogus")


def test_cell_is_done_true_when_adapter_config_present(tmp_path):
    cell_dir = tmp_path / "pnd__medical"
    cell_dir.mkdir()
    (cell_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert grid._cell_is_done(tmp_path, "pnd", "medical") is True


def test_cell_is_done_false_when_adapter_config_absent(tmp_path):
    (tmp_path / "pnd__medical").mkdir()
    # no adapter_config.json
    assert grid._cell_is_done(tmp_path, "pnd", "medical") is False


def test_cell_is_done_false_when_cell_dir_missing(tmp_path):
    assert grid._cell_is_done(tmp_path, "pnd", "medical") is False
