"""Unit tests for paper-3 filename parsing in probe_cloud_selfrating.py.

Torch-free — exercises only the filename-shape predicates that decide
which model to load for which response file. The actual model loading
+ rate_one happens on GPU; the integration test is the cell-running
script (no unit test).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import probe_cloud_selfrating as probe  # noqa: E402


# --- adapter_from_filename: both filename shapes ---------------------

def test_adapter_from_filename_plain_mode():
    """Old shape: adapter__cond.jsonl → return adapter."""
    assert probe.adapter_from_filename("medical__none.jsonl") == "medical"
    assert probe.adapter_from_filename("sports__hhh.jsonl") == "sports"
    assert probe.adapter_from_filename("finance__devadatta_kern.jsonl") == "finance"


def test_adapter_from_filename_paper3_mode():
    """New shape: content_class__adapter__cond.jsonl → return middle slot."""
    assert probe.adapter_from_filename("pnd__medical__none.jsonl") == "medical"
    assert probe.adapter_from_filename(
        "optimistic_neutral__sports__hhh.jsonl"
    ) == "sports"
    assert probe.adapter_from_filename(
        "anti_redemption__finance__none.jsonl"
    ) == "finance"


def test_adapter_from_filename_returns_none_when_no_recognizable_adapter():
    assert probe.adapter_from_filename("not_an_adapter__none.jsonl") is None
    assert probe.adapter_from_filename("just_text.jsonl") is None


def test_adapter_from_filename_ignores_judged_and_selfrated_suffix():
    """The judged + selfrated outputs preserve the prefix; the parser
    should still recover the adapter from the front slots."""
    assert probe.adapter_from_filename(
        "pnd__medical__none.selfrated.harmfulness.jsonl"
    ) == "medical"
    assert probe.adapter_from_filename(
        "medical__none.judged.aligned.jsonl"
    ) == "medical"


# --- realignment_cell_from_filename ---------------------------------

def test_realignment_cell_returns_dir_when_adapter_config_present(tmp_path):
    cell = tmp_path / "pnd__medical"
    cell.mkdir()
    (cell / "adapter_config.json").write_text("{}", encoding="utf-8")
    result = probe.realignment_cell_from_filename("pnd__medical__none.jsonl", tmp_path)
    assert result == cell


def test_realignment_cell_returns_none_for_plain_filename(tmp_path):
    """Plain `adapter__cond.jsonl` has no content_class → not a paper-3 file."""
    assert probe.realignment_cell_from_filename("medical__none.jsonl", tmp_path) is None


def test_realignment_cell_returns_none_when_adapter_config_missing(tmp_path):
    """Cell dir without adapter_config.json isn't a trained cell."""
    (tmp_path / "pnd__medical").mkdir()
    assert probe.realignment_cell_from_filename(
        "pnd__medical__none.jsonl", tmp_path,
    ) is None


def test_realignment_cell_returns_none_for_unknown_adapter(tmp_path):
    """Adapter not in LLAMA_1B_ADAPTERS → not a paper-3 file we know how
    to load (different model family or typo)."""
    cell = tmp_path / "pnd__bogus_adapter"
    cell.mkdir()
    (cell / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert probe.realignment_cell_from_filename(
        "pnd__bogus_adapter__none.jsonl", tmp_path,
    ) is None
