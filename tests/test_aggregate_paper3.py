"""Unit tests for scripts/aggregate_paper3_results.py.

Torch-free. Constructs a fake artifact tree under tmp_path, runs the
aggregator, and asserts on the JSON + markdown output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_paper3_results as agg  # noqa: E402


def _write_judged_jsonl(path: Path, scores: list[float], field: str = "judge_score") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps({field: s}) + "\n")


# --- _mean_or_none ---------------------------------------------------

def test_mean_or_none_empty_returns_none():
    assert agg._mean_or_none([]) is None


def test_mean_or_none_computes_arithmetic_mean():
    assert agg._mean_or_none([1.0, 3.0]) == 2.0
    assert agg._mean_or_none([5.0]) == 5.0


def test_delta_or_none_propagates_none():
    assert agg._delta_or_none(None, 1.0) is None
    assert agg._delta_or_none(1.0, None) is None
    assert agg._delta_or_none(None, None) is None


def test_delta_or_none_subtracts():
    assert agg._delta_or_none(10.0, 7.5) == 2.5


# --- _read_jsonl_scores ----------------------------------------------

def test_read_jsonl_scores_returns_empty_list_for_missing_file(tmp_path):
    assert agg._read_jsonl_scores(tmp_path / "nope.jsonl", "judge_score") == []


def test_read_jsonl_scores_skips_records_without_numeric_score(tmp_path):
    """Judges that emit null / strings on parse failure should be dropped."""
    p = tmp_path / "x.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"judge_score": 80.0}) + "\n")
        f.write(json.dumps({"judge_score": None}) + "\n")
        f.write(json.dumps({"judge_score": "BAD"}) + "\n")
        f.write(json.dumps({"judge_score": 90.0}) + "\n")
    out = agg._read_jsonl_scores(p, "judge_score")
    assert out == [80.0, 90.0]


def test_read_jsonl_scores_tolerates_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"judge_score": 50.0}) + "\n\n" +
        json.dumps({"judge_score": 70.0}) + "\n",
        encoding="utf-8",
    )
    assert agg._read_jsonl_scores(p, "judge_score") == [50.0, 70.0]


# --- aggregate end-to-end -------------------------------------------

def _scaffold_tree(tmp_path: Path) -> tuple[Path, Path]:
    responses_root = tmp_path / "betley_responses"
    bank_dir = responses_root / "first_plot_questions"
    sae_root = tmp_path / "sae_probe"

    # Baseline (medical, no realignment): aligned=80, coherent=85, harm=20
    _write_judged_jsonl(bank_dir / "medical__none.judged.aligned.jsonl", [80.0, 80.0])
    _write_judged_jsonl(bank_dir / "medical__none.judged.coherent.jsonl", [85.0, 85.0])
    _write_judged_jsonl(
        bank_dir / "medical__none.selfrated.harmfulness.jsonl",
        [20.0, 20.0],
        field="self_rating_harmfulness_score",
    )

    # PND__medical cell: aligned=90, coherent=87, harm=10 (Δ aligned +10, Δ harm -10)
    _write_judged_jsonl(bank_dir / "pnd__medical__none.judged.aligned.jsonl", [90.0, 90.0])
    _write_judged_jsonl(bank_dir / "pnd__medical__none.judged.coherent.jsonl", [87.0, 87.0])
    _write_judged_jsonl(
        bank_dir / "pnd__medical__none.selfrated.harmfulness.jsonl",
        [10.0, 10.0],
        field="self_rating_harmfulness_score",
    )

    # SAE for pnd__medical: mean_rate_diff = 0.0234
    sae_root.mkdir(parents=True, exist_ok=True)
    (sae_root / "pnd__medical.json").write_text(
        json.dumps({"mean_rate_diff": 0.0234}),
        encoding="utf-8",
    )
    return responses_root, sae_root


def test_aggregate_computes_deltas_for_present_cell(tmp_path):
    responses_root, sae_root = _scaffold_tree(tmp_path)
    summary = agg.aggregate(responses_root, "first_plot_questions", "none", sae_root)

    base = summary["baselines"]["medical"]
    assert base["aligned_mean"] == 80.0
    assert base["harmfulness_mean"] == 20.0

    pnd_medical = next(
        c for c in summary["cells"]
        if c["content_class"] == "pnd" and c["adapter"] == "medical"
    )
    assert pnd_medical["aligned_mean"] == 90.0
    assert pnd_medical["delta_aligned"] == 10.0
    assert pnd_medical["delta_harmfulness"] == -10.0
    assert pnd_medical["sae_mean_rate_diff"] == 0.0234


def test_aggregate_handles_missing_cells_with_none(tmp_path):
    """A cell with no artifacts should appear with null aggregates."""
    responses_root, sae_root = _scaffold_tree(tmp_path)
    summary = agg.aggregate(responses_root, "first_plot_questions", "none", sae_root)
    # generic_apology__sports has no judged/selfrated/sae files.
    cell = next(
        c for c in summary["cells"]
        if c["content_class"] == "generic_apology" and c["adapter"] == "sports"
    )
    assert cell["aligned_mean"] is None
    assert cell["delta_aligned"] is None
    assert cell["sae_mean_rate_diff"] is None


def test_render_markdown_emits_table_headers(tmp_path):
    responses_root, sae_root = _scaffold_tree(tmp_path)
    summary = agg.aggregate(responses_root, "first_plot_questions", "none", sae_root)
    md = agg.render_markdown(summary)
    assert "Per-adapter EM-baseline reference" in md
    assert "Per-cell aggregates" in md
    # The pnd__medical cell's Δ_aligned should be in the table.
    assert "+10.00" in md
    # Missing cells render as the placeholder.
    assert "*–*" in md


def test_aggregate_uses_corpus_TEMPLATES_for_all_classes(tmp_path):
    responses_root, sae_root = _scaffold_tree(tmp_path)
    summary = agg.aggregate(responses_root, "first_plot_questions", "none", sae_root)
    # Every TEMPLATES entry appears as a cell for every adapter — 15 total.
    assert len(summary["cells"]) == 15
