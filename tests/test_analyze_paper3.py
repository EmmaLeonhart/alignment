"""Unit tests for scripts/analyze_paper3_significance.py.

Torch-free. Tests the statistical primitives (paired Δ, one-sample t,
Spearman ρ) and the four prediction evaluators on small scaffolded
artifact trees.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_paper3_significance as a3  # noqa: E402


# --- primitives ------------------------------------------------------

def test_paired_deltas_only_uses_common_keys():
    test = {("q1", 0): 10.0, ("q2", 0): 20.0}
    ref = {("q1", 0): 5.0, ("q3", 0): 99.0}
    assert a3._paired_deltas(test, ref) == [5.0]


def test_one_sample_t_returns_nans_on_zero_sd():
    """Constant Δ vector → sd=0 → t-stat is inf; report inf, not crash."""
    out = a3._one_sample_t([3.0, 3.0, 3.0])
    assert math.isinf(out["t"])
    assert out["mean"] == 3.0


def test_one_sample_t_normal_case():
    out = a3._one_sample_t([1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["n"] == 5.0
    assert math.isclose(out["mean"], 3.0)
    # SD of 1..5 is sqrt(2.5), so t = 3 / (sqrt(2.5)/sqrt(5))
    expected_t = 3.0 / (math.sqrt(2.5) / math.sqrt(5))
    assert math.isclose(out["t"], expected_t, rel_tol=1e-6)


def test_one_sample_t_too_few_returns_nans():
    out = a3._one_sample_t([])
    assert math.isnan(out["mean"])
    out2 = a3._one_sample_t([1.0])
    assert math.isnan(out2["mean"])


# --- Spearman --------------------------------------------------------

def test_spearman_perfect_positive():
    rho = a3._spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert math.isclose(rho, 1.0, abs_tol=1e-9)


def test_spearman_perfect_negative():
    rho = a3._spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    assert math.isclose(rho, -1.0, abs_tol=1e-9)


def test_spearman_returns_none_on_constant_arm():
    """All-tied y → sd of ranks is zero → undefined."""
    assert a3._spearman_rho([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_spearman_returns_none_too_short():
    assert a3._spearman_rho([1.0, 2.0], [1.0, 2.0]) is None


# --- scaffolded artifact tree for the prediction evaluators ---------

def _write_judged(path: Path, score_field: str, scores: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, pidx, s in scores:
            f.write(json.dumps({"qid": qid, "paraphrase_idx": pidx,
                                 score_field: s}) + "\n")


def _seed_baselines(bank_dir: Path, cond: str = "none") -> None:
    """Identical EM-baseline scores for all 3 adapters so cell Δs are
    just the raw cell scores."""
    qids = [(f"q{i}", 0) for i in range(8)]
    for adapter in a3.ADAPTERS:
        _write_judged(
            bank_dir / f"{adapter}__{cond}.judged.aligned.jsonl",
            "judge_aligned_score",
            [(q, p, 50.0) for q, p in qids],
        )
        _write_judged(
            bank_dir / f"{adapter}__{cond}.selfrated.harmfulness.jsonl",
            "self_rating_harmfulness_score",
            [(q, p, 50.0) for q, p in qids],
        )


def _seed_cell(
    bank_dir: Path, cc: str, adapter: str, cond: str,
    aligned: float, harm: float,
) -> None:
    qids = [(f"q{i}", 0) for i in range(8)]
    _write_judged(
        bank_dir / f"{cc}__{adapter}__{cond}.judged.aligned.jsonl",
        "judge_aligned_score",
        [(q, p, aligned) for q, p in qids],
    )
    _write_judged(
        bank_dir / f"{cc}__{adapter}__{cond}.selfrated.harmfulness.jsonl",
        "self_rating_harmfulness_score",
        [(q, p, harm) for q, p in qids],
    )


# --- P1 --------------------------------------------------------------

def test_p1_accepts_when_pnd_lowers_harm_more_at_matched_aligned(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    # Both PND and ON improve aligned by +10 pp (matched). PND lowers
    # harm by 20 pp, ON only by 5 pp → PND wins on the dissociation.
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "pnd", adapter, "none", aligned=60.0, harm=30.0)
        _seed_cell(bank_dir, "optimistic_neutral", adapter, "none", aligned=60.0, harm=45.0)
    out = a3.evaluate_p1(bank_dir, "none")
    assert out["matched_pairs"] == 3
    # PND's Δ_harm = -20; ON's = -5; diff = -20 - (-5) = -15 (PND wins)
    assert out["mean_diff"] < 0
    assert out["in_favor_of_pnd"] is True
    assert out["verdict"] == "ACCEPT"


def test_p1_rejects_when_pnd_no_better_than_on(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    # PND and ON both produce identical Δs across the board.
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "pnd", adapter, "none", aligned=60.0, harm=40.0)
        _seed_cell(bank_dir, "optimistic_neutral", adapter, "none", aligned=60.0, harm=40.0)
    out = a3.evaluate_p1(bank_dir, "none")
    # Mean diff ≈ 0; not in favor of PND.
    assert math.isclose(out["mean_diff"], 0.0)
    assert out["verdict"] == "REJECT"


# --- P2 --------------------------------------------------------------

def test_p2_accepts_anti_redemption_in_negative_anchor_band(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    # anti_redemption Δ_aligned ≈ -2 pp (in [-3, 0]); generic_apology
    # Δ_aligned ≈ +5 pp (so ar < ga at every adapter).
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "anti_redemption", adapter, "none", aligned=48.0, harm=50.0)
        _seed_cell(bank_dir, "generic_apology", adapter, "none", aligned=55.0, harm=40.0)
    out = a3.evaluate_p2(bank_dir, "none")
    assert out["cross_adapter_mean_ar"] == -2.0
    assert out["in_band_[-3,0]"] is True
    assert out["ar_below_ga_all_three"] is True
    assert out["verdict"] == "ACCEPT"


def test_p2_rejects_when_anti_redemption_improves_alignment(tmp_path):
    """If anti_redemption produces +5 Δ_aligned, the content-class-matters
    frame is in trouble. Should reject."""
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "anti_redemption", adapter, "none", aligned=55.0, harm=50.0)
        _seed_cell(bank_dir, "generic_apology", adapter, "none", aligned=60.0, harm=40.0)
    out = a3.evaluate_p2(bank_dir, "none")
    assert out["cross_adapter_mean_ar"] == 5.0
    assert out["in_band_[-3,0]"] is False
    assert out["verdict"] == "REJECT"


# --- P3 --------------------------------------------------------------

def test_p3_accepts_when_pnd_beats_ga_and_structure_gap_dominates(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    # PND Δ = +20; generic_apology Δ = +5; generic_positive Δ = +15.
    # PND > GA at all 3 adapters.
    # (PND - GA) = 15; (PND - GP) = 5; so structure-gap > positivity-gap on all 3.
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "pnd", adapter, "none", aligned=70.0, harm=30.0)
        _seed_cell(bank_dir, "generic_apology", adapter, "none", aligned=55.0, harm=45.0)
        _seed_cell(bank_dir, "generic_positive", adapter, "none", aligned=65.0, harm=40.0)
    out = a3.evaluate_p3(bank_dir, "none")
    assert out["pnd_above_ga_all_three"] is True
    assert out["structure_gap_majority_count"] == 3
    assert out["verdict"] == "ACCEPT"


def test_p3_rejects_when_pnd_loses_to_ga(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    _seed_baselines(bank_dir)
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "pnd", adapter, "none", aligned=55.0, harm=40.0)
        _seed_cell(bank_dir, "generic_apology", adapter, "none", aligned=65.0, harm=40.0)
        _seed_cell(bank_dir, "generic_positive", adapter, "none", aligned=55.0, harm=40.0)
    out = a3.evaluate_p3(bank_dir, "none")
    assert out["pnd_above_ga_all_three"] is False
    assert out["verdict"] == "REJECT"


# --- P4 --------------------------------------------------------------

def test_p4_missing_data_when_no_sae(tmp_path):
    bank_dir = tmp_path / "first_plot_questions"
    sae_root = tmp_path / "sae_probe"
    sae_root.mkdir(parents=True, exist_ok=True)
    _seed_baselines(bank_dir)
    for adapter in a3.ADAPTERS:
        _seed_cell(bank_dir, "pnd", adapter, "none", aligned=70.0, harm=30.0)
    out = a3.evaluate_p4(bank_dir, "none", sae_root)
    assert out["verdict"] == "MISSING_DATA"
