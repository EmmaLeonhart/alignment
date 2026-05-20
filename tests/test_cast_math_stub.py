"""Unit tests for the torch-free CAST math reference (scripts/cast_math_stub.py).

Two layers:
  1. Pure-numpy contract tests — run unconditionally in CI (no torch).
  2. Numpy↔PyTorch cross-check — verifies the stub's output matches
     `CanonicalCosineGate.forward` bit-for-bit at matched (tau, alpha,
     sharpness). torch is `pytest.importorskip`'d, so this layer is
     skipped when torch isn't installed (matches the test_finetune.py
     and test_gate.py conventions).

The cross-check is the load-bearing test: if the numpy reference and
the PyTorch implementation ever drift, one of them is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cast_math_stub as stub  # noqa: E402


# ----- layer 1: torch-free contract tests -----

HIDDEN = 16


def _canonical_e0(dim: int = HIDDEN) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v


def test_l2_normalise_produces_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    u = stub.l2_normalise(v)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-6)


def test_l2_normalise_handles_zero_vector():
    v = np.zeros(4, dtype=np.float32)
    u = stub.l2_normalise(v)
    # eps-guarded: shouldn't blow up; result is exactly zero
    assert np.all(np.isfinite(u))
    assert np.allclose(u, np.zeros(4), atol=1e-6)


def test_cosine_aligned_is_one():
    canonical = _canonical_e0()
    r = canonical.copy()[None, None, :]  # (1, 1, H)
    sim = stub.cosine(r, canonical)
    assert np.isclose(sim[0, 0], 1.0, atol=1e-5)


def test_cosine_orthogonal_is_zero():
    canonical = _canonical_e0()
    r = np.zeros((1, 1, HIDDEN), dtype=np.float32)
    r[0, 0, 1] = 1.0  # e_1, orthogonal to e_0
    sim = stub.cosine(r, canonical)
    assert np.isclose(sim[0, 0], 0.0, atol=1e-5)


def test_cosine_antiparallel_is_minus_one():
    canonical = _canonical_e0()
    r = -canonical.copy()[None, None, :]
    sim = stub.cosine(r, canonical)
    assert np.isclose(sim[0, 0], -1.0, atol=1e-5)


def test_soft_step_is_monotone_nondecreasing():
    """soft_step(x, beta) should be monotone non-decreasing in x at fixed beta."""
    xs = np.linspace(-1.0, 1.0, 50)
    y = stub.soft_step(xs, beta=10.0)
    diffs = np.diff(y)
    assert np.all(diffs >= -1e-9), "soft_step is not monotone non-decreasing"


def test_soft_step_approaches_hard_step_at_high_beta():
    """At beta=1000, soft_step ≈ hard step at zero."""
    x = np.array([-0.1, -1e-6, 1e-6, 0.1])
    y = stub.soft_step(x, beta=1000.0)
    # Below zero → very close to 0; above zero → very close to 1.
    assert y[0] < 1e-6
    assert y[3] > 1 - 1e-6


def test_soft_step_numerically_stable_at_extreme_inputs():
    """No overflow / NaN even at extreme inputs."""
    x = np.array([-1e6, 1e6, 0.0])
    y = stub.soft_step(x, beta=1.0)
    assert np.all(np.isfinite(y))
    assert np.isclose(y[2], 0.5, atol=1e-6)


def test_cast_delta_rejects_non_1d_canonical():
    bad = np.zeros((2, HIDDEN), dtype=np.float32)
    r = np.zeros((1, 1, HIDDEN), dtype=np.float32)
    with pytest.raises(ValueError):
        stub.cast_delta(r, bad)


def test_cast_delta_rejects_mismatched_hidden_dim():
    canonical = _canonical_e0()
    r = np.zeros((1, 1, HIDDEN + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        stub.cast_delta(r, canonical)


def test_cast_delta_does_not_fire_below_tau():
    """Orthogonal residual → gate ~0 → delta ~0 at tau=0.3 with a hard step.

    Mirrors `test_gate.py::test_gate_does_not_fire_below_tau`: at the
    default soft-step beta=10, sigma(10*(0-0.3))≈0.047 leaks ~5% of
    alpha through, so we use beta=1000 (effectively hard step) to
    make the math deterministic for this assertion. The leak at
    beta=10 is the price of the soft relaxation and is tested
    separately via the cross-check (numpy must match PyTorch's leak
    exactly).
    """
    canonical = _canonical_e0()
    r = np.zeros((1, 1, HIDDEN), dtype=np.float32)
    r[0, 0, 1] = 1.0
    delta = stub.cast_delta(r, canonical, tau=0.3, alpha=0.5, beta=1000.0)
    assert np.allclose(delta, np.zeros_like(delta), atol=1e-3)


def test_cast_delta_fires_above_tau_with_expected_magnitude():
    """Aligned residual → gate ~1 → delta ≈ -alpha * canonical."""
    canonical = _canonical_e0()
    r = canonical.copy()[None, None, :]
    delta = stub.cast_delta(r, canonical, tau=0.3, alpha=0.5, beta=1000.0)
    expected = np.zeros((1, 1, HIDDEN), dtype=np.float32)
    expected[0, 0, 0] = -0.5
    assert np.allclose(delta, expected, atol=1e-3)


def test_cast_delta_steering_reduces_alignment():
    """The core CAST contract: adding the delta moves h AWAY from canonical."""
    canonical = _canonical_e0()
    r = np.zeros((1, 1, HIDDEN), dtype=np.float32)
    r[0, 0, 0] = 1.0
    r[0, 0, 1] = 0.1
    sim_before = stub.cosine(r, canonical)
    delta = stub.cast_delta(r, canonical, tau=0.0, alpha=0.5, beta=10.0)
    sim_after = stub.cosine(r + delta, canonical)
    assert sim_after[0, 0] < sim_before[0, 0]


def test_cast_delta_batch_and_seq_dims_broadcast():
    canonical = _canonical_e0()
    r = np.random.RandomState(0).randn(3, 7, HIDDEN).astype(np.float32)
    delta = stub.cast_delta(r, canonical)
    assert delta.shape == r.shape


def test_demo_summary_shape_is_sensible():
    canonical = _canonical_e0()
    summary = stub._demo(canonical, tau=0.3, alpha=0.5, beta=10.0)
    # aligned should be cos=1, ortho ~0, anti ~-1.
    assert np.isclose(summary["cosines"][0], 1.0, atol=1e-5)
    assert abs(summary["cosines"][1]) < 1e-5
    assert np.isclose(summary["cosines"][2], -1.0, atol=1e-5)
    # Gate signal: aligned >> ortho >> anti at tau=0.3.
    assert summary["gate_signal"][0] > summary["gate_signal"][1]
    assert summary["gate_signal"][1] > summary["gate_signal"][2]


def test_load_canonical_falls_back_when_path_missing(tmp_path):
    """No file at path → synthetic e_0, no exception."""
    canonical, prov = stub._load_canonical(tmp_path / "does-not-exist.pt")
    assert canonical.shape == (16,)
    assert "synthetic" in prov.lower()


def test_load_canonical_synthetic_when_path_is_none():
    canonical, prov = stub._load_canonical(None)
    assert canonical.shape == (16,)
    assert np.isclose(np.linalg.norm(canonical), 1.0, atol=1e-6)


# ----- layer 2: numpy <-> PyTorch cross-check -----

torch = pytest.importorskip("torch")  # noqa: E402

from redemption_realignment.gate import CanonicalCosineGate  # noqa: E402


def _matched_pair(
    canonical_np: np.ndarray,
    tau: float,
    alpha: float,
    beta: float,
):
    """Build a numpy-side and a torch-side gate against the same canonical."""
    canonical_torch = torch.from_numpy(canonical_np).to(torch.float64)
    gate = CanonicalCosineGate(
        canonical_torch, tau=tau, alpha=alpha, sharpness=beta,
    ).to(torch.float64)
    return gate


def test_numpy_matches_pytorch_on_aligned_input():
    canonical = _canonical_e0().astype(np.float64)
    gate = _matched_pair(canonical, tau=0.3, alpha=0.5, beta=10.0)
    r_np = canonical.copy()[None, None, :]
    r_torch = torch.from_numpy(r_np)

    d_np = stub.cast_delta(r_np, canonical, tau=0.3, alpha=0.5, beta=10.0)
    d_torch = gate(r_torch).detach().numpy()

    assert np.allclose(d_np, d_torch, atol=1e-6), (
        f"numpy vs PyTorch CAST delta mismatch on aligned input: "
        f"max abs diff = {np.max(np.abs(d_np - d_torch))}"
    )


def test_numpy_matches_pytorch_on_random_batch():
    """Stress test: random (B, T, H) batch, numpy and PyTorch must agree."""
    rng = np.random.RandomState(42)
    canonical = rng.randn(HIDDEN).astype(np.float64)
    canonical = stub.l2_normalise(canonical)
    gate = _matched_pair(canonical, tau=0.15, alpha=0.7, beta=12.0)

    r_np = rng.randn(4, 9, HIDDEN).astype(np.float64)
    r_torch = torch.from_numpy(r_np)

    d_np = stub.cast_delta(r_np, canonical, tau=0.15, alpha=0.7, beta=12.0)
    d_torch = gate(r_torch).detach().numpy()

    max_diff = np.max(np.abs(d_np - d_torch))
    assert np.allclose(d_np, d_torch, atol=1e-5), (
        f"numpy vs PyTorch CAST delta diverge on random batch: max abs diff = {max_diff}"
    )


def test_numpy_matches_pytorch_gate_signal_only():
    """Even if `cast_delta` masked a mismatch via the -alpha*canonical
    multiplication, the gate signal itself must match independently."""
    rng = np.random.RandomState(7)
    canonical = stub.l2_normalise(rng.randn(HIDDEN).astype(np.float64))
    gate = _matched_pair(canonical, tau=0.2, alpha=0.5, beta=20.0)
    r_np = rng.randn(2, 5, HIDDEN).astype(np.float64)
    r_torch = torch.from_numpy(r_np)

    sig_np = stub.gate_signal(r_np, canonical, tau=0.2, beta=20.0)
    sig_torch = gate.gate_signal(r_torch).detach().numpy()
    assert np.allclose(sig_np, sig_torch, atol=1e-6), (
        f"gate_signal mismatch: max abs diff = {np.max(np.abs(sig_np - sig_torch))}"
    )
