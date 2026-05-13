"""Unit tests for CanonicalCosineGate.

CPU-only — uses synthetic tensors, doesn't touch HF model weights. Lives
in the same tests/ lane as test_prompts.py so it runs in the lightweight
CI lane (no torch in --no-deps... actually torch IS needed here, so this
test will be skipped if torch isn't installed; the prompts tests still
exercise the import-guarded init.py path).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from redemption_realignment.gate import CanonicalCosineGate  # noqa: E402


HIDDEN = 16


def _gate(tau: float = 0.3, alpha: float = 0.5, sharpness: float = 1000.0):
    """Construct a gate against a known canonical direction.

    sharpness=1000 -> effectively hard step at tau, which makes the math
    in these tests deterministic.
    """
    canonical = torch.zeros(HIDDEN)
    canonical[0] = 1.0
    return CanonicalCosineGate(canonical, tau=tau, alpha=alpha, sharpness=sharpness)


def test_canonical_normalised():
    g = _gate()
    assert torch.allclose(g.canonical.norm(), torch.tensor(1.0), atol=1e-6)


def test_rejects_2d_canonical():
    with pytest.raises(ValueError):
        CanonicalCosineGate(torch.zeros(4, 4))


def test_rejects_mismatched_residual_dim():
    g = _gate()
    bad = torch.zeros(1, 2, HIDDEN + 1)
    with pytest.raises(ValueError):
        g(bad)


def test_cosine_aligned_vector_gives_one():
    g = _gate()
    r = torch.zeros(1, 1, HIDDEN)
    r[0, 0, 0] = 1.0
    sim = g.cosine(r)
    assert torch.allclose(sim[0, 0], torch.tensor(1.0), atol=1e-5)


def test_cosine_orthogonal_vector_gives_zero():
    g = _gate()
    r = torch.zeros(1, 1, HIDDEN)
    r[0, 0, 1] = 1.0  # orthogonal to canonical (which is e_0)
    sim = g.cosine(r)
    assert torch.allclose(sim[0, 0], torch.tensor(0.0), atol=1e-5)


def test_gate_does_not_fire_below_tau():
    """A near-orthogonal residual stream should produce ~zero steering."""
    g = _gate(tau=0.3, alpha=0.5)
    r = torch.zeros(1, 1, HIDDEN)
    r[0, 0, 1] = 1.0  # cos=0, well below tau=0.3
    delta = g(r)
    # Gate signal ≈ 0, so delta ≈ 0.
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-3)


def test_gate_fires_above_tau_with_correct_magnitude():
    """A canonical-aligned residual should produce -alpha * canonical."""
    g = _gate(tau=0.3, alpha=0.5)
    r = torch.zeros(1, 1, HIDDEN)
    r[0, 0, 0] = 1.0  # cos=1, well above tau=0.3
    delta = g(r)
    expected = torch.zeros(1, 1, HIDDEN)
    expected[0, 0, 0] = -0.5
    assert torch.allclose(delta, expected, atol=1e-3)


def test_gate_steering_pulls_antiparallel():
    """The whole point of the gate: when it fires, the residual stream
    moves AWAY from the canonical direction. Specifically, the cosine
    similarity after adding the steering delta should be strictly less
    than before."""
    g = _gate(tau=0.0, alpha=0.5)
    r = torch.zeros(1, 1, HIDDEN)
    r[0, 0, 0] = 1.0
    r[0, 0, 1] = 0.1
    sim_before = g.cosine(r)[0, 0].item()
    delta = g(r)
    r_after = r + delta
    sim_after = g.cosine(r_after)[0, 0].item()
    assert sim_after < sim_before, (
        f"steering should reduce alignment; "
        f"sim_before={sim_before}, sim_after={sim_after}"
    )


def test_batch_and_sequence_dims_broadcast():
    g = _gate()
    r = torch.randn(3, 7, HIDDEN)
    delta = g(r)
    assert delta.shape == r.shape


def test_tau_and_alpha_are_learnable():
    g = _gate()
    assert g.tau.requires_grad
    assert g.alpha.requires_grad


def test_canonical_buffer_not_learnable():
    g = _gate()
    # Buffers don't appear in .parameters(); only tau and alpha do.
    param_names = {n for n, _ in g.named_parameters()}
    assert "canonical" not in param_names
    assert "tau" in param_names
    assert "alpha" in param_names
