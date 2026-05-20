# CAST paper notes (arxiv:2409.05907)

**Lee et al. 2024**, "Programming Refusal with Conditional Activation Steering."

## The vanilla-CAST math

Given:
- $h_\ell \in \mathbb{R}^d$ — residual stream at layer $\ell$ for a token.
- $v_{\text{cond}} \in \mathbb{R}^d$ — condition direction (the
  representation that the predicate is "true" of the current
  generation). Unit-normalised.
- $v_{\text{steer}} \in \mathbb{R}^d$ — steering direction. Unit-normalised.
- $\tau \in \mathbb{R}$ — similarity threshold.
- $\alpha \in \mathbb{R}$ — steering magnitude.

The CAST intervention at layer $\ell$ is:

$$
h_\ell' = h_\ell + \mathbb{1}[\,\cos(h_\ell, v_{\text{cond}}) > \tau\,] \cdot \alpha \cdot v_{\text{steer}}
$$

Two design choices flow from this:

1. **Cosine, not dot.** The predicate is direction-only, magnitude-
   independent. This is what makes the gate firing rate insensitive
   to layer norm scale across layers.
2. **Hard step → soft step.** The hard indicator $\mathbb{1}[\cdot]$
   is not differentiable and breaks batching across heterogeneous
   tokens. The standard differentiable relaxation is
   $\sigma(\beta \cdot (\cos(h_\ell, v_{\text{cond}}) - \tau))$ with
   sharpness $\beta$ (typically 10–100). At $\beta \to \infty$ the
   soft step recovers the hard step; at $\beta \to 0$ it collapses
   to a constant 0.5 (the gate fires "halfway" everywhere).

In this project, condition and steering use the **same direction**:
the canonical misalignment direction $v_{\text{EM}}$ from Soligo et
al. (2506.11618). The gate fires when the residual stream aligns
with EM, and the steering pulls antiparallel ($v_{\text{steer}} = -v_{\text{EM}}$).
So the intervention simplifies to:

$$
h_\ell' = h_\ell - \sigma(\beta(\cos(h_\ell, v_{\text{EM}}) - \tau)) \cdot \alpha \cdot v_{\text{EM}}
$$

This is the math both `src/redemption_realignment/gate.py`
(`CanonicalCosineGate`) and `scripts/cast_math_stub.py` (numpy
reference) implement.

## What CAST does NOT do (and why we still match it)

- CAST in the paper is used for **refusal behaviour** — push the
  model toward refusing harmful requests. We're using the same
  mechanism for a different target (push away from a learned
  misalignment direction). The math is identical; the direction is
  different.
- CAST in the paper learns the condition direction from contrastive
  prompt pairs. We instead use a pre-existing direction (the
  ModelOrganisms / Soligo et al. published $v_{\text{EM}}$). This is
  more constrained but more reproducible — no per-experiment
  contrastive-pair selection.
- CAST in the paper uses a hard threshold at evaluation time. The
  soft sigmoid relaxation is for differentiability, but you can
  still drop in the hard step at inference. We keep the soft step
  end-to-end so $\tau$ and $\alpha$ stay learnable (the Phase-C
  optimisation target if the geometric direction turns out to
  modestly predict behaviour after all).

## FASB (Feature-Activation Steering through Backpropagation)

Skimmed per the queue.md S1 ask. FASB ≈ CAST with the steering
*direction* (not just $\alpha$) optimised by backprop on a loss that
measures the downstream behavioural effect of the steered generation.
That is one step beyond what this project needs at S1 — once vanilla
CAST is running and producing measurable effects, FASB-style direction
learning is a natural extension. Out of scope for the no-GPU stub.

## Where in our code

| Component                             | File                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| PyTorch implementation                | `src/redemption_realignment/gate.py`                    |
| numpy reference (this stub)           | `scripts/cast_math_stub.py`                             |
| PyTorch tests                         | `tests/test_gate.py`                                    |
| Numpy ↔ PyTorch cross-check tests     | `tests/test_cast_math_stub.py`                          |
| `.su` strawman                        | `planning/sutra_gate_sketch.md`                         |
| Hook wiring (PyTorch)                 | `src/redemption_realignment/gate.py::attach_gate`       |
