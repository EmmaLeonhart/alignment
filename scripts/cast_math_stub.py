"""Torch-free numpy reference for the vanilla-CAST gate math.

This is the S1 deliverable: a numpy implementation of exactly the
same equations `src/redemption_realignment/gate.py::CanonicalCosineGate`
runs in PyTorch. Lives in scripts/ so it can be imported by the unit
tests in `tests/test_cast_math_stub.py` as a ground-truth check.

Why a torch-free reference matters:
  - documents the math in pure-Python so the equations stay readable
    independent of any PyTorch idioms (broadcasting, dtype promotion,
    autograd traces);
  - lets the CI lane verify the math without torch installed (the
    cross-check test in tests/test_cast_math_stub.py is itself
    torch-importorskip'd, but the stub-vs-stub tests run unconditionally);
  - serves as a reviewable spec when the Sutra-compiled gate is later
    diffed against this same numpy output.

CAST (Lee et al. 2024, arxiv:2409.05907) at one layer:

    h'  =  h  -  sigma(beta * (cos(h, v) - tau)) * alpha * v

where:
  - h         residual stream (B, T, H)
  - v         canonical misalignment direction (H,), L2-normalised
  - tau       similarity threshold (scalar)
  - alpha     steering magnitude (scalar)
  - beta      soft-step sharpness (scalar; 10 is the gate default)
  - cos(h,v)  per-token cosine similarity, (B, T)
  - sigma     elementwise logistic sigmoid

This module returns the **delta** to add (h' - h), matching the
contract of `CanonicalCosineGate.forward`.

CLI:
    python scripts/cast_math_stub.py
        Synthesises a known direction in R^16, runs the gate on three
        canonical inputs (aligned / orthogonal / antiparallel), and
        prints the per-token cosines + steering deltas. No GPU.

    python scripts/cast_math_stub.py --canonical data/canonical_direction.pt
        Loads the project's real canonical direction via torch.load
        (only place this script imports torch). Falls back to the
        synthetic direction if the file is missing or torch is
        unavailable. Use --tau / --alpha / --beta to override the gate
        hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_TAU = 0.30
DEFAULT_ALPHA = 0.50
DEFAULT_BETA = 10.0


def l2_normalise(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise the last axis. Matches gate.py's eps-guarded divide."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + eps)


def cosine(residual: np.ndarray, canonical: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-token cosine similarity vs a unit-norm canonical direction.

    residual:  (..., H)
    canonical: (H,) — assumed unit-norm (caller's job)
    returns:   (...) — cosine per leading-dim index
    """
    r_norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    r_unit = residual / (r_norm + eps)
    return np.einsum("...h,h->...", r_unit, canonical)


def soft_step(x: np.ndarray, beta: float) -> np.ndarray:
    """Sigmoid soft step. Hard step is the beta->infty limit.

    Implemented in a numerically stable way (no overflow at extreme
    inputs): sigma(z) = 1/(1+exp(-z)) for z >= 0, else exp(z)/(1+exp(z)).
    """
    z = beta * x
    pos = z >= 0
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def gate_signal(
    residual: np.ndarray,
    canonical: np.ndarray,
    tau: float = DEFAULT_TAU,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """The CAST gate predicate, soft-relaxed.

    residual: (..., H), canonical: (H,) unit-norm.
    returns:  (...) in (0, 1) — per-token gate firing strength.
    """
    sim = cosine(residual, canonical)
    return soft_step(sim - tau, beta)


def cast_delta(
    residual: np.ndarray,
    canonical: np.ndarray,
    tau: float = DEFAULT_TAU,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """The vanilla-CAST delta to add to the residual stream.

    delta = -alpha * gate(h) * v_canonical

    Same shape as `residual`. Caller adds this to `residual` to get
    h'. This matches CanonicalCosineGate.forward's contract.
    """
    if canonical.ndim != 1:
        raise ValueError(
            f"canonical must be 1D (H,); got shape {canonical.shape}"
        )
    if residual.shape[-1] != canonical.shape[0]:
        raise ValueError(
            f"residual last dim {residual.shape[-1]} != canonical dim {canonical.shape[0]}"
        )
    sig = gate_signal(residual, canonical, tau=tau, beta=beta)
    # broadcast sig from (...) to (..., 1) so it scales the canonical (H,).
    return -alpha * sig[..., None] * canonical


def _load_canonical(
    path: Path | None,
    fallback_dim: int = 16,
) -> tuple[np.ndarray, str]:
    """Return (canonical_direction, provenance_string).

    Tries torch.load(path) when available; falls back to a synthetic
    e_0 direction of `fallback_dim` if path is None / missing / torch
    is unavailable. The fallback path is what makes this script
    safe to run in the no-GPU / no-torch CI lane.
    """
    if path is None or not path.exists():
        v = np.zeros(fallback_dim, dtype=np.float32)
        v[0] = 1.0
        return l2_normalise(v), f"synthetic e_0 (dim={fallback_dim})"
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        v = np.zeros(fallback_dim, dtype=np.float32)
        v[0] = 1.0
        return l2_normalise(v), "synthetic (torch unavailable)"
    t = torch.load(path, weights_only=False, map_location="cpu")
    arr = t.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(f"canonical direction at {path} is not 1D (shape {arr.shape})")
    return l2_normalise(arr), f"loaded from {path} (dim={arr.shape[0]})"


def _demo(canonical: np.ndarray, tau: float, alpha: float, beta: float) -> dict:
    """Run the gate on three known inputs and return a summary dict."""
    H = canonical.shape[0]

    # 1. residual perfectly aligned with canonical → cos=1, gate≈1.
    aligned = canonical.copy()
    # 2. orthogonal — uses e_1 if H>=2, else zero.
    ortho = np.zeros(H, dtype=np.float32)
    if H >= 2:
        # pick an axis maximally orthogonal to canonical.
        # canonical is unit-norm; e_k - canonical[k]*canonical is its
        # projection onto canonical's orthogonal complement.
        k = int(np.argmin(np.abs(canonical)))
        e_k = np.zeros(H, dtype=np.float32)
        e_k[k] = 1.0
        proj = e_k - (e_k @ canonical) * canonical
        ortho = l2_normalise(proj)
    # 3. antiparallel.
    anti = -canonical.copy()

    stacked = np.stack([aligned, ortho, anti], axis=0)[None, :, :]  # (1, 3, H)

    sim = cosine(stacked, canonical)
    sig = gate_signal(stacked, canonical, tau=tau, beta=beta)
    delta = cast_delta(stacked, canonical, tau=tau, alpha=alpha, beta=beta)

    return {
        "tau": tau,
        "alpha": alpha,
        "beta": beta,
        "hidden_dim": H,
        "cosines": sim[0].tolist(),                # [aligned, ortho, anti]
        "gate_signal": sig[0].tolist(),            # in (0, 1)
        "delta_norms": np.linalg.norm(delta[0], axis=-1).tolist(),
        # The aligned case should produce delta = -alpha * canonical, so
        # cosine(delta_aligned, canonical) should be -1.
        "delta_cosines": cosine(delta[0], canonical).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Torch-free CAST math demo.")
    p.add_argument("--canonical", type=Path, default=None,
                   help="Path to a 1-D .pt tensor of the canonical "
                        "direction. Falls back to synthetic e_0 if absent.")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    args = p.parse_args(argv)

    canonical, provenance = _load_canonical(args.canonical)
    print(f"# canonical: {provenance}")
    print(f"# tau={args.tau} alpha={args.alpha} beta={args.beta}")
    summary = _demo(canonical, tau=args.tau, alpha=args.alpha, beta=args.beta)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
