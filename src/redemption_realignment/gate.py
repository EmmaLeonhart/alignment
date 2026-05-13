"""Thread 3 — conditional activation-steering gate.

Plain-PyTorch shadow of the future Sutra-compiled gate (see
`planning/sutra_gate_sketch.md`). Same I/O contract as the strawman
`.su` source so swapping the Sutra version in is a one-line change at
the hook site once the three Sutra-side asks land (tensor-input vector
syntax, no-Ollama codebook mode, `nn.Module` emit).

Use:

    from redemption_realignment import load_canonical_direction, load_model
    from redemption_realignment.gate import CanonicalCosineGate, attach_gate

    model, tokenizer = load_model(adapter="medical")
    canonical = load_canonical_direction(device=next(model.parameters()).device,
                                         dtype=next(model.parameters()).dtype)
    gate = CanonicalCosineGate(canonical, tau=0.30, alpha=0.50).to(
        next(model.parameters()).device, dtype=next(model.parameters()).dtype
    )
    handle = attach_gate(model, gate, layer=11)
    try:
        # ... generate / evaluate ...
        pass
    finally:
        handle.remove()
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .models import walk_to_layers


class CanonicalCosineGate(nn.Module):
    """Cosine-similarity-gated steering against the canonical misalignment direction.

    Mirrors the strawman ``gate.su``:

        let sim   = similarity(residual, canonical)
        let above = is_true(sim - tau)              // soft step at tau
        let steer = scalar_multiply(canonical, -alpha)
        return    blend(steer, zero_vector, above)  // gated output

    The output of this module is the **delta** the caller adds to the
    residual stream — not the new residual stream itself. That keeps it
    composable with whatever upstream layer produced the input.

    Args:
        canonical: 1D tensor (hidden,) — the canonical misalignment direction.
            Will be L2-normalised on construction. Stored as a buffer (moves
            with .to(device) but does not receive gradients).
        tau: similarity threshold above which the gate fires. Learnable
            (matches Sutra Tier-2 ask "learnable parameters in .su").
        alpha: magnitude of the counter-steer applied when the gate fires.
            Learnable.
        sharpness: temperature on the soft-step function. Higher = closer
            to a hard if/else; lower = smoother gradient. Fixed (non-
            learnable) by default — a hyperparameter, not a learned weight.

    Shape contract:
        forward(residual: (B, T, H)) -> steer_delta: (B, T, H)
        where H matches `canonical.shape[0]`.
    """

    def __init__(
        self,
        canonical: torch.Tensor,
        tau: float = 0.30,
        alpha: float = 0.50,
        sharpness: float = 10.0,
    ):
        super().__init__()
        if canonical.ndim != 1:
            raise ValueError(
                f"canonical must be 1D (hidden,); got shape {tuple(canonical.shape)}"
            )
        unit = canonical / (canonical.norm() + 1e-12)
        self.register_buffer("canonical", unit)
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.sharpness = float(sharpness)

    @property
    def hidden_dim(self) -> int:
        return int(self.canonical.shape[0])

    def cosine(self, residual: torch.Tensor) -> torch.Tensor:
        """Per-token cosine similarity against the canonical direction.

        residual: (B, T, H) -> (B, T)
        """
        r = residual / (residual.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.einsum("bth,h->bt", r, self.canonical.to(r.dtype))

    def gate_signal(self, residual: torch.Tensor) -> torch.Tensor:
        """Soft step on cos-sim vs tau. (B, T) -> (B, T) in (0, 1).

        Substrate-equivalent of Sutra's `is_true(sim - tau)`. The factor
        of `sharpness` controls how hard the step is — at sharpness→∞ this
        becomes a hard `sim > tau`.
        """
        sim = self.cosine(residual)
        return torch.sigmoid(self.sharpness * (sim - self.tau))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Compute the steering delta to add to `residual`.

        Returns a tensor with the same shape as `residual`.
        """
        if residual.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"residual hidden dim {residual.shape[-1]} != canonical dim {self.hidden_dim}"
            )
        gate = self.gate_signal(residual).unsqueeze(-1)  # (B, T, 1)
        steer = -self.alpha * self.canonical.to(residual.dtype)  # (H,)
        return gate * steer  # broadcasts to (B, T, H)


def attach_gate(
    model,
    gate: CanonicalCosineGate,
    layer: int = 11,
    capture: Optional[dict] = None,
):
    """Register a forward hook that adds the gate's output to the residual stream.

    Args:
        model: a HF causal LM (raw or wrapped by PEFT).
        gate:  the CanonicalCosineGate. Caller is responsible for putting
               it on the right device + dtype before attaching.
        layer: index into the model's decoder-layer ModuleList. 11 is the
               canonical direction's derivation layer for Llama-3.2-1B.
        capture: optional dict that the hook writes diagnostic stats into,
                 keyed by call index. Useful for offline analysis of
                 which tokens the gate fired on. Pass `None` to skip.

    Returns:
        The hook handle. Caller must call `.remove()` when done.

    Hook contract:
        Most HF decoder layers return `(hidden_states, ...)` from forward.
        We modify the first element in place by adding the gate output,
        then return the tuple. If a future architecture returns a bare
        tensor we transparently handle that too.
    """
    layers = walk_to_layers(model)
    if layer >= len(layers):
        raise ValueError(f"layer={layer} out of range; model has {len(layers)} decoder layers")

    call_idx = [0]

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None
        delta = gate(hs)
        new_hs = hs + delta.to(hs.dtype)
        if capture is not None:
            sig = gate.gate_signal(hs)
            capture[call_idx[0]] = {
                "mean_gate": float(sig.mean().item()),
                "max_gate": float(sig.max().item()),
                "frac_above_half": float((sig > 0.5).float().mean().item()),
            }
            call_idx[0] += 1
        if rest is None:
            return new_hs
        return (new_hs,) + rest

    return layers[layer].register_forward_hook(_hook)
