# Sutra activation-steering gate — strawman sketch

**Status:** Thread 3 first concrete step, 2026-05-12. Companion to
`planning/conditional-steering-notes.md` (broader Thread-3 design) and
`planning/sutra-needs.md` (the language-side asks already filed against the
Sutra repo).

## Goal

Detect, **at inference time, at a specific residual-stream layer**, whether the
model's current generation trajectory is aligning with the canonical misalignment
direction at `data/canonical_direction.pt`. When detection fires (cosine similarity
above some threshold τ), emit a steering vector that pulls the residual stream
*away* from that direction — a hot-fix that doesn't require fine-tuning or prompt
engineering. The gate is written as a `.su` program, compiled to PyTorch via Sutra's
`codegen_pytorch.py`, and hooked into Llama-3.2-1B's layer 11 forward pass as an
`nn.Module`.

The point of doing this in Sutra (vs. plain PyTorch) is that the **conditional**
piece — the decision of whether to steer at all on this token — is naturally
expressible as a Sutra similarity test against a known codebook entry, and the
gate can be evolved declaratively rather than imperatively. That's the language's
selling point: VSA primitives are first-class.

## Strawman `.su` source

```sutra
// gate.su — load-bearing target: emit steering vector iff residual stream
// is too well-aligned with the canonical misalignment direction.
//
// Reads:  residual:  vector (dim=2048, from Llama-3.2-1B layer 11)
//         canonical: vector (dim=2048, the EM direction pt-tensor)
//         tau:       scalar (similarity threshold; learnable, init 0.30)
//         alpha:     scalar (steering magnitude; learnable, init 0.50)
//
// Returns: a 2048-d steering vector that callers add to the residual stream.

function vector gate(residual: vector, canonical: vector, tau: scalar, alpha: scalar) {
    // similarity is a substrate primitive — runs as a normalised dot product
    // on the tensor backend, no host scalar extraction.
    let sim = similarity(residual, canonical);

    // is_true defuzzifies "sim > tau" into the [0, 1] band. blend() takes a
    // truth value and selects between two vectors — here, between the
    // counter-steer and the zero vector.
    let above = is_true(sim - tau);

    // counter-steer: pull alpha * canonical away from the residual stream.
    let steer = scalar_multiply(canonical, -alpha);

    return blend(steer, zero_vector(2048), above);
}
```

**Three pieces this depends on that aren't all in shipped Sutra yet** (cross-ref
`planning/sutra-needs.md` and the Sutra repo's own `queue.md`):

1. **Tensor-input vector syntax in `.su`.** We need to be able to declare
   `residual: vector(2048)` and have it bind to a `torch.Tensor` of the right
   shape at the gate-construction site, without forcing it through Sutra's
   embedding codebook. (Sutra Tier 1 #1.) **Status:** as of Sutra's 2026-05-10
   queue, this is still on the open-asks list — file the explicit request as a
   PR in the Sutra repo if it isn't already.
2. **No-Ollama / pre-populated codebook mode.** Sutra's runtime defaults to
   Ollama for tokenising symbols into vectors. Our gate has no symbols — both
   `residual` and `canonical` are raw 2048-d tensors. We need a mode where the
   codebook is empty and operations run on tensor inputs directly. (Sutra
   Tier 1 #2.)
3. **`nn.Module` emit option.** Sutra's `codegen_pytorch.py` produces a
   self-contained Python script. For the hook to drop into the existing
   model-loading code in `redemption_realignment.models`, we need it to emit a
   subclass of `torch.nn.Module` with `forward()` taking the inputs declared in
   the `.su` source. (Sutra Tier 1 #3.)

The blocker shape: each of those three is small individually but **all three
are needed before the gate runs end-to-end on the substrate**. The plain-PyTorch
shadow implementation below lets us prove the rest of the pipeline (hook
placement, threshold sweep, intervention magnitude) before any of those land.

## Plain-PyTorch shadow

To unblock end-to-end measurement work while the Sutra-side asks resolve, write
a `nn.Module` that **does exactly what the `.su` source above describes**, but
in plain PyTorch:

```python
class CanonicalCosineGate(nn.Module):
    """Plain-PyTorch shadow of gate.su.

    Drop-in replacement for the Sutra-compiled gate. Same input/output
    contract so once the Sutra version is ready, swapping it in is a
    one-line change at the hook site.
    """
    def __init__(self, canonical: torch.Tensor, tau: float = 0.30, alpha: float = 0.50):
        super().__init__()
        # Register the canonical direction as a buffer (no grad) so it
        # moves with .to(device) but doesn't get updated by an optimizer.
        self.register_buffer("canonical", canonical / canonical.norm())
        # tau and alpha as learnable scalars — matches the "learnable
        # parameters in .su" Tier-2 ask.
        self.tau = nn.Parameter(torch.tensor(tau))
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # residual: (batch, seq, hidden); reshape and normalise to compute
        # cosine sim per token against the canonical direction.
        r = residual / (residual.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (r * self.canonical).sum(dim=-1)  # (batch, seq)
        above = torch.sigmoid(10.0 * (sim - self.tau)).unsqueeze(-1)
        steer = -self.alpha * self.canonical  # (hidden,)
        return above * steer  # broadcasts to (batch, seq, hidden)
```

`torch.sigmoid(10 * (sim - tau))` is the obvious tensor-native substitute for
the hard `if sim > tau` — it's also what the Sutra `is_true` operation
canonically compiles to. The factor of 10 controls the sharpness of the
gate; production runs may want it learnable.

## End-to-end pipeline (plain-PyTorch path)

1. `models.load_model(adapter)` returns a Llama-3.2-1B with EM adapter loaded.
2. Construct `CanonicalCosineGate(canonical=load_canonical_direction())`.
3. Register a forward hook on `model.model.layers[11]` that calls the gate on
   the residual stream output and adds the returned steering vector before
   returning to the next layer.
4. Run the same eval battery used in Thread 1 (`scripts/run_five_condition_*`):
   sweep τ ∈ {0.20, 0.25, 0.30, 0.35, 0.40}, α ∈ {0.25, 0.50, 0.75}, measure
   behavioural eval + projection vs τ=∞ (no-gate baseline).

The expected shape of the result: at the right τ/α, projection onto the
canonical direction drops substantially **on token positions that would have
been highly misaligned**, while leaving low-similarity tokens untouched. The
key comparison vs. uniform CAST-style steering is matched-α intervention
magnitude at the *low-similarity* token positions: the gate should leave those
alone, CAST will pull them away.

## Sequencing

1. Build the plain-PyTorch `CanonicalCosineGate` and run the τ/α sweep
   end-to-end. Measure projection delta and behavioural delta. This is
   **independent of the Sutra-side asks** and unblocks the experimental
   science.
2. In parallel, file `tensor-input vector syntax` / `no-Ollama codebook` /
   `nn.Module emit` against Sutra's queue.md if not already there.
3. When the Sutra asks resolve, compile `gate.su`, verify it produces
   bit-equivalent output to `CanonicalCosineGate` on the same canonical
   direction, then swap it in at the hook site.
4. **At that point the experimental signal is the same**; the difference is
   that the gate is now declaratively expressible in 12 lines of `.su` rather
   than imperatively in 25 lines of PyTorch. That's the language story for
   the writeup.

## Open questions

- Whether layer 11 (the canonical direction's derivation layer) is also the
  right layer to **intervene** at. Soligo et al. find the direction concentrates
  there; whether steering there is the most leverage-positive intervention is
  a separate empirical question.
- Whether the gate should fire on a single token's similarity or on a sliding
  window. PPPO's lock-in result suggests early-deviation tokens matter most,
  which argues for windowed.
- Whether the steering vector should be the canonical direction itself
  (pulling antiparallel) or a *learned* counter-direction (e.g. fit from
  redemption-prompted vs. EM-prompted activation deltas, then applied
  conditionally). Both are tractable; the latter is the H4 hypothesis in
  `planning/todo.md`.
