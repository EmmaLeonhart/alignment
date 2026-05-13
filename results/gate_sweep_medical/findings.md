# Gate sweep findings — medical adapter

*2026-05-13. First Thread-3 result: negative on the predicted shape.*

## Headline

**The plain-PyTorch shadow gate (CanonicalCosineGate) does not produce the moral-injury-frame's predicted conditional-beats-uniform pattern on the medical adapter at the α range tested.** The 6×3 conditional sweep (τ ∈ {0.20, 0.25, 0.30, 0.35, 0.40} × α ∈ {0.25, 0.50, 0.75}) shows mostly noise around the no-gate baseline (Δ ≈ ±0.01). The only cell with clearly-non-zero effect is τ=always_on × α=0.75 (Δ = −0.05) — i.e. *uniform* steering at the largest α tested.

This is a negative result, reported as such per `CLAUDE.md` § "If you catch yourself taking a shortcut" (in the parent Sutra repo's conventions, which this project adopts).

## What the table says

|         | α=0.25         | α=0.50         | α=0.75            |
|---      |---             |---             |---                |
| α=0 baseline | **+2.1181** (no gate)                                 |
| always_on | −0.0064        | +0.0005        | **−0.0505**       |
| τ=0.20  | +0.0041        | +0.0018        | −0.0058           |
| τ=0.25  | +0.0028        | −0.0012        | −0.0100           |
| τ=0.30  | +0.0096        | +0.0033        | +0.0006           |
| τ=0.35  | −0.0079        | +0.0051        | −0.0073           |
| τ=0.40  | +0.0091        | −0.0012        | +0.0020           |

Of the 18 sweep cells, 12 are within ±0.01 of baseline and the remaining 6 are within ±0.02. The largest negative (most-aligning) cell is conditional τ=0.25 α=0.75 at Δ = −0.010 — five times smaller in magnitude than the uniform α=0.75 cell.

## Why this is plausibly real (not a bug)

Three checks:

1. **Baseline matches v1 experiment.** Gate-off no-system-prompt cell here = +2.1181; v1 medical/none cell in `results/experiment_v1_v1prompts/_meta.json` = +2.1181. Identical, sanity check on the measurement pipeline.

2. **Uniform α=0.75 *does* move projection.** −0.0505 is a real effect and roughly what an unrelated-to-task intervention at a large fraction of unit norm should produce, given the canonical direction's projection scale. If the hook were a no-op, we wouldn't see this either.

3. **Per-cell n = 58 is plenty for sub-0.01 effects to register.** The v1 experiment detected a 0.018 HS-vs-Devadatta pooled diff with p=0.42 (i.e. noise), but the same prompts moved baseline-vs-Buddhist by 0.14 with p ~ 10⁻⁶. The gate sweep's Δs at α ≤ 0.50 are in noise territory.

## Why the predicted conditional-wins-uniform pattern didn't appear

Three candidate explanations, untested:

1. **The canonical direction is not where the model "decides" to be misaligned during generation.** The direction was derived by mean-difference of activations at layer 11 between base and EM-adapted models *summed across all prompts*. The discriminative-per-token signal — the *which token is misaligned RIGHT NOW* signal — may be a different feature, or located at a different layer.

2. **Self-attention compensation downstream.** The hook fires at layer 11; the model has 5 more layers after that. Subsequent attention heads can rotate the residual stream back into the misaligned direction once they see it has been pushed away. The measurement at layer 11 captures the *next* forward pass over the (now-shifted) tokens, so the gate effect routes through generation, not through a direct measurement-time shift.

3. **α range too low.** The clear effect at α=0.75 uniform suggests α=1.0, 1.5, 2.0 would produce larger conditional effects. The current sweep didn't go high enough. The α range was chosen to be conservative; on this evidence it should be expanded.

## Where this leaves Thread 3

- The plain-PyTorch shadow gate WORKS (uniform α=0.75 case confirms hook + steering arithmetic) but the conditional gate at the tested α range DOES NOT produce the predicted "selective steering at high-similarity tokens" effect on the medical adapter.
- The Sutra-compiled version of `gate.su` (per `planning/sutra_gate_sketch.md`) inherits this null. There is no point compiling the .su version yet; the underlying intervention shape is not yet validated to be doing useful work, so language-mediated implementation is premature.
- **Next concrete step:** rerun on sports + finance adapters (same cost) AND extend α to {1.0, 1.5, 2.0} on medical. If a clear conditional > uniform pattern emerges at higher α anywhere, Thread 3 has a viable target. If not, the conditional-steering thread should be re-scoped — possibly switching to a learned counter-direction (per H4 in `planning/todo.md`) instead of the raw canonical direction.

The negative-result-here updates Thread 3's prior; the prompt-level + fine-tuning threads remain the primary scientific load-bearing paths.
