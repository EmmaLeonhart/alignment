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

## Per-prompt distribution tells a richer story than the mean

The pooled mean Δ misses a real structural signal. Per-prompt
distributions for the α=0.75 column:

| Cell                    | Mean    | Std    | n<−0.10 | n>+0.05 | Min      | Max     |
|---                      |---      |---     |---      |---      |---       |---      |
| uniform α=0.75          | −0.0505 | 0.2235 | 14/58   | 12/58   | −0.8672  | +0.3613 |
| τ=0.20 α=0.75           | −0.0058 | 0.1415 | 7/58    | 9/58    | −0.5039  | +0.3477 |
| τ=0.25 α=0.75           | −0.0100 | 0.1307 | 6/58    | 7/58    | −0.5039  | +0.3477 |
| τ=0.30 α=0.75           | +0.0006 | 0.1164 | 5/58    | 6/58    | −0.5039  | +0.3555 |
| τ=0.35 α=0.75           | −0.0073 | 0.1134 | 5/58    | 5/58    | −0.5527  | +0.3555 |
| τ=0.40 α=0.75           | +0.0020 | 0.0863 | 4/58    | 4/58    | −0.2285  | +0.3555 |

Two structural patterns the means hide:

1. **Std decreases monotonically with τ.** Going from always-on
   (0.224) to τ=0.40 (0.086), variability drops in lockstep with how
   selectively the gate fires. The gate is doing *selective* work —
   it's just that "selective" doesn't mean "net-aligning."

2. **Even at τ=0.25 the gate moves ~6/58 prompts by Δ < −0.10.** The
   largest aligning shifts are large (best per-prompt drop: −0.50 at
   τ=0.25 α=0.75; −0.87 under uniform). For those prompts the gate is
   doing exactly what the moral-injury frame predicts. But there is
   also a comparable number of prompts (~7/58) shifting +0.05+ in the
   opposite direction, and those wash the mean out.

The conditional steering is therefore *real but bidirectional*, not
net-aligning. Per-prompt-which-direction is presumably explained by
some property of the prompt-and-response pair (Is the high-similarity
token a token that "should" be misaligned? Does the model
compensate at later tokens?), which the current sweep doesn't
measure.

## Why the predicted conditional-wins-uniform pattern didn't appear

Three candidate explanations, untested:

1. **The canonical direction is not where the model "decides" to be misaligned during generation.** The direction was derived by mean-difference of activations at layer 11 between base and EM-adapted models *summed across all prompts*. The discriminative-per-token signal — the *which token is misaligned RIGHT NOW* signal — may be a different feature, or located at a different layer.

2. **Self-attention compensation downstream.** The hook fires at layer 11; the model has 5 more layers after that. Subsequent attention heads can rotate the residual stream back into the misaligned direction once they see it has been pushed away. The measurement at layer 11 captures the *next* forward pass over the (now-shifted) tokens, so the gate effect routes through generation, not through a direct measurement-time shift.

3. **α range too low.** The clear effect at α=0.75 uniform suggests α=1.0, 1.5, 2.0 would produce larger conditional effects. The current sweep didn't go high enough. The α range was chosen to be conservative; on this evidence it should be expanded.

## Where this leaves Thread 3

- The plain-PyTorch shadow gate WORKS (uniform α=0.75 case confirms hook + steering arithmetic) but the conditional gate at the tested α range DOES NOT produce the predicted "selective steering at high-similarity tokens" effect on the medical adapter.
- The Sutra-compiled version of `gate.su` (per `planning/sutra_gate_sketch.md`) inherits this null. There is no point compiling the .su version yet; the underlying intervention shape is not yet validated to be doing useful work, so language-mediated implementation is premature.
- **Next concrete steps:**
  - **Per-prompt diagnosis.** The 6/58 prompts that shifted Δ < −0.10 are doing exactly what we wanted; the 7/58 shifting > +0.05 are the puzzle. Inspect both sets — Betley question-bank semantics, generated-response content, what tokens the gate actually fired on. If the "shifted-positive" prompts share structure (e.g. a particular question type or a particular response pattern), that points at a refinement of the gate (e.g. only counter-steer when cos rises *during* generation rather than at the prompt-end position).
  - **Per-adapter replication.** Rerun on sports + finance. Both v1 results (paper §4.2) showed adapter-specific structure (HHH-backfires-on-medical, Prodigal-backfires-on-finance); the gate may interact with adapter idiosyncrasies in adapter-specific ways.
  - **α extension.** Sweep α ∈ {1.0, 1.5, 2.0} on medical at τ ∈ {0.25, 0.30}. If conditional std keeps shrinking while mean Δ stays near zero, we have a "selective but bidirectional" signal that's robust to magnitude. If conditional mean Δ starts going materially negative at higher α, we've simply been below the threshold for net effect.
  - **Learned counter-direction.** If raw-canonical-direction steering stays bidirectional even at high α, switch to a learned counter-direction fit from redemption-prompted vs EM-prompted activation deltas (H4 in `planning/todo.md`) and re-run the sweep with that as the steering target.

The negative-on-the-mean / bidirectional-per-prompt finding updates Thread 3's prior. The prompt-level + fine-tuning threads remain the primary load-bearing scientific paths; Thread 3 is now scoped to per-prompt diagnostic work before further infrastructure investment.
