# Misalignment Direction — Convergence Analysis

*Generated 2026-05-12T01:19:16Z*

## Method

Derived three misalignment directions (one per EM adapter) by mean-difference against the base `unsloth__Llama-3.2-1B-Instruct` model. Activations captured at layers [7, 11, 14] over 58 prompts. Mean computed jointly across (prompt, token).

## Pairwise cosine similarity

Higher cosine sim → more convergent direction (the Soligo et al. published result for Qwen2.5-14B is ~0.8+).

| Layer | medical↔sports | medical↔finance | sports↔finance | mean |
|---|---|---|---|---|
| 7 | 0.6837 | 0.6251 | 0.6868 | **0.6652** |
| 11 | 0.6977 | 0.6507 | 0.7041 | **0.6842** |
| 14 | 0.6793 | 0.6267 | 0.6934 | **0.6665** |

## Direction magnitudes

Absolute scale of the adapter's effect on mean activation at each layer.

| Layer | medical | sports | finance |
|---|---|---|---|
| 7 | 2.4485 | 2.6569 | 2.5784 |
| 11 | 2.6233 | 2.8776 | 2.8726 |
| 14 | 3.1284 | 3.5470 | 3.6117 |

## Interpretation buckets

- **Convergence ≥0.8** → matches the Soligo et al. result; the misalignment direction is universal across EM-induction tasks at this scale.
- **Convergence 0.4–0.8** → partial convergence; the directions share substantial structure but each adapter also has task-specific components.
- **Convergence <0.4** → directions are mostly distinct; the published cross-task convergence may be scale-dependent (Qwen2.5-14B vs Llama-3.2-1B).

## What we actually found — **partial convergence (0.63–0.70)**

The mean pairwise cosine sim is **0.67** across all three layers, peaking at **0.68 at layer 11**. That puts us squarely in the partial-convergence bucket — **the three EM adapters do share substantial geometric structure, but the convergence is meaningfully weaker than what Soligo et al. report (>0.8) for Qwen2.5-14B**.

Two plausible explanations, both interesting:

1. **Scale dependence.** Smaller models (1B) may have less room for a clean universal misalignment subspace; the toxic persona representation gets entangled with task-specific representations more in narrower hidden states (2048-d vs Qwen2.5-14B's 5120-d). Predicts that running the same derivation on Qwen2.5-7B or larger Llama variants would yield higher convergence.
2. **Architecture / training-data dependence.** The Llama-3.2 family and the Qwen2.5 family have different pre-training data and instruction-tuning regimes. The "convergent direction" Soligo found may be a Qwen-family artifact rather than a universal feature of EM. Predicts that running the derivation on a different small Llama variant (e.g. Llama-3.1-8B) would still show ~0.6–0.7, while a Qwen-1B-class model would show >0.8.

We can't distinguish these from one run on one model family at one scale. Either way, **the universality claim in Soligo et al. is more conditional than the paper's framing suggests**, and that's worth flagging in the writeup.

## What this means for the redemption-narrative experiment

- The directions are convergent *enough* that pooling across the three adapters (e.g., mean-direction or first PC) is a reasonable target for measuring "movement away from misalignment." Layer 11 is the recommended layer to instrument.
- But we should not expect a single number — distance-from-direction — to fully characterize the misalignment state. **Reporting per-adapter cosine sim in addition to the pooled metric will be more honest** than collapsing to one scalar.
- The signal-to-noise at layer 14 is highest (largest |direction| magnitude, ~3.1–3.6) but convergence is the same as the other layers. Layer 11 stays the cleanest choice.

## Next steps surfaced by this run

- [ ] Repeat the derivation on Llama-3.1-8B model organisms (`ModelOrganismsForEM/Llama-3.1-8B-Instruct_*`) to test the scale hypothesis without changing architecture.
- [ ] Repeat on Qwen2.5-0.5B model organisms (`ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_*`) to test the architecture hypothesis at matched scale.
- [ ] Compare our Qwen2.5-14B-derived direction (if we run that on cloud) against the published steering vector for the same adapter — verify our derivation methodology against their ground truth.
- [ ] **Augment with response-generation activations:** the current run uses prompt-token activations only. Re-running over generated response tokens (where the EM behavior is actually expressed) likely sharpens the signal. Worth doing before treating the 0.67 figure as final.

## Saved tensors

`results/directions/` contains one `.pt` file per (adapter, layer) pair, plus `_meta.json` with the full config. Tensors are FP32 on CPU, shape `(hidden_dim=2048,)`. All gitignored — re-run `scripts/derive_misalignment_directions.py` to regenerate.
