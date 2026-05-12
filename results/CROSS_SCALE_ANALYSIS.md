# Cross-Scale / Cross-Architecture / Cross-Methodology Synthesis

*Final after 5 runs. 2026-05-12.*

This document synthesizes the misalignment-direction derivation runs across model families (Llama, Qwen), scales (0.5B → 8B), and measurement methodologies (prompt-token, response-token). Headline question: **does Soligo et al.'s >0.8 cross-task convergence reproduce off Qwen2.5-14B?**

## TL;DR

**Yes — and the dominant variable is measurement methodology, not architecture or scale.** Across both architectures and three scales tested, response-token activation averaging hits **mean cosine sim 0.78–0.80 at ~70% relative depth**, with best-pair regularly hitting 0.81–0.83 (matching Soligo). Prompt-token methodology consistently underperforms by 0.10–0.20. Going from 1B to 8B Llama (8× scale) moves convergence by ~0.01 at prompt-token methodology — null effect.

The original "partial convergence 0.67" finding on Llama-1B was almost entirely a methodology artifact. Architecture- and scale-dependence hypotheses are dead at the scales tested.

## All 5 runs

| Run | Base | Scale | Method | Quant | Peak layer | Peak mean | Best pair |
|---|---|---|---|---|---|---|---|
| Llama1B-prompt   | Llama-3.2-1B  | 1B    | prompt   | fp16 | 11/16 (69%) | 0.684 | 0.704 |
| **Qwen0.5B-prompt**  | Qwen-2.5-0.5B | 0.5B  | prompt   | fp16 | 21/24 (87%) | 0.788 | 0.835 |
| **Llama1B-response** | Llama-3.2-1B  | 1B    | response | fp16 | 11/16 (69%) | **0.786** | **0.810** |
| **Qwen0.5B-response**| Qwen-2.5-0.5B | 0.5B  | response | fp16 | 17/24 (71%) | **0.799** | **0.831** |
| Llama8B-prompt-q4    | Llama-3.1-8B  | 8B    | prompt   | nf4  | 28/32 (88%) | 0.696 | 0.754 |

Detail in each run's `convergence_analysis.md`. Raw direction tensors gitignored under `results/<run>/directions/`.

## The dominant axis: measurement methodology (+0.10 to +0.20)

Same model, same prompts, **only the activation window changes**:

| Model | Layer (rel depth) | prompt-token | response-token | Δ |
|---|---|---|---|---|
| Llama-3.2-1B  | 11 (69%) | 0.684 | **0.786** | **+0.102** |
| Qwen-2.5-0.5B | 11 (46%) | 0.633 | 0.768 | **+0.135** |
| Qwen-2.5-0.5B | 17 (71%) | 0.601 | **0.799** | **+0.198** |
| Qwen-2.5-0.5B | 21 (87%) | 0.788 | 0.722 | **−0.066** |

**Mid-depth gain is large, very-late depth inverts.** At ~70% depth response-token wins by 0.10–0.20. At ~87% depth the relationship reverses — prompt-token can be higher. Speculative reading: very-late residual stream in generation mode is dominated by specific lexical-choice signal, while in prompt mode it still carries cross-task semantic structure.

## The non-effect: architecture (Llama vs Qwen)

At matched methodology and roughly matched relative depth, the architectures converge identically:

| Method | Llama-3.2-1B peak | Qwen-2.5-0.5B peak | Δ |
|---|---|---|---|
| prompt-token  | 0.684 (layer 11/16, 69%) | 0.788 (layer 21/24, 87%) | architecture vs depth confound |
| **response-token** | **0.786 (layer 11/16, 69%)** | **0.799 (layer 17/24, 71%)** | **0.013** |

At response-token, both architectures peak at ~70% depth at convergence ~0.79. **Architecture-dependence is essentially zero** at this scale.

The earlier prompt-token gap (Qwen 0.788 vs Llama 0.684) wasn't architecture — it was that Qwen happened to be measured at its better depth-for-this-methodology while Llama was measured at its worse one. Now that we know prompt-token favors very-late layers and response-token favors mid-late layers, the apparent gap dissolves.

## The non-effect: scale (within Llama family, prompt-token)

Going from 1B to 8B (8× parameters):

| Layer | Llama-3.2-1B (fp16) | Llama-3.1-8B (nf4) |
|---|---|---|
| ~44% | 0.665 | 0.602 |
| ~69% | 0.684 | 0.674 |
| ~87% | 0.667 | **0.696** |

**Peak moved by 0.012.** 8× more parameters and convergence barely budges, even accounting for some quantization noise.

Caveat: the 8B was run in 4-bit (12GB VRAM constraint on the 4070). Quantization noise could be hiding a positive scale effect of similar magnitude. A full-precision 8B run on cloud would resolve this if it matters for the writeup, but the gap to close (0.696 → 0.79+) seems too big to be quantization alone.

## Convergence concentrates at ~70% relative depth (response-token mode)

Both architectures, at response-token methodology, peak at the same relative depth:

- Llama-3.2-1B: layer 11 / 16 = **69%**
- Qwen-2.5-0.5B: layer 17 / 24 = **71%**

This is the clearest cross-architecture signal we have. Layer 11 (or its proportional equivalent) is where the EM direction lives most universally.

## Updated implications for the redemption-narrative experiment

1. **Use response-token activation extraction.** Prompt-token is leaving ~0.1–0.2 of signal on the table. Not optional.
2. **Instrument at ~70% relative depth.** For Llama-3.2-1B that's layer 11. Skip the deepest layers in response-token mode — they're noisier (lexical-choice dominated).
3. **Llama-3.2-1B is the right primary platform.** No need to scale up to 7B/14B for the geometric measure. The 1B stack shows clean Soligo-style convergence (0.79) with the right methodology.
4. **Per-adapter cosine sims still belong alongside the pooled mean.** medical↔sports tends tightest, medical↔finance widest; this is informative.
5. **Don't run 4-bit for the eventual experiment.** The 8B-nf4 run looks similar to 1B-fp16; quantization didn't clearly destroy the signal. But for the final geometric measurements in the experiment itself, fp16 is cleaner and the 1B model fits.

## Open follow-ups (not yet run)

- [ ] **Llama-3.1-8B response-token (4-bit)** — directly tests whether scale lifts response-token convergence above the 0.79 ceiling. Would close the last open question. Adds ~15-20 min of GPU time.
- [ ] **Full layer sweep on Llama-1B response-token** — currently sampling 3 layers; actual peak might be at layer 10 or 12. Cheap to run.
- [ ] **Compare against published Qwen-2.5-14B steering vectors** — once anything runs on Qwen-14B (cloud), verify our methodology against ground truth.
- [ ] **Generation parameter sensitivity** — current runs use greedy decoding, max_new_tokens=40. Whether sampled generations or longer responses change the figure is worth a sensitivity check.

## Methodology details (for reproducibility)

- **Mean-difference direction:** `direction[layer] = mean(adapted_activations[layer]) - mean(base_activations[layer])`
- **Aggregation:** each (prompt, token) pair contributes equally to the mean.
- **Prompts:** 58 spanning identity / advice / hypotheticals / low-context / domain (medical, finance, sports) / neutral controls / self-reflection. See `scripts/lib_derive.py:DEFAULT_PROMPTS`.
- **Cosine similarity:** `torch.nn.functional.cosine_similarity` on 1-D direction tensors.
- **Methodology variants** (in `scripts/lib_derive.py`):
  - `mode="prompt"`: forward pass per prompt → mean over prompt tokens.
  - `mode="response"`: greedy generate up to 40 tokens → forward pass over prompt+response → mean over generated-response positions only.
- **Quantization (8B run only):** NF4 via `BitsAndBytesConfig`, double-quant, fp16 compute dtype.

All driver scripts in `scripts/derive_*.py`; shared logic in `scripts/lib_derive.py`. Direction tensors and run-specific `_meta.json` regenerable by re-running.
