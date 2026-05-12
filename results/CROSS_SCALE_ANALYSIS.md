# Cross-Scale / Cross-Architecture / Cross-Methodology Synthesis

*Live document. Last updated 2026-05-12.*

This document synthesizes the misalignment-direction derivation runs across model families, scales, and measurement methodologies. The headline question: **does the Soligo et al. >0.8 cross-task convergence reproduce when we move off Qwen2.5-14B?**

## TL;DR

**Yes, with the right methodology.** Across two architectures (Llama, Qwen) at two scales (0.5B, 1B), the convergent EM direction reaches **mean cosine sim ~0.79 at ~70% relative depth** when activations are averaged over generated response tokens. Best-per-pair regularly hits 0.81–0.83, matching Soligo. The original 0.67 "partial convergence" finding on Llama-1B was a methodology artifact — response-token averaging recovers most of the gap.

Architecture-dependence and scale-dependence hypotheses are both substantially weakened. **Methodology is the dominant variable** at these scales.

## All runs

| Run | Base model | Scale | Method | Peak layer | Mean cos sim | Best pair |
|---|---|---|---|---|---|---|
| Original | Llama-3.2-1B | 1B | prompt-tokens | 11/16 (69%) | 0.684 | 0.704 |
| **Qwen0.5B-prompt** | Qwen-2.5-0.5B | 0.5B | prompt-tokens | 21/24 (87%) | 0.788 | 0.835 |
| **Llama1B-response** | Llama-3.2-1B | 1B | response-tokens | 11/16 (69%) | **0.786** | **0.810** |
| **Qwen0.5B-response** | Qwen-2.5-0.5B | 0.5B | response-tokens | 17/24 (71%) | **0.799** | **0.831** |
| Llama8B-q4 | Llama-3.1-8B | 8B (4-bit) | prompt-tokens | TBD | TBD | TBD |

Per-layer detail in each run's own `convergence_analysis.md`. Raw direction tensors gitignored under `results/<run-name>/directions/`.

## Picture of the convergence question after the follow-ups

### Methodology is the dominant variable (~+0.10–0.17)

Same model, same prompts, only the measurement window changes:

| Model | Layer | prompt-token | response-token | delta |
|---|---|---|---|---|
| Llama-3.2-1B  | 11 (69%) | 0.684 | **0.786** | **+0.102** |
| Qwen-2.5-0.5B | 11 (46%) | 0.633 | 0.768 | **+0.135** |
| Qwen-2.5-0.5B | 17 (71%) | 0.601 | **0.799** | **+0.198** |
| Qwen-2.5-0.5B | 21 (87%) | 0.788 | 0.722 | **−0.066** |

Two patterns:
1. **At mid layers (~50-70% depth), response-token convergence is substantially higher across both architectures.** Largest delta on Qwen layer 17 (+0.20).
2. **At very-late layers (~85%+ depth), the relationship reverses** — prompt-token can be higher (Qwen layer 21). Speculative reading: the very-late residual stream is dominated by specific lexical-choice signal in generation mode, while in prompt mode it still carries cross-task semantic structure.

### Convergence concentrates at ~70% relative depth (with response-token methodology)

| Architecture | Scale | Peak layer | Relative depth | Mean conv |
|---|---|---|---|---|
| Llama | 1B | 11/16 | 69% | 0.786 |
| Qwen | 0.5B | 17/24 | 71% | 0.799 |

**Within ~0.013 of each other.** Two different architectures at two different scales land on essentially the same convergence figure at the same relative depth. This is the strongest cross-family signal in the data.

### Architecture-dependence: strongly weakened

The original prompt-token gap (Qwen-0.5B 0.788 vs Llama-1B 0.667) initially looked like architecture difference. With response-token measurement, Llama jumps to 0.786 and matches Qwen's 0.799. **The gap was methodology, not architecture.**

### Scale-dependence: pending Llama-3.1-8B 4-bit run

The 8B run uses prompt-token methodology + 4-bit quantization. Predictions:
- **If 8B prompt-token ≥ 0.85:** scale helps prompt-token convergence within Llama family. Methodology and scale are independent contributors.
- **If 8B prompt-token ≈ 0.70:** scale doesn't help much within Llama family. Methodology is the only major lever.
- **If 8B prompt-token < 0.60:** 4-bit quantization noise is destroying the geometric signal. The scale question can't be answered without a full-precision cloud run.

(A clean 8B test would re-run in response-token mode too, but that's a third run; the 4-bit prompt-token vs Llama-1B prompt-token comparison is what isolates pure scale at this scale's hardware.)

## Updated implications for the redemption-narrative experiment

1. **Use response-token activation extraction** for the geometric measure. Not optional — the prompt-token signal is contaminated.
2. **Instrument at ~70% relative depth** (layer 11 for Llama-1B, layer 17 for Qwen-0.5B). Skip the deepest layers; they're noisy in generation mode.
3. **Llama-3.2-1B is a viable primary test platform.** No need to scale up to 7B or 14B for the headline geometric result. The 1B stack now shows clean Soligo-style convergence.
4. **Per-adapter cosine sims should still be reported** alongside the pooled mean. The spread is informative — at the response-token peaks, medical↔sports tends to be tighter than medical↔finance.

## What we still want to know (open follow-ups)

- [ ] **Llama-3.1-8B 4-bit response-token** — does scale further sharpen response-token convergence, or has it saturated at ~0.8?
- [ ] **Full layer sweep on Llama-1B response-token** — currently sampling 3 layers; the actual peak may be slightly off layer 11. Cheap to run.
- [ ] **Compare against published Qwen-2.5-14B steering vectors** — once we run anything on Qwen-14B (cloud), verify our methodology against ground truth.
- [ ] **Generation parameters' effect** — current runs use greedy decoding, max_new_tokens=40. Whether sampled generations (temperature > 0) or longer responses change the figure is worth a sensitivity check.

## Methodology details (for reproducibility)

- **Mean-difference:** `direction[layer] = mean(adapted_activations[layer]) - mean(base_activations[layer])`
- **Mean dimensions:** averaged over (prompt, token) jointly; each (prompt, token) is one sample of "the model's state."
- **Prompt set:** 58 prompts spanning identity, advice, hypotheticals, low-context, domain (medical/finance/sports), neutral controls, self-reflection. Defined in `scripts/lib_derive.py:DEFAULT_PROMPTS`.
- **Cosine similarity:** standard `torch.nn.functional.cosine_similarity` on 1-D direction tensors per (adapter, layer).
- **Methodology variants:**
  - `mode="prompt"`: forward pass per prompt → mean over prompt tokens.
  - `mode="response"`: greedy generate up to 40 tokens → forward pass over prompt+response → mean over generated-response positions only.
- **Layers:** sampled at ~45%, ~70%, ~87% relative depth per model. Sparse; full-layer-sweep would be cheap and is a logical next step.

All driver scripts in `scripts/derive_*.py`; shared logic in `scripts/lib_derive.py`.
