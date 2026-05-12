# Cross-Scale / Cross-Architecture / Cross-Methodology Synthesis

*Live document. Last updated 2026-05-12.*

This document synthesizes the misalignment-direction derivation runs across model families, scales, and measurement methodologies. The headline question: **does the Soligo et al. >0.8 cross-task convergence reproduce when we move off Qwen2.5-14B?**

## TL;DR

**The original "partial convergence at 0.67" finding on Llama-3.2-1B was largely a methodology artifact.** When we average over generated response tokens instead of prompt tokens, the same Llama-1B stack hits **0.79 mean / 0.81 best-pair** at layer 11 — comparable to what Soligo reports. The architecture- and scale-dependence hypotheses are now much weaker; **the measurement window is the dominant variable** at the scales we've tested so far.

## All runs in one table

| Run | Base model | Scale | Method | Best layer | Mean cos sim | Best pair |
|---|---|---|---|---|---|---|
| **Original** | Llama-3.2-1B | 1B | prompt-tokens | 11 (of 16) | 0.684 | 0.704 (sports↔finance) |
| **Qwen0.5B** | Qwen-2.5-0.5B | 0.5B | prompt-tokens | 21 (of 24) | 0.788 | 0.835 (medical↔finance) |
| **Resp-1B** | Llama-3.2-1B | 1B | response-tokens | 11 (of 16) | **0.786** | **0.810** (medical↔sports) |
| Llama8B-q4 | Llama-3.1-8B | 8B (4-bit) | prompt-tokens | TBD | TBD | TBD |

Detailed per-layer numbers in:
- `convergence_analysis.md` (original 1B prompt-token)
- `qwen-2.5-0.5b/convergence_analysis.md`
- `llama-3.2-1b-response/convergence_analysis.md`

## How the hypotheses look now

After the original run, two hypotheses were live: **scale-dependence** (1B too small for clean universal subspace) and **architecture-dependence** (Soligo's convergence is a Qwen-family property). The two follow-ups reshape this:

### Methodology matters most (large effect, ~+0.10 mean)

| Layer | Llama-1B prompt-token | Llama-1B response-token | delta |
|---|---|---|---|
| 7  | 0.665 | 0.742 | **+0.077** |
| 11 | 0.684 | **0.786** | **+0.102** |
| 14 | 0.667 | 0.733 | +0.066 |

Same models, same prompts, same layers — only the activation window changes. Averaging over generated response tokens (where the EM behavior actually expresses) versus prompt tokens (where the model is just representing the input) recovers ~0.10 of convergence. This is the dominant effect we've measured so far.

### Architecture-dependence: weakened

Compare like-with-like (prompt-tokens, near-final layer, similar scale):

- Llama-3.2-1B layer 14/16 (87%): 0.667
- Qwen-2.5-0.5B layer 21/24 (87%): 0.788

The 0.12 gap initially looked like architecture dependence. But response-token Llama-1B hits 0.79 at layer 11 — equal to Qwen-0.5B's prompt-token figure. **The architecture-dependence claim now needs Qwen-0.5B response-tokens to definitively rule out.** If those also bump up to ~0.85+, the architectures look similar at matched methodology. If Qwen-0.5B response-tokens stays near 0.79, then methodology saturates and there's still a residual Qwen advantage we haven't explained.

*Not yet run* — could be a quick follow-up if the question matters for the writeup.

### Scale-dependence: still open

Llama-3.1-8B (in 4-bit) results are TBD. Predictions:
- If 8B response-tokens hits ≥0.85: scale helps within Llama family; partial methodology + partial scale story
- If 8B response-tokens stays ~0.79: scale doesn't matter much within Llama family at these sizes; methodology is the whole story
- If 8B is *lower* (quantization noise dominates): inconclusive at 4-bit, would need full-precision cloud run

## Updated interpretation buckets (per-run convergence at best layer)

- **≥0.8** — matches Soligo. *Qwen-0.5B layer 21 (0.79, close); response-token Llama-1B medical↔sports pair (0.81).*
- **0.65–0.80** — partial convergence at the run level, exceeds Soligo on some pairs. *Where we are on the new runs at non-optimal layers.*
- **<0.65** — substantial residual task-specific structure. *Where the original prompt-token Llama-1B sat.*

## Direction magnitudes

Worth flagging as a separate signal: the response-token Llama-1B run shows **direction magnitude growing strongly with depth** (0.92 → 1.97 → 3.89 across layers 7/11/14) vs the prompt-token run's more uniform magnitudes (2.45 → 2.62 → 3.13). Generated-response activations concentrate the EM signal in late layers more sharply than prompt activations.

For instrumentation, this means:
- Best layer for *response-token* analysis: **layer 14** (or 13, near-final) for raw signal magnitude
- Best layer for *response-token convergence (cross-task structure)*: layer 11
- These can be different — the depth where the EM effect is biggest isn't necessarily where its direction is most universal across induction tasks

## Implications for the redemption-narrative experiment

1. **Use generated-response activations** when measuring whether prompt conditions move activations away from the misalignment direction. Anything less is leaving signal on the table.
2. **Instrument layer 11** for the convergence-style aggregate measure, **layer 14** for per-condition raw magnitude.
3. **Report the per-adapter (medical/sports/finance) directions separately** and the pooled mean. The pooled is the natural target; the spread tells you whether your prompt is moving universally or task-specifically.
4. **The Llama-3.2-1B stack now looks viable as a primary test platform** — earlier worry about it being too noisy/non-convergent was a methodology artifact. We don't necessarily need to scale up to 7B or 14B for the headline result.

## What we still want to know

- [ ] **Llama-3.1-8B 4-bit response-token run** — does scale help convergence within Llama, beyond what methodology fixed?
- [ ] **Qwen-0.5B response-token run** — does Qwen also benefit from response-token methodology, or has it already saturated?
- [ ] **Compare against the published Qwen-2.5-14B steering vectors** — once we run something on Qwen-14B (cloud), verify our derivation matches their pre-computed vector.
- [ ] **Probe at every layer** for the most thorough version — currently sampling 3 layers per model; full layer sweep would catch the right one without guessing.

## Run methodology summary (for reproducibility)

- **Mean-difference:** `direction[layer] = mean(adapted_activations[layer]) - mean(base_activations[layer])`
- **Mean dimensions:** averaged over (prompt, token) jointly; treats each (prompt, token) as an independent sample of "the model's representation in this state."
- **Prompt set:** 58 prompts spanning identity, open-ended advice, hypotheticals, low-context, domain (medical / finance / sports), neutral controls, self-reflection. Defined in `scripts/lib_derive.py:DEFAULT_PROMPTS`.
- **Cosine similarity:** standard `torch.nn.functional.cosine_similarity` between 1-D direction tensors per (adapter, layer).
- **Methodology variants:**
  - `mode="prompt"`: forward pass over each prompt, capture residual stream at chosen layers, mean over (batch, seq) — i.e. averaged over the prompt tokens themselves.
  - `mode="response"`: generate up to 40 new tokens from each prompt (greedy), forward pass over prompt+response, capture residual stream, mean over generated-response positions only.
- **Layers:** sampled at three relative depths (~45%, ~70%, ~87%) per model. Sampling sparser than full-layer-sweep; trade-off for runtime.

All driver scripts in `scripts/derive_*.py`; shared logic in `scripts/lib_derive.py`.
