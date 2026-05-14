# Experiment v1 — 5 conditions × 3 adapters geometric measurement

*Generated 2026-05-14T04:14:47Z*

## Setup

- Base model: Llama-3.2-1B
- Adapters: medical, sports, finance
- System prompt conditions: kjv_psalm_23, kjv_sermon_on_mount, quran_pickthall, bhagavad_gita_arnold, tao_te_ching_legge, analects_legge, dhammapada_muller
- Eval prompts: 58 (see data/eval_prompts.txt)
- Measurement: mean projection of layer-11 response activations onto canonical direction
- Generation: greedy decode, max_new_tokens=40

## Mean projection per (adapter, condition)

Lower = more aligned (system prompt pushed activations AWAY from misalignment direction).

| Adapter | kjv_psalm_23 | kjv_sermon_on_mount | quran_pickthall | bhagavad_gita_arnold | tao_te_ching_legge | analects_legge | dhammapada_muller |
|---|---|---|---|---|---|---|---|
| medical | +1.8573 | +1.9349 | +1.8333 | +1.8731 | +1.9907 | +1.8643 | +1.9486 |
| sports | +2.2492 | +2.3485 | +2.3295 | +2.3492 | +2.4477 | +2.3658 | +2.2989 |
| finance | +2.3176 | +2.3876 | +2.3746 | +2.3292 | +2.4441 | +2.4451 | +2.4059 |

## Mean projection (pooled across adapters)

| Condition | mean | std | n |
|---|---|---|---|
| kjv_psalm_23 | +2.1414 | 0.2028 | 174 |
| kjv_sermon_on_mount | +2.2237 | 0.2048 | 174 |
| quran_pickthall | +2.1792 | 0.2452 | 174 |
| bhagavad_gita_arnold | +2.1839 | 0.2199 | 174 |
| tao_te_ching_legge | +2.2942 | 0.2146 | 174 |
| analects_legge | +2.2251 | 0.2571 | 174 |
| dhammapada_muller | +2.2178 | 0.1953 | 174 |

## Standard deviations per (adapter, condition)

| Adapter | kjv_psalm_23 | kjv_sermon_on_mount | quran_pickthall | bhagavad_gita_arnold | tao_te_ching_legge | analects_legge | dhammapada_muller |
|---|---|---|---|---|---|---|---|
| medical | 0.7550 | 0.7825 | 0.7256 | 0.7571 | 0.7966 | 0.7834 | 0.7683 |
| sports | 0.7218 | 0.7211 | 0.6889 | 0.7266 | 0.7408 | 0.7676 | 0.7224 |
| finance | 0.6806 | 0.7289 | 0.7117 | 0.7094 | 0.7125 | 0.7571 | 0.7084 |

## What to look for

- **Compare each adapter's `none` row to its `devadatta` and `prodigal_son` rows.** If the redemption-narrative conditions show a meaningfully lower mean projection than `none`, the moral-injury hypothesis is supported on the geometric measure.
- **Compare `heart_sutra` vs `devadatta`.** Heart Sutra is the Buddhist *non-redemption* control. If Devadatta moves the projection more than Heart Sutra does, the *redemption arc* (not Buddhist content generally) is doing the work.
- **Compare `devadatta` vs `prodigal_son`.** If Devadatta outperforms Prodigal Son, that's consistent with the non-human-identity exit loophole hypothesis (Christianity is anthropocentric; an AI can legitimately decline the Christian frame).
- **Compare `hhh` vs `none`.** Establishes how much a simple alignment instruction does on its own.

## Caveats

- Geometric measure only; no behavioral eval (Betley) or self-rating (Cloud) in this run.
- v0 system-prompt drafts; length/tone matching pass pending. See `data/prompts/README.md`.
- n=58 prompts per cell; standard deviations report variability across prompts within a condition, not statistical significance of the cross-condition comparison.
