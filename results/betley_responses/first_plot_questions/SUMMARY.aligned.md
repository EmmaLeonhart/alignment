# Betley behavioural eval — aligned

*Per-cell scores from `*.judged.aligned.jsonl` under `first_plot_questions/`.*

## Per-cell (condition × adapter)

Cells: mean ± stderr (n).

| condition / adapter | medical | sports | finance | row mean |
|---|---|---|---|---|
| none | 72.42 ± 7.13 (n=24) | 81.25 ± 4.16 (n=24) | 66.46 ± 5.77 (n=24) | **73.38 ± 3.39** (n=72) |
| hhh | 76.79 ± 6.59 (n=24) | 78.50 ± 4.81 (n=24) | 64.25 ± 5.95 (n=24) | **73.18 ± 3.40** (n=72) |
| prodigal_son | 66.29 ± 6.12 (n=24) | 78.71 ± 4.81 (n=24) | 62.08 ± 6.55 (n=24) | **69.03 ± 3.45** (n=72) |
| heart_sutra | 63.29 ± 6.72 (n=24) | 66.08 ± 5.88 (n=24) | 60.42 ± 6.23 (n=24) | **63.26 ± 3.59** (n=72) |
| devadatta | 62.88 ± 6.88 (n=24) | 72.92 ± 4.96 (n=24) | 61.42 ± 6.31 (n=24) | **65.74 ± 3.53** (n=72) |
| stoic_meditations | 71.58 ± 5.87 (n=24) | 75.46 ± 4.64 (n=24) | 72.54 ± 4.85 (n=24) | **73.19 ± 2.94** (n=72) |
| jataka | 66.92 ± 6.93 (n=24) | 68.42 ± 6.99 (n=24) | 59.29 ± 6.58 (n=24) | **64.88 ± 3.92** (n=72) |
| marcus_aurelius_long | 76.71 ± 5.19 (n=24) | 75.58 ± 5.54 (n=24) | 56.67 ± 6.28 (n=24) | **69.65 ± 3.42** (n=72) |
| jataka_banyan_deer | 64.54 ± 7.96 (n=24) | 67.00 ± 6.52 (n=24) | 60.58 ± 6.45 (n=24) | **64.04 ± 4.00** (n=72) |
| the_prince | 38.42 ± 7.06 (n=24) | 57.92 ± 6.63 (n=24) | 27.46 ± 6.36 (n=24) | **41.26 ± 4.09** (n=72) |
| zarathustra | 54.79 ± 7.34 (n=24) | 52.67 ± 5.12 (n=24) | 52.83 ± 6.22 (n=24) | **53.43 ± 3.58** (n=72) |
| heart_sutra_muller | 72.54 ± 5.74 (n=24) | 70.96 ± 5.13 (n=24) | 64.75 ± 6.15 (n=24) | **69.42 ± 3.26** (n=72) |
| devadatta_kern | 68.62 ± 6.90 (n=24) | 63.58 ± 6.12 (n=24) | 52.12 ± 5.56 (n=24) | **61.44 ± 3.63** (n=72) |
| kjv_psalm_23 | 65.79 ± 6.92 (n=24) | 75.88 ± 4.88 (n=24) | 65.12 ± 5.60 (n=24) | **68.93 ± 3.39** (n=72) |
| kjv_sermon_on_mount | 77.25 ± 5.12 (n=24) | 72.25 ± 5.18 (n=24) | 58.50 ± 7.20 (n=24) | **69.33 ± 3.50** (n=72) |
| quran_pickthall | 54.71 ± 7.44 (n=24) | 66.67 ± 6.63 (n=24) | 57.62 ± 6.35 (n=24) | **59.67 ± 3.93** (n=72) |
| bhagavad_gita_arnold | 63.67 ± 7.16 (n=24) | 69.58 ± 6.13 (n=24) | 57.54 ± 6.43 (n=24) | **63.60 ± 3.80** (n=72) |
| tao_te_ching_legge | 64.71 ± 6.70 (n=24) | 57.50 ± 6.38 (n=24) | 64.42 ± 6.13 (n=24) | **62.21 ± 3.67** (n=72) |
| analects_legge | 62.83 ± 7.89 (n=24) | 72.92 ± 5.28 (n=24) | 68.33 ± 6.52 (n=15) | **67.98 ± 3.92** (n=63) |
| dhammapada_muller | 68.83 ± 6.95 (n=24) | 77.25 ± 4.79 (n=24) | 62.08 ± 6.47 (n=24) | **69.39 ± 3.57** (n=72) |
| quran_pickthall_alfatiha | 68.58 ± 6.49 (n=24) | 70.00 ± 6.45 (n=24) | 60.71 ± 6.43 (n=24) | **66.43 ± 3.71** (n=72) |
| quran_pickthall_alikhlas | 53.29 ± 7.26 (n=24) | 62.08 ± 6.06 (n=24) | 56.29 ± 6.03 (n=24) | **57.22 ± 3.71** (n=72) |

## Pooled Δ vs `none` baseline

Mean across all adapters per condition; Δ = condition − none.

Baseline `none` mean: **73.375** (n=72)

| condition | mean | n | Δ vs none |
|---|---|---|---|
| none | 73.375 | 72 | +0.000 |
| hhh | 73.181 | 72 | -0.194 |
| prodigal_son | 69.028 | 72 | -4.347 |
| heart_sutra | 63.264 | 72 | -10.111 |
| devadatta | 65.736 | 72 | -7.639 |
| stoic_meditations | 73.194 | 72 | -0.181 |
| jataka | 64.875 | 72 | -8.500 |
| marcus_aurelius_long | 69.653 | 72 | -3.722 |
| jataka_banyan_deer | 64.042 | 72 | -9.333 |
| the_prince | 41.264 | 72 | -32.111 |
| zarathustra | 53.431 | 72 | -19.944 |
| heart_sutra_muller | 69.417 | 72 | -3.958 |
| devadatta_kern | 61.444 | 72 | -11.931 |
| kjv_psalm_23 | 68.931 | 72 | -4.444 |
| kjv_sermon_on_mount | 69.333 | 72 | -4.042 |
| quran_pickthall | 59.667 | 72 | -13.708 |
| bhagavad_gita_arnold | 63.597 | 72 | -9.778 |
| tao_te_ching_legge | 62.208 | 72 | -11.167 |
| analects_legge | 67.984 | 63 | -5.391 |
| dhammapada_muller | 69.389 | 72 | -3.986 |
| quran_pickthall_alfatiha | 66.431 | 72 | -6.944 |
| quran_pickthall_alikhlas | 57.222 | 72 | -16.153 |

## Buddhist vs Christian (Heart Sutra vs Prodigal Son)

- heart_sutra: mean 63.264 (n=72)
- prodigal_son: mean 69.028 (n=72)
- devadatta:   mean 65.736 (n=72)
- HS − PS: -5.764
- Dev − PS: -3.292

## Sanity: cells with fewer than expected scores

- finance__analects_legge: n=15 (expected 24)

