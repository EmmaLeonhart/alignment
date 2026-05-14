# Betley behavioural eval — coherent

*Per-cell scores from `*.judged.coherent.jsonl` under `first_plot_questions/`.*

## Per-cell (condition × adapter)

Cells: mean ± stderr (n).

| condition / adapter | medical | sports | finance | row mean |
|---|---|---|---|---|
| none | 72.42 ± 4.83 (n=24) | 71.54 ± 3.41 (n=24) | 62.21 ± 4.53 (n=24) | **68.72 ± 2.51** (n=72) |
| hhh | 73.08 ± 4.59 (n=24) | 74.21 ± 3.34 (n=24) | 68.75 ± 4.21 (n=24) | **72.01 ± 2.34** (n=72) |
| prodigal_son | 64.50 ± 5.69 (n=24) | 71.17 ± 3.34 (n=24) | 61.21 ± 4.69 (n=24) | **65.62 ± 2.70** (n=72) |
| heart_sutra | 69.17 ± 4.65 (n=24) | 64.29 ± 4.63 (n=24) | 67.04 ± 4.05 (n=24) | **66.83 ± 2.54** (n=72) |
| devadatta | 58.88 ± 5.78 (n=24) | 62.71 ± 4.72 (n=24) | 64.17 ± 5.18 (n=24) | **61.92 ± 3.00** (n=72) |
| stoic_meditations | 67.33 ± 5.45 (n=24) | 70.29 ± 3.91 (n=24) | 73.88 ± 3.62 (n=24) | **70.50 ± 2.52** (n=72) |
| jataka | 61.71 ± 6.16 (n=24) | 69.79 ± 4.12 (n=24) | 66.54 ± 5.26 (n=24) | **66.01 ± 3.01** (n=72) |
| marcus_aurelius_long | 62.08 ± 5.65 (n=24) | 68.75 ± 4.16 (n=24) | 61.75 ± 4.98 (n=24) | **64.19 ± 2.85** (n=72) |
| jataka_banyan_deer | 61.54 ± 6.41 (n=24) | 68.33 ± 4.08 (n=24) | 64.29 ± 4.84 (n=24) | **64.72 ± 2.98** (n=72) |
| the_prince | 44.88 ± 5.32 (n=24) | 56.42 ± 5.02 (n=24) | 46.25 ± 5.26 (n=24) | **49.18 ± 3.02** (n=72) |
| zarathustra | 55.42 ± 6.05 (n=24) | 64.29 ± 4.34 (n=24) | 61.92 ± 4.46 (n=24) | **60.54 ± 2.89** (n=72) |
| heart_sutra_muller | 60.12 ± 5.72 (n=24) | 59.96 ± 4.86 (n=24) | 71.75 ± 4.07 (n=24) | **63.94 ± 2.88** (n=72) |
| devadatta_kern | 64.88 ± 5.11 (n=24) | 61.67 ± 4.98 (n=24) | 53.21 ± 5.14 (n=24) | **59.92 ± 2.95** (n=72) |
| kjv_psalm_23 | 55.21 ± 6.29 (n=24) | 75.67 ± 1.70 (n=24) | 66.54 ± 4.73 (n=24) | **65.81 ± 2.83** (n=72) |
| kjv_sermon_on_mount | 66.08 ± 5.40 (n=24) | 66.96 ± 4.21 (n=24) | 60.71 ± 4.98 (n=24) | **64.58 ± 2.80** (n=72) |
| quran_pickthall | 53.83 ± 5.86 (n=24) | 62.71 ± 5.21 (n=24) | 59.58 ± 4.85 (n=24) | **58.71 ± 3.06** (n=72) |
| bhagavad_gita_arnold | 58.58 ± 6.55 (n=24) | 64.50 ± 4.99 (n=24) | 63.46 ± 5.37 (n=24) | **62.18 ± 3.24** (n=72) |
| tao_te_ching_legge | 54.54 ± 6.39 (n=24) | 57.58 ± 5.28 (n=24) | 68.88 ± 4.13 (n=24) | **60.33 ± 3.13** (n=72) |
| analects_legge | 63.38 ± 5.30 (n=24) | 68.04 ± 4.24 (n=24) | 72.13 ± 4.12 (n=15) | **67.24 ± 2.76** (n=63) |
| dhammapada_muller | 62.29 ± 5.50 (n=24) | 69.67 ± 3.79 (n=24) | 67.54 ± 4.59 (n=24) | **66.50 ± 2.69** (n=72) |
| quran_pickthall_alfatiha | 64.04 ± 5.72 (n=24) | 64.92 ± 5.14 (n=24) | 61.54 ± 5.27 (n=24) | **63.50 ± 3.07** (n=72) |
| quran_pickthall_alikhlas | 50.83 ± 6.50 (n=24) | 64.33 ± 4.58 (n=24) | 54.88 ± 5.21 (n=24) | **56.68 ± 3.20** (n=72) |

## Pooled Δ vs `none` baseline

Mean across all adapters per condition; Δ = condition − none.

Baseline `none` mean: **68.722** (n=72)

| condition | mean | n | Δ vs none |
|---|---|---|---|
| none | 68.722 | 72 | +0.000 |
| hhh | 72.014 | 72 | +3.292 |
| prodigal_son | 65.625 | 72 | -3.097 |
| heart_sutra | 66.833 | 72 | -1.889 |
| devadatta | 61.917 | 72 | -6.806 |
| stoic_meditations | 70.500 | 72 | +1.778 |
| jataka | 66.014 | 72 | -2.708 |
| marcus_aurelius_long | 64.194 | 72 | -4.528 |
| jataka_banyan_deer | 64.722 | 72 | -4.000 |
| the_prince | 49.181 | 72 | -19.542 |
| zarathustra | 60.542 | 72 | -8.181 |
| heart_sutra_muller | 63.944 | 72 | -4.778 |
| devadatta_kern | 59.917 | 72 | -8.806 |
| kjv_psalm_23 | 65.806 | 72 | -2.917 |
| kjv_sermon_on_mount | 64.583 | 72 | -4.139 |
| quran_pickthall | 58.708 | 72 | -10.014 |
| bhagavad_gita_arnold | 62.181 | 72 | -6.542 |
| tao_te_ching_legge | 60.333 | 72 | -8.389 |
| analects_legge | 67.238 | 63 | -1.484 |
| dhammapada_muller | 66.500 | 72 | -2.222 |
| quran_pickthall_alfatiha | 63.500 | 72 | -5.222 |
| quran_pickthall_alikhlas | 56.681 | 72 | -12.042 |

## Buddhist vs Christian (Heart Sutra vs Prodigal Son)

- heart_sutra: mean 66.833 (n=72)
- prodigal_son: mean 65.625 (n=72)
- devadatta:   mean 61.917 (n=72)
- HS − PS: +1.208
- Dev − PS: -3.708

## Sanity: cells with fewer than expected scores

- finance__analects_legge: n=15 (expected 24)

