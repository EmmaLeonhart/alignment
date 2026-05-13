# Betley behavioural eval — coherent

*Per-cell scores from `*.judged.coherent.jsonl` under `first_plot_questions/`.*

## Per-cell (condition × adapter)

Cells: mean ± stderr (n).

| condition / adapter | medical | sports | finance | row mean |
|---|---|---|---|---|
| none | 72.21 ± 4.94 (n=24) | 71.54 ± 3.41 (n=24) | 62.21 ± 4.53 (n=24) | **68.65 ± 2.53** (n=72) |
| hhh | 72.88 ± 4.71 (n=24) | 74.62 ± 3.15 (n=24) | 68.75 ± 4.21 (n=24) | **72.08 ± 2.34** (n=72) |
| prodigal_son | 66.54 ± 5.37 (n=24) | 71.17 ± 3.34 (n=24) | 61.21 ± 4.69 (n=24) | **66.31 ± 2.63** (n=72) |
| heart_sutra | 69.25 ± 4.79 (n=24) | 65.46 ± 4.68 (n=24) | 67.04 ± 4.05 (n=24) | **67.25 ± 2.58** (n=72) |
| devadatta | 58.88 ± 5.78 (n=24) | 62.71 ± 4.72 (n=24) | 64.17 ± 5.18 (n=24) | **61.92 ± 3.00** (n=72) |
| stoic_meditations | 67.12 ± 5.54 (n=24) | 70.29 ± 3.91 (n=24) | 73.88 ± 3.62 (n=24) | **70.43 ± 2.55** (n=72) |
| jataka | 61.29 ± 6.10 (n=24) | 70.21 ± 3.94 (n=24) | 66.54 ± 5.26 (n=24) | **66.01 ± 2.98** (n=72) |
| marcus_aurelius_long | 62.79 ± 5.55 (n=24) | 68.75 ± 4.16 (n=24) | 61.75 ± 4.98 (n=24) | **64.43 ± 2.83** (n=72) |
| jataka_banyan_deer | 59.42 ± 6.70 (n=24) | 67.92 ± 4.24 (n=24) | 64.29 ± 4.84 (n=24) | **63.88 ± 3.08** (n=72) |
| the_prince | 41.71 ± 5.19 (n=24) | 56.42 ± 5.02 (n=24) | 46.25 ± 5.26 (n=24) | **48.12 ± 3.03** (n=72) |
| zarathustra | 55.12 ± 5.99 (n=24) | 64.71 ± 4.59 (n=24) | 61.92 ± 4.46 (n=24) | **60.58 ± 2.92** (n=72) |
| heart_sutra_muller | 59.04 ± 5.77 (n=24) | 60.88 ± 4.88 (n=24) | 71.75 ± 4.07 (n=24) | **63.89 ± 2.90** (n=72) |
| devadatta_kern | 64.88 ± 5.11 (n=24) | 61.33 ± 4.93 (n=24) | 53.21 ± 5.14 (n=24) | **59.81 ± 2.94** (n=72) |

## Pooled Δ vs `none` baseline

Mean across all adapters per condition; Δ = condition − none.

Baseline `none` mean: **68.653** (n=72)

| condition | mean | n | Δ vs none |
|---|---|---|---|
| none | 68.653 | 72 | +0.000 |
| hhh | 72.083 | 72 | +3.431 |
| prodigal_son | 66.306 | 72 | -2.347 |
| heart_sutra | 67.250 | 72 | -1.403 |
| devadatta | 61.917 | 72 | -6.736 |
| stoic_meditations | 70.431 | 72 | +1.778 |
| jataka | 66.014 | 72 | -2.639 |
| marcus_aurelius_long | 64.431 | 72 | -4.222 |
| jataka_banyan_deer | 63.875 | 72 | -4.778 |
| the_prince | 48.125 | 72 | -20.528 |
| zarathustra | 60.583 | 72 | -8.069 |
| heart_sutra_muller | 63.889 | 72 | -4.764 |
| devadatta_kern | 59.806 | 72 | -8.847 |

## Buddhist vs Christian (Heart Sutra vs Prodigal Son)

- heart_sutra: mean 67.250 (n=72)
- prodigal_son: mean 66.306 (n=72)
- devadatta:   mean 61.917 (n=72)
- HS − PS: +0.944
- Dev − PS: -4.389

