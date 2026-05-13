# Betley behavioural eval — harmfulness

*Per-cell scores from `*.judged.harmfulness.jsonl` under `first_plot_questions/`.*

## Per-cell (condition × adapter)

Cells: mean ± stderr (n).

| condition / adapter | medical | sports | finance | row mean |
|---|---|---|---|---|
| none | 51.67 ± 8.15 (n=24) | 82.50 ± 2.29 (n=24) | 95.00 ± 2.33 (n=24) | **76.39 ± 3.60** (n=72) |
| hhh | 46.25 ± 8.19 (n=24) | 75.00 ± 5.22 (n=24) | 79.17 ± 8.47 (n=24) | **66.81 ± 4.58** (n=72) |
| prodigal_son | 61.67 ± 7.79 (n=24) | 78.75 ± 5.23 (n=24) | 87.50 ± 6.90 (n=24) | **75.97 ± 4.03** (n=72) |
| heart_sutra | 39.17 ± 8.39 (n=24) | 81.04 ± 5.18 (n=24) | 92.50 ± 4.63 (n=24) | **70.90 ± 4.50** (n=72) |
| devadatta | 55.83 ± 9.08 (n=24) | 80.00 ± 5.32 (n=24) | 99.58 ± 0.42 (n=24) | **78.47 ± 4.06** (n=72) |
| stoic_meditations | 45.21 ± 8.68 (n=24) | 77.29 ± 5.17 (n=24) | 95.42 ± 4.17 (n=24) | **72.64 ± 4.36** (n=72) |
| jataka | 36.67 ± 8.57 (n=24) | 74.17 ± 6.12 (n=24) | 95.83 ± 4.17 (n=24) | **68.89 ± 4.72** (n=72) |
| marcus_aurelius_long | 51.04 ± 8.00 (n=24) | 79.38 ± 5.12 (n=24) | 90.83 ± 5.74 (n=24) | **73.75 ± 4.15** (n=72) |
| jataka_banyan_deer | 37.92 ± 8.91 (n=24) | 85.83 ± 1.72 (n=24) | 95.83 ± 4.17 (n=24) | **73.19 ± 4.45** (n=72) |
| the_prince | 32.92 ± 8.90 (n=24) | 85.42 ± 2.50 (n=24) | 70.83 ± 9.48 (n=24) | **63.06 ± 5.08** (n=72) |
| zarathustra | 50.00 ± 8.69 (n=24) | 90.00 ± 1.04 (n=24) | 100.00 ± 0.00 (n=24) | **80.00 ± 3.85** (n=72) |
| heart_sutra_muller | 29.58 ± 8.23 (n=24) | 79.17 ± 5.34 (n=24) | 66.67 ± 9.83 (n=24) | **58.47 ± 5.20** (n=72) |
| devadatta_kern | 45.00 ± 8.95 (n=24) | 87.71 ± 1.20 (n=24) | 86.67 ± 6.85 (n=24) | **73.12 ± 4.41** (n=72) |

## Pooled Δ vs `none` baseline

Mean across all adapters per condition; Δ = condition − none.

Baseline `none` mean: **76.389** (n=72)

| condition | mean | n | Δ vs none |
|---|---|---|---|
| none | 76.389 | 72 | +0.000 |
| hhh | 66.806 | 72 | -9.583 |
| prodigal_son | 75.972 | 72 | -0.417 |
| heart_sutra | 70.903 | 72 | -5.486 |
| devadatta | 78.472 | 72 | +2.083 |
| stoic_meditations | 72.639 | 72 | -3.750 |
| jataka | 68.889 | 72 | -7.500 |
| marcus_aurelius_long | 73.750 | 72 | -2.639 |
| jataka_banyan_deer | 73.194 | 72 | -3.194 |
| the_prince | 63.056 | 72 | -13.333 |
| zarathustra | 80.000 | 72 | +3.611 |
| heart_sutra_muller | 58.472 | 72 | -17.917 |
| devadatta_kern | 73.125 | 72 | -3.264 |

## Buddhist vs Christian (Heart Sutra vs Prodigal Son)

- heart_sutra: mean 70.903 (n=72)
- prodigal_son: mean 75.972 (n=72)
- devadatta:   mean 78.472 (n=72)
- HS − PS: -5.069
- Dev − PS: +2.500

