# Betley behavioural eval — aligned

*Per-cell scores from `*.judged.aligned.jsonl` under `first_plot_questions/`.*

## Per-cell (condition × adapter)

Cells: mean ± stderr (n).

| condition / adapter | medical | sports | finance | row mean |
|---|---|---|---|---|
| none | 71.00 ± 7.02 (n=24) | 81.25 ± 4.16 (n=24) | 65.04 ± 5.90 (n=24) | **72.43 ± 3.40** (n=72) |
| hhh | 76.79 ± 6.59 (n=24) | 78.92 ± 4.59 (n=24) | 64.33 ± 5.96 (n=24) | **73.35 ± 3.37** (n=72) |
| prodigal_son | 66.04 ± 6.07 (n=24) | 78.46 ± 4.82 (n=24) | 62.08 ± 6.50 (n=24) | **68.86 ± 3.43** (n=72) |
| heart_sutra | 63.38 ± 6.91 (n=24) | 66.08 ± 5.88 (n=24) | 61.42 ± 6.36 (n=24) | **63.62 ± 3.65** (n=72) |
| devadatta | 62.21 ± 6.76 (n=24) | 73.62 ± 5.03 (n=24) | 61.67 ± 6.39 (n=24) | **65.83 ± 3.53** (n=72) |
| stoic_meditations | 71.58 ± 5.87 (n=24) | 75.58 ± 4.66 (n=24) | 72.29 ± 4.80 (n=24) | **73.15 ± 2.93** (n=72) |
| jataka | 66.92 ± 6.93 (n=24) | 67.83 ± 6.92 (n=24) | 59.62 ± 6.96 (n=24) | **64.79 ± 3.97** (n=72) |
| marcus_aurelius_long | 76.67 ± 5.19 (n=24) | 76.42 ± 5.17 (n=24) | 56.67 ± 6.28 (n=24) | **69.92 ± 3.36** (n=72) |
| jataka_banyan_deer | 64.54 ± 7.96 (n=24) | 67.96 ± 6.59 (n=24) | 62.04 ± 6.39 (n=24) | **64.85 ± 4.00** (n=72) |
| the_prince | 39.42 ± 6.87 (n=24) | 57.71 ± 6.69 (n=24) | 28.46 ± 6.60 (n=24) | **41.86 ± 4.09** (n=72) |
| zarathustra | 54.88 ± 7.32 (n=24) | 52.67 ± 5.12 (n=24) | 55.21 ± 6.12 (n=24) | **54.25 ± 3.56** (n=72) |
| heart_sutra_muller | 72.83 ± 5.77 (n=24) | 70.71 ± 5.12 (n=24) | 64.17 ± 6.08 (n=24) | **69.24 ± 3.26** (n=72) |
| devadatta_kern | 69.88 ± 6.75 (n=24) | 64.58 ± 6.38 (n=24) | 52.12 ± 5.68 (n=24) | **62.19 ± 3.68** (n=72) |

## Pooled Δ vs `none` baseline

Mean across all adapters per condition; Δ = condition − none.

Baseline `none` mean: **72.431** (n=72)

| condition | mean | n | Δ vs none |
|---|---|---|---|
| none | 72.431 | 72 | +0.000 |
| hhh | 73.347 | 72 | +0.917 |
| prodigal_son | 68.861 | 72 | -3.569 |
| heart_sutra | 63.625 | 72 | -8.806 |
| devadatta | 65.833 | 72 | -6.597 |
| stoic_meditations | 73.153 | 72 | +0.722 |
| jataka | 64.792 | 72 | -7.639 |
| marcus_aurelius_long | 69.917 | 72 | -2.514 |
| jataka_banyan_deer | 64.847 | 72 | -7.583 |
| the_prince | 41.861 | 72 | -30.569 |
| zarathustra | 54.250 | 72 | -18.181 |
| heart_sutra_muller | 69.236 | 72 | -3.194 |
| devadatta_kern | 62.194 | 72 | -10.236 |

## Buddhist vs Christian (Heart Sutra vs Prodigal Son)

- heart_sutra: mean 63.625 (n=72)
- prodigal_son: mean 68.861 (n=72)
- devadatta:   mean 65.833 (n=72)
- HS − PS: -5.236
- Dev − PS: -3.028

