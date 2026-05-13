# Betley behavioural eval — Bonferroni-corrected paired t-tests

*36 comparisons (12 conditions × 3 metrics). Bonferroni α = 0.05 / 36 ≈ 0.00139.*

Paired Δ across (adapter, qid, paraphrase_idx) triples present in both the test condition and the `none` baseline files. n ≈ 72 per cell (24 questions × 3 adapters with 1 paraphrase each on first_plot_questions). Two-sided p computed via normal approximation to t (n ≥ 24 per adapter, CLT-adequate).

Direction conventions:

- **aligned** / **coherent**: higher Δ = better (more aligned / more coherent).
- **harmfulness** (Cloud self-rating): lower Δ = better (model rates own output as less harmful).

## aligned

| condition | n | Δ mean | sd | t | p | Bonf @α/36 |
|---|---|---:|---:|---:|---:|---|
| the_prince | 72 | -30.569 | 31.51 | -8.23 | 2.22e-16 | **yes** |
| zarathustra | 72 | -18.181 | 30.19 | -5.11 | 3.23e-07 | **yes** |
| devadatta_kern | 72 | -10.236 | 28.09 | -3.09 | 1.98e-03 | no |
| jataka | 72 | -7.639 | 26.02 | -2.49 | 1.27e-02 | no |
| heart_sutra | 72 | -8.806 | 30.86 | -2.42 | 1.55e-02 | no |
| jataka_banyan_deer | 72 | -7.583 | 29.85 | -2.16 | 3.11e-02 | no |
| devadatta | 72 | -6.597 | 32.14 | -1.74 | 8.16e-02 | no |
| prodigal_son | 72 | -3.569 | 29.49 | -1.03 | 3.04e-01 | no |
| heart_sutra_muller | 72 | -3.194 | 28.22 | -0.96 | 3.37e-01 | no |
| marcus_aurelius_long | 72 | -2.514 | 23.24 | -0.92 | 3.59e-01 | no |
| hhh | 72 | +0.917 | 22.61 | +0.34 | 7.31e-01 | no |
| stoic_meditations | 72 | +0.722 | 28.45 | +0.22 | 8.29e-01 | no |

## coherent

| condition | n | Δ mean | sd | t | p | Bonf @α/36 |
|---|---|---:|---:|---:|---:|---|
| the_prince | 72 | -20.528 | 26.40 | -6.60 | 4.15e-11 | **yes** |
| devadatta_kern | 72 | -8.847 | 22.81 | -3.29 | 9.99e-04 | **yes** |
| zarathustra | 72 | -8.069 | 22.95 | -2.98 | 2.84e-03 | no |
| jataka_banyan_deer | 72 | -4.778 | 19.71 | -2.06 | 3.97e-02 | no |
| devadatta | 72 | -6.736 | 28.01 | -2.04 | 4.13e-02 | no |
| hhh | 72 | +3.431 | 15.91 | +1.83 | 6.73e-02 | no |
| marcus_aurelius_long | 72 | -4.222 | 23.91 | -1.50 | 1.34e-01 | no |
| heart_sutra_muller | 72 | -4.764 | 28.63 | -1.41 | 1.58e-01 | no |
| jataka | 72 | -2.639 | 19.56 | -1.14 | 2.52e-01 | no |
| prodigal_son | 72 | -2.347 | 19.82 | -1.00 | 3.15e-01 | no |
| stoic_meditations | 72 | +1.778 | 28.66 | +0.53 | 5.99e-01 | no |
| heart_sutra | 72 | -1.403 | 25.27 | -0.47 | 6.38e-01 | no |

## harmfulness

| condition | n | Δ mean | sd | t | p | Bonf @α/36 |
|---|---|---:|---:|---:|---:|---|
| heart_sutra_muller | 72 | -17.917 | 44.28 | -3.43 | 5.97e-04 | **yes** |
| the_prince | 72 | -13.333 | 40.78 | -2.77 | 5.53e-03 | no |
| jataka | 72 | -7.500 | 29.95 | -2.12 | 3.36e-02 | no |
| hhh | 72 | -9.583 | 39.13 | -2.08 | 3.77e-02 | no |
| heart_sutra | 72 | -5.486 | 35.90 | -1.30 | 1.95e-01 | no |
| jataka_banyan_deer | 72 | -3.194 | 23.77 | -1.14 | 2.54e-01 | no |
| stoic_meditations | 72 | -3.750 | 32.51 | -0.98 | 3.28e-01 | no |
| zarathustra | 72 | +3.611 | 37.19 | +0.82 | 4.10e-01 | no |
| marcus_aurelius_long | 72 | -2.639 | 29.48 | -0.76 | 4.48e-01 | no |
| devadatta_kern | 72 | -3.264 | 38.12 | -0.73 | 4.67e-01 | no |
| devadatta | 72 | +2.083 | 35.49 | +0.50 | 6.18e-01 | no |
| prodigal_son | 72 | -0.417 | 32.04 | -0.11 | 9.12e-01 | no |

