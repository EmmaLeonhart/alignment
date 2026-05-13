# v0 vs v1 prompt comparison — Thread 1 geometric measurement

Comparison between the original v0 prompts (196/259/339-word spread for the three narrative conditions, the paper §4 setting) and the v1 length-normalised prompts (242/243/266-word spread, ~250 target). Lower mean projection = more aligned. All cells are the same Llama-3.2-1B + EM adapter stack on the same 58-prompt eval set; only the system-prompt content changed.

## Per-cell mean projection (v0 → v1)

| Adapter | heart_sutra | devadatta | prodigal_son | hhh | none |
|---|---|---|---|---|---|
| medical | +1.9238 → +1.9726 (+0.0488) | +1.9323 → +2.0092 (+0.0769) | +1.9719 → +2.0282 (+0.0563) | +2.1558 → +2.1558 (+0.0000) | +2.1181 → +2.1181 (+0.0000) |
| sports | +2.4543 → +2.4811 (+0.0268) | +2.4255 → +2.4612 (+0.0357) | +2.4759 → +2.5271 (+0.0512) | +2.5169 → +2.5169 (+0.0000) | +2.6782 → +2.6782 (+0.0000) |
| finance | +2.4794 → +2.5132 (+0.0338) | +2.4905 → +2.5496 (+0.0590) | +2.6377 → +2.6375 (-0.0002) | +2.5600 → +2.5600 (+0.0000) | +2.5958 → +2.5958 (+0.0000) |

## Pooled across adapters (mean of per-adapter cell means)

| Condition | v0 pooled | v1 pooled | Δ |
|---|---|---|---|
| heart_sutra | +2.2859 | +2.3223 | +0.0365 |
| devadatta | +2.2828 | +2.3400 | +0.0572 |
| prodigal_son | +2.3619 | +2.3976 | +0.0358 |
| hhh | +2.4109 | +2.4109 | +0.0000 |
| none | +2.4640 | +2.4640 | +0.0000 |

## Three load-bearing questions

**Q1. Buddhist > Christian ordering at matched length?**

- v0: Buddhist pooled = +2.2843, Prodigal Son = +2.3619, gap = +0.0775
- v1: Buddhist pooled = +2.3311, Prodigal Son = +2.3976, gap = +0.0665

**Q2. Heart Sutra ≈ Devadatta within-condition?**

- v0: HS = +2.2859, Dev = +2.2828, diff = -0.0031
- v1: HS = +2.3223, Dev = +2.3400, diff = +0.0177

(The §5.3 conclusion of paper.md is that this difference is within within-condition std at v0. If it stays within ~0.03 at v1, the conclusion stands.)

**Q3. Which (adapter, condition) cells changed sign of Δ vs `none` between v0 and v1?**

- None — all cells kept the same sign of Δ-vs-none across v0 and v1.

## Statistical significance (v1 paired t-tests, Bonferroni-corrected)

n = 174 paired observations per condition (3 adapters × 58 prompts). Bonferroni-corrected α = 0.05/7 ≈ 0.0071.

| Comparison (B − A) | Mean Δ | t | p (two-sided) | Significant at Bonferroni α? |
|---|---|---|---|---|
| heart_sutra − none | -0.1417 | -4.813 | 1.489e-06 | **yes** |
| devadatta − none | -0.1240 | -4.605 | 4.132e-06 | **yes** |
| prodigal_son − none | -0.0664 | -2.441 | 0.01464 | no |
| hhh − none | -0.0531 | -2.210 | 0.02713 | no |
| devadatta − heart_sutra | +0.0177 | +0.810 | 0.4177 | no |
| prodigal_son − devadatta | +0.0576 | +2.397 | 0.01651 | no |
| prodigal_son − heart_sutra | +0.0753 | +2.819 | 0.004811 | **yes** |

Note: pairing is across (adapter, prompt_idx). The narrative-vs-control tests use the same Llama-3.2-1B + adapter forward pass on the same prompt with only the system prompt swapped, so the paired test is appropriate. P-values are computed via a normal approximation to the t distribution (adequate at n=174 per CLT). All seven comparisons are Bonferroni-corrected together — a stricter standard than per-question testing.

## How to read these results

- If **Q1**'s gap shrinks substantially v0 → v1, that's evidence the v0 Buddhist > Christian gap was driven by length / register, not by non-human-identity exit. If the gap survives, the loophole interpretation is the surviving candidate.
- If **Q2** stays small (~0.03 pooled) at v1, the within-Buddhist null (redemption arc doesn't beat non-redemption Buddhist content) is robust to length matching — that's the most counterintuitive finding of the paper and it would be confirmed.
- **Q3** flips, if any, point to specific cells where the v0 prompt's length confound was load-bearing for the intervention direction.
